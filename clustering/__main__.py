import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import LEVELS, ResourceLimit, atomic_json, read_json


def effective_policy(args):
    from .fixtures import default_policy
    policy=read_json(args.config) if getattr(args,'config',None) else default_policy()
    changes={}
    for arg,key in (('max_rows','max_rows'),('max_pairwise_bytes','max_pairwise_bytes'),('max_memory_mb','max_memory_mb')):
        value=getattr(args,arg,None)
        if value is not None:
            changes[key]=value;policy['plan']['resources'][key]=value
            if key=='max_rows':policy['plan']['sampling']['max_rows']=value
    seed=getattr(args,'seed',None)
    if seed is not None:policy['plan']['seeds']=[seed];policy['plan']['sampling']['seed']=seed;changes['seed']=seed
    if changes and getattr(args,'mode',None)=='real' and args.command=='run':raise ValueError('REAL_OVERRIDES_REQUIRE_NEW_REGISTERED_POLICY')
    return policy


def dispatch(args):
    policy=effective_policy(args)
    if args.command=='fixture':
        from .fixtures import make_fixture
        return make_fixture(args.output)
    if args.command in ('inspect','validate-input'):
        from .inputs import inspect_input
        result=inspect_input(args.input,args.mode,policy['plan']['resources'])
    elif args.command=='sample':
        from .inputs import collect
        from .policy import validate_policy
        validate_policy(policy,'synthetic');_,_,result,_=collect(args.input,args.mode,policy['plan'])
    elif args.command=='run':
        from .runner import run
        return run(args.input,args.output,policy,args.mode,args.config,args.profiles)
    elif args.command in ('compare','report'):
        from . import runner
        return getattr(runner,args.command)(args.input)
    elif args.command=='verify':
        from .verification import verify_experiment
        root=Path(args.input)
        if (root/'trial_manifest.json').exists():root=root.parent.parent
        result=verify_experiment(root)
    elif args.command=='seal-policy':
        from .policy import seal_policy
        return seal_policy(args.config,args.output)
    elif args.command=='prepare-pilot':
        from .pilot import prepare_pilot
        return prepare_pilot(args.source,args.cohorts,args.output,args.max_rows,args.strategy,args.seed,args.max_scan_rows,args.max_input_bytes)
    else:raise ValueError('UNKNOWN_COMMAND')
    if getattr(args,'output',None):atomic_json(Path(args.output),result)
    return result


def supervise(args):
    import psutil
    policy=effective_policy(args);limits=policy['plan']['resources'];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['STAGE_B_SUPERVISED']='1'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[name]=str(limits['threads'])
    command=[sys.executable,'-m','clustering',*sys.argv[1:]];started=time.perf_counter();peak=0;reason=None
    with (out/'worker.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try:rss=sum(p.memory_info().rss for p in [process,*process.children(recursive=True)] if p.is_running())
            except psutil.NoSuchProcess:rss=0
            peak=max(peak,rss)
            if rss>limits['max_memory_mb']*1024**2:reason='MEMORY'
            if time.perf_counter()-started>limits['max_wall_seconds']:reason='TIME'
            if reason:
                for p in process.children(recursive=True):
                    try:p.kill()
                    except psutil.NoSuchProcess:pass
                child.kill();child.wait();break
            time.sleep(.05)
    result={'status':'SKIPPED_RESOURCE_LIMIT' if reason else 'PASS' if child.returncode==0 else 'FAIL','evidence_level':LEVELS[args.mode],'reason':reason,'exit_code':child.returncode,'peak_sampled_process_tree_rss_bytes':peak,'elapsed_seconds':time.perf_counter()-started,'poll_seconds':.05,'bound_scope':'whole command including native model fit; short transient peaks may exceed sampled observation','model_resume':'completed trials only; interrupted fit restarts'}
    atomic_json(out/'supervisor_receipt.json',result)
    return result


def main():
    p=argparse.ArgumentParser(prog='python -m clustering');sub=p.add_subparsers(dest='command',required=True)
    for name in ('inspect','validate-input','sample','run'):
        q=sub.add_parser(name);q.add_argument('--input',required=True);q.add_argument('--config',required=True);q.add_argument('--mode',choices=['synthetic','real'],required=True);q.add_argument('--output',required=name in ('sample','run'))
        q.add_argument('--max-rows',type=int);q.add_argument('--max-pairwise-bytes',type=int);q.add_argument('--max-memory-mb',type=int);q.add_argument('--seed',type=int)
        if name=='run':q.add_argument('--profiles')
    for name in ('compare','report','verify'):
        q=sub.add_parser(name);q.add_argument('--input',required=True);q.add_argument('--output')
    q=sub.add_parser('fixture');q.add_argument('--output',required=True)
    q=sub.add_parser('seal-policy');q.add_argument('--config',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('prepare-pilot');q.add_argument('--source',required=True);q.add_argument('--cohorts',required=True,nargs='+');q.add_argument('--output',required=True);q.add_argument('--max-rows',type=int,default=800);q.add_argument('--strategy',choices=['natural','balanced'],default='balanced');q.add_argument('--seed',type=int,default=42);q.add_argument('--max-scan-rows',type=int,default=1000000);q.add_argument('--max-input-bytes',type=int,default=1073741824)
    args=p.parse_args()
    try:
        result=supervise(args) if args.command=='run' and os.environ.get('STAGE_B_SUPERVISED')!='1' else dispatch(args)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result.get('status') in ('FAIL','INCOMPLETE','SKIPPED_RESOURCE_LIMIT'):sys.exit(2)
    except (ValueError,FileNotFoundError,ResourceLimit) as exc:
        result={'status':'SKIPPED' if isinstance(exc,(FileNotFoundError,ResourceLimit)) else 'INVALID','evidence_level':LEVELS.get(getattr(args,'mode',None),'NOT_RUN'),'reason':str(exc)}
        if getattr(args,'output',None) and args.command in ('run','prepare-pilot'):
            Path(args.output).mkdir(parents=True,exist_ok=True);atomic_json(Path(args.output)/'blocked_receipt.json',result)
        print(json.dumps(result,ensure_ascii=False));sys.exit(2)


if __name__=='__main__':main()
