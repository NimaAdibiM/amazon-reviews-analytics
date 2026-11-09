"""One-command full four-category EDA runner with rigorous Q-link and send-to-chat ZIP.

This launcher does not patch an existing user's project. It executes only the
versioned implementation vendored beside this file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from uuid import uuid4
import zipfile

HERE = Path(__file__).resolve().parent
CATEGORIES = ('All_Beauty', 'Appliances', 'Beauty_and_Personal_Care', 'Electronics')
Q_NAME = 'QUALITY_REFERENCE_Q01_Q25.zip'
SOURCE_NAMES = ('quality_audit.py', 'structure_audit.py', 'PROJECT_CARD.md', 'RESEARCH_CONTRACT_E01.json')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def say(message, logfile=None):
    message = str(message)
    print(message, flush=True)
    if logfile:
        with logfile.open('a', encoding='utf-8') as handle:
            handle.write(message + '\n')


def load_settings():
    obj = json.loads((HERE/'settings.json').read_text(encoding='utf-8'))
    for k in ('raw_dir', 'min_free_gib', 'sample_size', 'seed', 'progress_every_lines', 'batch_size'):
        if k not in obj:
            raise ValueError(f'Invalid settings.json: missing {k}')
    if type(obj['min_free_gib']) not in (int,float) or not 0 < obj['min_free_gib'] < 1e4:
        raise ValueError('min_free_gib must be positive')
    if any(type(obj[k]) is not int for k in ('sample_size','seed','progress_every_lines','batch_size')):
        raise ValueError('sample_size, seed, progress_every_lines and batch_size must be integers')
    if not 0 <= obj['sample_size'] <= 10000 or obj['batch_size'] < 1 or obj['progress_every_lines'] < 0:
        raise ValueError('Invalid numeric setting')
    return obj


def quality_reference():
    path=HERE/Q_NAME
    if not path.is_file():
        raise FileNotFoundError(f'Missing required quality archive: {path}')
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad:
            raise ValueError(f'Quality ZIP failed CRC verification: {bad}')
        q=json.loads(z.read('RUN_MANIFEST.json'))
    if q.get('status')!='completed' or set(q.get('full',{}))!=set(CATEGORIES):
        raise ValueError('Quality ZIP is not a completed four-category full run')
    for name,key in (('quality_audit.py','code_sha256'),('structure_audit.py','structure_sha256')):
        if sha(HERE/name)!=q.get(key):
            raise ValueError(f'Versioned {name} does not match the Q01-Q25 reference ZIP')
    return path,q


def raw_preflight(project, settings, q):
    p=Path(settings['raw_dir'])
    raw_dir=p.resolve() if p.is_absolute() else (project/p).resolve()
    paths={}; errors=[]; stored=0;size_guard={}
    expected=q.get('input_fingerprints',{})
    if set(expected)!=set(CATEGORIES):
        raise ValueError('Q01-Q25 reference is missing input fingerprints for some categories')
    for category in CATEGORIES:
        paths[category]={}
        for role in ('reviews','metadata'):
            name=('meta_' if role=='metadata' else '')+category+'.jsonl.gz'
            path=raw_dir/name
            paths[category][role]=path
            reference=expected[category][role]
            if not path.is_file():
                errors.append(f'MISSING: {path}')
                size_guard[f'{category}/{role}']={'status':'missing','expected_bytes':reference['bytes']}
                continue
            size=path.stat().st_size
            status='size_matches_quality' if size==reference['bytes'] else 'size_mismatch'
            size_guard[f'{category}/{role}']={'status':status, 'bytes':size,'expected_bytes':reference['bytes'], 'path':str(path)}
            stored+=size
            if status!='size_matches_quality':
                errors.append(f'SIZE MISMATCH: {path}: found {size}, Q reference {reference["bytes"]}')
    return paths,stored,size_guard,errors


def run_command(cmd,cwd,log):
    say('COMMAND: '+' '.join('"'+x+'"' if ' ' in x else x for x in cmd),log)
    with subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', bufsize=1,
                          env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONUTF8':'1'}) as child:
        assert child.stdout
        for line in child.stdout:
            say(line.rstrip('\r\n'),log)
        code=child.wait()
    say('EXIT CODE: '+str(code),log)
    return code


def pack_send_zip(destination,run_root,run_id,mode,validation,quality_path,completed):
    """Capture all run outputs but never raw inputs, scratch databases or the ZIP itself."""
    files=[]
    for p in sorted(run_root.rglob('*')):
        if not p.is_file():continue
        if p.suffix in ('.sqlite','.sqlite3') or p.name.startswith('_working.sqlite'):
            continue
        rel=p.relative_to(run_root).as_posix()
        files.append((p,'outputs/'+rel))
    # Reference is small and is required to independently re-check source matching.
    files.append((quality_path,'reference/'+Q_NAME))
    receipt={
        'schema_version':1,'runner_id':run_id,'generated_utc':utcnow(),
        'mode':mode,'completed_eda':bool(completed),
        'validation_status':validation.get('status') if isinstance(validation,dict) else 'unavailable',
        'real_validation':mode=='full' and validation.get('status')=='PASS_REAL_Q_MATCHED',
        'evidence_only_no_raw_dataset_bytes':True,
        'included_run_paths':[arc for _,arc in files],
        'launcher_sha256':sha(HERE/'run_eda_all.py'),
        'quality_archive_sha256':sha(quality_path),
        'note':'PASS_REAL_Q_MATCHED is required for matched real-run integrity; it is not final approval of all 57 scientific requirements.'
    }
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_name(destination.name+'.part')
    tmp.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            z.writestr('RUN_MANIFEST.json', json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
            z.writestr('README_SEND_FA.txt',
                'Amazon Reviews 2023 — Phase 1 EDA results\n'
                'The outputs/ directory contains this exact local run and all four categories.\n'
                'See outputs/validation/real_validation.json and outputs/full/RUN_STATUS.json.\n'
                'Never treat PASS_INTERNAL_ONLY or FAIL as verified real analysis.\n'
                'The source datasets are NOT in this archive.\n'
                'Even PASS_REAL_Q_MATCHED is input/output consistency evidence, not full scientific E01–E57 approval.\n')
            for path,arc in files:
                z.write(path,arc)
        with zipfile.ZipFile(tmp) as z:
            broken=z.testzip()
            if broken or 'RUN_MANIFEST.json' not in z.namelist():
                raise ValueError(f'Generated ZIP failed CRC: {broken}')
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)
    return destination


def install_dependency_if_needed(log):
    try:
        import matplotlib
    except ImportError:
        say('Matplotlib not installed in this isolated environment. Installing pinned dependency...',log)
        code=run_command([sys.executable,'-m','pip','install','-r',str(HERE/'requirements.txt')], HERE, log)
        if code:
            raise RuntimeError('Cannot install matplotlib. Check network access or install requirements.txt into this environment.')
    else:
        # A version mismatch is better corrected than silently using an unknown build.
        required='3.10.8'
        if matplotlib.__version__ != required:
            say(f'Matplotlib {matplotlib.__version__} found; expected {required}. Installing pinned version.',log)
            code=run_command([sys.executable,'-m','pip','install','-r',str(HERE/'requirements.txt')],HERE,log)
            if code:raise RuntimeError('Failed to install pinned matplotlib version')


def full_run(project,check_only=False):
    settings=load_settings()
    quality,q=quality_reference()
    paths,compressed_bytes,guards,problems=raw_preflight(project,settings,q)
    print('\nFour category input preflight:',flush=True)
    for id,info in sorted(guards.items()):print(' ',id,info['status'],flush=True)
    if problems:
        raise RuntimeError('Input preflight failed. NO scan started:\n'+'\n'.join(problems))
    output_parent=project/'reports'/'phase1'/'eda_all_runs'
    output_parent.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(output_parent).free
    minimum=int(settings['min_free_gib']*1024**3)
    rough=compressed_bytes*4+sum(q['full'][c]['review_objects']+q['full'][c]['metadata_objects']
                                 for c in CATEGORIES)*640
    print(f'Compressed inputs: {compressed_bytes/1024**3:.1f} GiB; free output space: {free/1024**3:.1f} GiB;',
          f'heuristic scratch estimate: {rough/1024**3:.1f} GiB (NOT a bound)',flush=True)
    if free < minimum:
        raise RuntimeError(f'Insufficient disk space: below configured {settings["min_free_gib"]} GiB floor.')
    if check_only:
        print('PRECHECK_OK: 8 file names/sizes and reference code verified. This is not content-hash verification.')
        return 0
    if free < rough:
        print('DISK WARNING: free disk is below the heuristic estimate; run may exhaust disk. The estimate is not a bound.',flush=True)
        if not settings.get('proceed_below_heuristic',False):
            raise RuntimeError('Stopped before scan. Set proceed_below_heuristic=true in settings.json only after planning disk usage; do not treat heuristic as a guarantee.')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    run_root=output_parent/run_id
    run_root.mkdir(parents=True,exist_ok=False)
    log=run_root/'runner.log'
    config={
        'project_root':str(project), 'input_role':'dataset', 'timestamp_unit':'ms',
        'batch_size':settings['batch_size'], 'progress_every_lines':settings['progress_every_lines'],
        'source_provenance':{'dataset':'Amazon Reviews 2023','reference':'Q01-Q25 completed four-category archive'},
        'datasets':{c:{f'{role}_path':str(paths[c][role]) for role in ('reviews','metadata')} for c in CATEGORIES}}
    config_path=run_root/'RUNNER_CONFIG.json'
    dump(config_path,config)
    dump(run_root/'PREFLIGHT.json',{'timestamp':utcnow(),'status':'passed_filename_and_stored_size_only',
         'input_size_guards':guards,'stored_input_bytes':compressed_bytes,'free_disk_bytes':free,
         'scratch_heuristic_bytes_not_guaranteed':rough,'quality_archive_sha256':sha(quality),
         'note':'Only the subsequent scan plus release gate establish a matched real run.'})
    validation={}; completed=False; stage='start'
    try:
        install_dependency_if_needed(log)
        stage='plan'
        plan_cmd=[sys.executable,'-m','eda','--config',str(config_path),'--mode','plan',
                  '--expected-records',str(sum(q['full'][c]['review_objects']+q['full'][c]['metadata_objects'] for c in CATEGORIES)),
                  '--min-free-gib',str(settings['min_free_gib']),'--out',str(run_root/'plan')]
        if run_command(plan_cmd,HERE,log):raise RuntimeError('EDA plan stage failed')
        stage='full_eda'
        full_cmd=[sys.executable,'-m','eda','--config',str(config_path),'--mode','full',
                  '--quality-archive',str(quality),'--conflict-policy','exclude',
                  '--min-free-gib',str(settings['min_free_gib']),'--sample-size',str(settings['sample_size']),
                  '--seed',str(settings['seed']),'--out',str(run_root/'full')]
        if run_command(full_cmd,HERE,log):raise RuntimeError('Four-category EDA scan failed; see full/RUN_STATUS.json')
        completed=True
        stage='release_gate'
        gate_path=run_root/'validation'/'real_validation.json'
        gate_cmd=[sys.executable,'-m','audit_lab.release_gate','--run',str(run_root/'full'),
                  '--quality-archive',str(quality),'--real','--verify-source-hashes','--output',str(gate_path)]
        code=run_command(gate_cmd,HERE,log)
        validation=json.loads(gate_path.read_text(encoding='utf-8')) if gate_path.is_file() else {'status':'unavailable'}
        if code or validation.get('status')!='PASS_REAL_Q_MATCHED':
            say('REAL VALIDATION DID NOT PASS. Results will be packaged as UNVERIFIED evidence.',log)
        else:
            say('PASS_REAL_Q_MATCHED: full scan and reference checks verified. Methodological acceptance still pending.',log)
    except Exception as exc:
        say(f'FAILED at stage {stage}: {type(exc).__name__}: {exc}',log)
        validation={'status':'FAIL','reason':f'{stage}:{type(exc).__name__}:{exc}'}
        dump(run_root/'validation'/'real_validation.json',validation)
    finally:
        receipt={'stage':stage,'generated_utc':utcnow(),'eda_completed':completed,
                 'validation_status':validation.get('status','unavailable'),
                 'send_status':'VERIFIED_REAL' if validation.get('status')=='PASS_REAL_Q_MATCHED' else 'NOT_VERIFIED'}
        dump(run_root/'RUNNER_RECEIPT.json',receipt)
        send_root=project/'reports'/'phase1'/'eda_send_to_chat'
        zip_path=send_root/('EDA_SEND_TO_CHAT_'+run_id+'.zip')
        say('Packaging complete run evidence (excluding raw data and scratch SQLite)...',log)
        pack_send_zip(zip_path,run_root,run_id,'full',validation,quality,completed)
        say('SEND THIS ZIP: '+str(zip_path),log)
        say('RESULT: '+receipt['send_status'],log)
    return 0 if validation.get('status')=='PASS_REAL_Q_MATCHED' else 2


def self_test(project):
    install_root=project/'reports'/'phase1'/'eda_selftest_runs'
    ident=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    place=install_root/ident
    place.mkdir(parents=True,exist_ok=False)
    log=place/'runner.log'
    quality,q=quality_reference()
    install_dependency_if_needed(log)
    from tests.fixture import generate
    config=generate(place/'fixture')
    cmd=[sys.executable,'-m','eda','--config',str(config),'--mode','synthetic','--out',str(place/'full')]
    if run_command(cmd,HERE,log):raise RuntimeError('Self-test EDA failed')
    gate=place/'validation'/'real_validation.json'
    code=run_command([sys.executable,'-m','audit_lab.release_gate','--run',str(place/'full'),
                      '--output',str(gate)],HERE,log)
    val=json.loads(gate.read_text(encoding='utf-8'))
    if code or val.get('status')!='PASS_INTERNAL_ONLY':
        raise RuntimeError(f'Self-test integrity check failed: {val.get("status")}')
    path=project/'reports'/'phase1'/'eda_send_to_chat'/('EDA_SELFTEST_SEND_'+ident+'.zip')
    pack_send_zip(path,place,ident,'synthetic',val,quality,True)
    say('SELF_TEST_PASS_INTERNAL_ONLY: '+str(path),log)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--check-only',action='store_true',help='Validate raw file names, sizes, and free disk only')
    p.add_argument('--self-test',action='store_true',help='Run synthetic EDA + packaging; never claims a real pass')
    args=p.parse_args()
    project=args.project_root.resolve()
    if not project.is_dir():raise FileNotFoundError(f'Project root does not exist: {project}')
    if args.self_test:return self_test(project)
    return full_run(project,args.check_only)


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,KeyError,zipfile.BadZipFile) as exc:
        print(f'RUNNER ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(1)
