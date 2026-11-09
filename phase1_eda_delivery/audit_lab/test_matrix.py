"""Run isolated independent unittest suites and record completed/timeout results.

A timeout is UNKNOWN, never a pass. Use --all for the full regression list.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

SUITES = (
    'tests.test_eda.Mechanics',
    'tests.test_e01_contract',
    'audit_lab.tests.test_batch1',
    'audit_lab.tests.test_foundation_batch2',
    'audit_lab.tests.test_univariate_batch3',
    'audit_lab.tests.test_multivariate_batch4',
    'audit_lab.tests.test_cohort_batch5',
    'tests.test_eda.EndToEnd',
    'audit_lab.tests.test_release_batch6',
)
QUICK=(SUITES[0],SUITES[1],SUITES[2],SUITES[-1])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--all',action='store_true')
    p.add_argument('--timeout-seconds',type=int,default=100)
    args=p.parse_args(argv)
    if args.timeout_seconds<1: p.error('timeout must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
            'timeout_per_suite_seconds':args.timeout_seconds,
            'requested_suites':list(SUITES if args.all else QUICK),
            'execution_environment':sys.version,'suites':[]}
    for suite in result['requested_suites']:
        name=suite.replace('.','_')
        try:
            proc=subprocess.run([sys.executable,'-m','unittest',suite,'-v'],
                                capture_output=True,text=True,encoding='utf-8',errors='replace',
                                timeout=args.timeout_seconds)
            content=proc.stdout+'\n'+proc.stderr
            status='PASS' if proc.returncode==0 else 'FAIL'
            returncode=proc.returncode
        except subprocess.TimeoutExpired as exc:
            status='TIMEOUT_NOT_A_PASS';returncode=None
            out=exc.stdout or b'';err=exc.stderr or b''
            content=(out.decode('utf-8','replace') if isinstance(out,bytes) else out)+ '\n'+ \
                    (err.decode('utf-8','replace') if isinstance(err,bytes) else err)
        (args.out/(name+'.log')).write_text(content,encoding='utf-8')
        matches=re.findall(r'Ran (\d+) tests? in',content)
        completed=int(matches[-1]) if matches and status=='PASS' else 0
        skipped=re.findall(r'OK \(skipped=(\d+)\)',content)
        entry={'suite':suite,'status':status,'completed_tests':completed,
               'skipped':int(skipped[-1]) if skipped and status=='PASS' else None,
               'return_code':returncode,'log':name+'.log'}
        result['suites'].append(entry)
        print(suite,status,completed,flush=True)
        (args.out/'test_matrix.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result['total_confirmed_successful_tests']=sum(x['completed_tests'] for x in result['suites'])
    result['all_suites_completed']=all(x['status']=='PASS' for x in result['suites'])
    (args.out/'test_matrix.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return 0 if result['all_suites_completed'] else 2

if __name__=='__main__':raise SystemExit(main())
