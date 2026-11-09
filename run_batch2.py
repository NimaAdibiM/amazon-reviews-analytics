"""Repeatable second foundation audit round; each suite has an independent log.

usage: python -m audit_lab.run_batch2 --quality-archive ORIGINAL_Q.zip
The full 70-million-row EDA is *not* invoked without its raw inputs.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from .benchmark_small import measure
from .finalize_batch2 import main as finalize
from .quality_comparison import build

ROOT=Path(__file__).resolve().parents[1]
SUITES=('tests.test_e01_contract','tests.test_eda','audit_lab.tests.test_batch1',
        'audit_lab.tests.test_foundation_batch2')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-archive',type=Path,required=True)
    p.add_argument('--timeout-per-suite',type=int,default=120)
    a=p.parse_args()
    folder=ROOT/'audit_lab/evidence/batch2'; folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,EDA_QUALITY_ARCHIVE=str(a.quality_archive.resolve()))
    results=[]
    for suite in SUITES:
        print('Executing',suite,flush=True)
        log=folder/(suite.replace('.','_')+'.log')
        try:
            run=subprocess.run([sys.executable,'-m','unittest',suite,'-v'],cwd=ROOT,env=env,
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                               encoding='utf-8',errors='replace',timeout=a.timeout_per_suite)
            log.write_text(run.stdout,encoding='utf-8')
            m=re.search(r'Ran (\d+) tests? in [\d.]+s',run.stdout)
            n=int(m.group(1)) if m else None
            results.append(dict(suite=suite,status='passed' if run.returncode == 0 else 'failed',
                                returncode=run.returncode,tests=n,log=str(log.relative_to(ROOT))))
        except subprocess.TimeoutExpired as exc:
            log.write_text((exc.stdout or b'').decode('utf-8','replace') if isinstance(exc.stdout,bytes) else (exc.stdout or ''),encoding='utf-8')
            results.append(dict(suite=suite,status='timeout',tests=None,log=str(log.relative_to(ROOT))))
        print(results[-1],flush=True)
    state={'status':'all_independent_suites_passed' if all(r['status']=='passed' for r in results)
           else 'one_or_more_suites_not_passed','suites':results,
           'test_count':sum(r['tests'] or 0 for r in results),
           'warning':'No full raw-data EDA was executed. Tests use fixtures plus the archived quality evidence.'}
    (folder/'BATCH2_TEST_SUMMARY.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    if state['status']!='all_independent_suites_passed':
        raise SystemExit('No acceptance report will be regenerated on failed/incomplete tests')
    build(a.quality_archive,ROOT/'audit_lab/evidence/batch2_real_q')
    (folder/'E52_MICRO_BENCHMARK.json').write_text(json.dumps(measure(),indent=2)+'\n',encoding='utf-8')
    finalize()
    print('Done:',state['test_count'],'tests independently passed.')

if __name__=='__main__':main()
