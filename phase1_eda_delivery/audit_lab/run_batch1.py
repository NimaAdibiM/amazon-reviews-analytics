"""Repeatable independent batch-one audit diagnostics.

Running this does not certify missing original raw files or future modeling outcomes.
Usage: python -m audit_lab.run_batch1 --quality-archive /path/to/QUALITY_SEND...zip
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from .build_ledger import load_requirements
from .quality_reference import summarize_archive

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'first_batch': [*range(2, 14),31,52],
    'univariate': list(range(14,31)),
    'multivariate': list(range(32,41)),
    'cohort': list(range(41,52)),
    'final': list(range(53,58)),
}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quality-archive', type=Path)
    ap.add_argument('--evidence-dir',type=Path,default=ROOT/'audit_lab/evidence')
    args=ap.parse_args()
    folder=args.evidence_dir.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    ledger=load_requirements(ROOT/'EDA_PHASE1_REQUIREMENTS_E01_E57_FA.md')
    env=os.environ.copy()
    if args.quality_archive: env['EDA_QUALITY_ARCHIVE']=str(args.quality_archive.resolve())
    result=subprocess.run([sys.executable,'-m','unittest','audit_lab.tests.test_batch1','-v'],
                          cwd=ROOT,text=True,capture_output=True,env=env)
    log=result.stdout+result.stderr
    (folder/'BATCH1_TESTS.log').write_text(log,encoding='utf-8')
    match=re.search(r'Ran (\d+) tests?',log)
    tests=int(match.group(1)) if match else 0
    if result.returncode or not tests:
        print(log[-4000:],file=sys.stderr)
        raise RuntimeError('Foundational tests did not pass; no acceptance status emitted')
    reference=None
    if args.quality_archive:
        reference=summarize_archive(args.quality_archive,ROOT)
        (folder/'REAL_QUALITY_REFERENCE.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    evidence_map={
        2:('tested_on_synthetic_pending_source_identity',['BATCH1_TESTS','E02_END_TO_END','E02_REGRESSION_COMPARISON'],
           'Role/category guard implemented; matching names are not proof of source content.'),
        3:('partially_evidenced_waiting_for_raw_sha256',['REAL_QUALITY_REFERENCE'] if reference else [],
           'Archive provenance and code hashes verified; no local raw files for fresh hash recomputation.'),
        6:('partially_tested_not_full_audit',['BATCH1_TESTS'],
           'Zero denominator and a 3/12 rate tested; not all output rates enumerated.'),
        7:('partially_tested_not_full_audit',['BATCH1_TESTS'],
           'Strict JSON counterexamples tested; not every parser path compared.'),
        8:('partially_tested_not_full_audit',['BATCH1_TESTS'],
           'Blank/null/absent and invalid rating tested; all data types remain pending.'),
        11:('archive_reference_loaded_pending_source_match',['REAL_QUALITY_REFERENCE'] if reference else [],
            'Full Q01..Q25 output is available but all EDA metric definitions require cross-check.'),
        31:('partially_tested_not_full_audit',['BATCH1_TESTS'],
            'Empty and constant distribution tested; high dynamic range and exactness remain open.'),
        52:('not_audited_on_real_scale',[],
            'A disk-backed design is present; >70M record resource run has not been performed.'),
    }
    for entry in ledger:
        n=int(entry['id'][1:])
        if n in evidence_map:
            status,ids,note=evidence_map[n]
            entry.update(audit_verdict=status,evidence_ids=ids,notes=note)
        elif n in GROUPS['first_batch']:
            entry.update(audit_verdict='triaged_pending_independent_test',
                         notes='Relevant code identified; formal A/B comparison and acceptance evidence pending.')
    target={'schema_version':2,'total_requirements':57,
            'progress_policy':'No requirement is a final scientific pass without its own acceptance evidence.',
            'formal_pass_count':0,
            'e01_prior_status':'contract tested in previous E01 bundle, full real EDA still pending',
            'batch1_new_tests_passed':tests,
            'quality_reference_available':bool(reference),
            'quality_reference_status':reference['status'] if reference else 'not_given',
            'batch_groups':{k:[f'E{i:02d}' for i in v] for k,v in GROUPS.items()},
            'requirements':ledger}
    (folder/'REQUIREMENTS_TRIAGE_57.json').write_text(json.dumps(target,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Batch1 diagnostic successful: {tests} tests, 57 recorded requirements, full real EDA not run')


if __name__=='__main__':main()
