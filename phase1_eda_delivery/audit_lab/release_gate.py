"""E53–E57 independent run acceptance gate; no model evaluation.

Examples:
  python -m audit_lab.release_gate --run RUN_DIR --output release_check.json
  python -m audit_lab.release_gate --run FULL_RUN --quality-archive Q.zip --real --verify-source-hashes --output check.json

A synthetic run can pass internal-integrity checks but NEVER the --real gate.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from eda.artifacts import verify_inventory, _sha256_file
from .quality_reference import summarize_archive, CATS


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _rate_issues(obj: Any, where='', depth=0):
    """Check every fraction with explicit numerator/denominator, not vague averages."""
    if depth > 100:
        return [f'{where}:too_deep']
    errors = []
    if isinstance(obj, list):
        for i, x in enumerate(obj):
            errors.extend(_rate_issues(x, f'{where}[{i}]', depth+1))
    if isinstance(obj, dict):
        if 'numerator' in obj and 'denominator' in obj:
            n, d = obj['numerator'], obj['denominator']
            if type(n) is not int or type(d) is not int or n < 0 or d < 0 or n > d:
                errors.append(f'{where}:invalid_count_fraction')
            elif 'fraction' in obj:
                p = obj['fraction']
                if d == 0:
                    if p is not None:
                        errors.append(f'{where}:undefined_fraction_not_null')
                elif type(p) not in (int, float) or not math.isfinite(p) or not math.isclose(p, n/d, abs_tol=1e-12):
                    errors.append(f'{where}:wrong_fraction')
        for k, v in obj.items():
            errors.extend(_rate_issues(v, f'{where}.{k}', depth+1))
    return errors


def inspect_run(run, quality_archive=None, require_real=False, verify_source_hashes=False):
    root=Path(run)
    checks={}; failures=[]; pending=[]
    def check(name, condition, detail=''):
        checks[name] = {'passed': bool(condition), 'detail': detail}
        if not condition: failures.append(name + (':' + detail if detail else ''))

    artifact = verify_inventory(root)
    check('E55_artifact_hash_integrity', artifact['passed'], ';'.join(artifact['errors'][:8]))
    try:
        status=load(root/'RUN_STATUS.json'); man=load(root/'manifest.json')
        check('E55_run_status_complete',status.get('status') == 'complete')
        check('E55_manifest_status_agrees',man.get('status') == status.get('status') and man.get('run_id') == status.get('run_id'))
        project=Path(__file__).resolve().parents[1]
        code_hashes=man.get('code_sha256',{})
        check('E55_code_hashes_match_checked_out_sources',
              bool(code_hashes) and all((project / filename).is_file() and
                        _sha256_file(project / filename)==digest
                        for filename,digest in code_hashes.items()))
        check('E55_configuration_snapshot', (root/'RUN_CONFIG_SNAPSHOT.json').is_file() and
              sha256((root/'RUN_CONFIG_SNAPSHOT.json').read_bytes()).hexdigest() == man.get('config_sha256'))
        input_keys=set(man.get('inputs',{}))
        check('E54_four_category_inputs',input_keys == set(CATS),repr(sorted(input_keys)))
        check('E54_eight_role_sources',all(set(man.get('inputs',{}).get(c,{})) == {'reviews','metadata'} for c in CATS))
        check('E55_research_contract', (root/'RESEARCH_CONTRACT_E01.json').is_file() and
              (root/'RESEARCH_CONTRACT_E01.json').read_text(encoding='utf-8') ==
              (Path(__file__).resolve().parents[1]/'RESEARCH_CONTRACT_E01.json').read_text(encoding='utf-8'))
        for cat in CATS:
            if cat not in man.get('inputs', {}):continue
            try:
                rv=load(root/cat/'univariate.json')
                mult=load(root/cat/'multivariate.json')
                docs=[rv,mult]
                if (root/cat/'cohort.json').exists():docs.append(load(root/cat/'cohort.json'))
                readrv=man['inputs'][cat]['reviews']['counts']['objects']
                readmd=man['inputs'][cat]['metadata']['counts']['objects']
                check(f'E53_{cat}_input_to_outputs', readrv==rv['reviews']['total'] and readmd==rv['metadata']['total'])
                stars=rv['reviews']['rating_balance']['all_five_classes']
                star_total=sum(stars[str(s)]['numerator'] for s in range(1,6))
                valid=rv['reviews']['numeric']['rating']['valid']
                check(f'E53_{cat}_stars_sum',star_total == valid and valid==mult['review_relations']['valid_stars']['numerator'])
                for x in docs:
                    issues=_rate_issues(x,cat)
                    check(f'E53_{cat}_{"univariate" if x is rv else "multivariate" if x is mult else "cohort"}_denominators',not issues,';'.join(issues[:5]))
                if len(docs)==3:
                    for pol, d in docs[2]['policies'].items():
                        outcomes = d['mutually_exclusive_outcomes']
                        check(f'E53_{cat}_{pol}_cohort_conservation',sum(row['numerator'] for row in outcomes.values())==d['total'])
                for role in ('reviews','metadata'):
                    src=man['inputs'][cat][role]
                    check(f'E54_{cat}_{role}_eof',src.get('eof_verified') is True and src.get('scan_limit') in (0,None))
                    if verify_source_hashes:
                        p=Path(src['path'])
                        check(f'E54_{cat}_{role}_source_file_sha',p.is_file() and _sha256_file(p) == src.get('stored_file_sha256'),str(p))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                check('E53_'+cat+'_readable',False,type(exc).__name__ + ':'+str(exc)[:130])
        check('E56_reports_present',all((root/c/'UNIVARIATE_FA.md').is_file() and
              (root/c/'MULTIVARIATE_FA.md').is_file() for c in CATS) and
              (root/'CATEGORY_COMPARISON_FA.md').is_file())
        check('E57_split_only_design', (root/'SPLIT_DESIGN_E51.json').is_file() and
              load(root/'SPLIT_DESIGN_E51.json').get('status')=='phase1_design_only_no_split_executed')
        # Automatic text presence isn't proof that every figure has a valid scientific interpretation.
        pending.append('E56_each_chart_human_interpretation_review_not_automatically_certified')
        full = (man.get('mode')=='full' and man.get('input_role')=='dataset' and
                man.get('scope')=='entire_supplied_dataset_files')
        check('E54_real_dataset_scope',full or not require_real,
              f"mode={man.get('mode')};input_role={man.get('input_role')};scope={man.get('scope')}")
        q = man.get('quality_evidence',{})
        matches = (q.get('status')=='matched_all_inputs_and_metric_definitions' and
                   all(q.get('by_category',{}).get(c,{}).get('status')=='checked' for c in CATS))
        if require_real:
            check('E54_independent_raw_rehash_requested', verify_source_hashes)
            check('E54_Q_metrics_match_actual_input',matches,q.get('status','unavailable'))
        if quality_archive:
            qref=summarize_archive(quality_archive,Path(__file__).resolve().parents[1])
            check('E54_Q_archive_internally_verified',qref.get('status')=='verified_archive_internal_consistency_only')
            if require_real:
                check('E54_same_Q_archive_used',q.get('source_sha256')==qref['archive_file_sha256'])
                for c in CATS:
                    for role in ('reviews','metadata'):
                        expected=qref['categories'][c]['quality_scanned_content_sha256'][role]
                        actual=man.get('inputs',{}).get(c,{}).get(role,{}).get('scanned_content_sha256')
                        check(f'E54_{c}_{role}_quality_source_match',expected==actual)
        elif require_real:
            check('E54_Q_archive_provided',False)
        if not require_real:
            pending.append('E54_real_70m_run_not_verified_by_synthetic_gate')
        if not verify_source_hashes:
            pending.append('E54_source_bytes_not_independently_rehashed_in_this_gate')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        check('run_readable',False,type(exc).__name__+':'+str(exc)[:200])
    return dict(schema_version=1, run=str(root.resolve()), status='PASS_INTERNAL_ONLY' if not failures and not require_real else
                'PASS_REAL_Q_MATCHED' if not failures else 'FAIL',
                all_checks_passed=not failures, real_validation_claimed=require_real and not failures,
                checks=checks, failures=failures, pending=pending, artifact_integrity=artifact)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--quality-archive',type=Path)
    p.add_argument('--real',action='store_true')
    p.add_argument('--verify-source-hashes',action='store_true')
    a=p.parse_args(argv)
    result=inspect_run(a.run,a.quality_archive,a.real,a.verify_source_hashes)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result['status'], 'checks=',len(result['checks']), 'failed=',len(result['failures']))
    return 0 if result['all_checks_passed'] else 2

if __name__=='__main__':
    raise SystemExit(main())
