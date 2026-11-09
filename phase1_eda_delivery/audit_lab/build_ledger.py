"""Create a complete 57-row triage register; status is NEVER inferred as pass.

Existing initial coverage signals are retained verbatim but are not audit verdicts.
"""
import json
from pathlib import Path
import re

HEAD = re.compile(r'^#### (E\d{2}) — (.+)$', re.M)
PROPERTIES = {
    'ماهیت': 'classification',
    'منبع': 'source',
    'تعریف': 'definition',
    'معیار پذیرش قابل‌آزمون': 'acceptance',
    'پوشش اولیه A': 'A_initial_coverage',
    'پوشش اولیه B': 'B_initial_coverage',
}

GROUPS = {
    'foundation': set(list(range(2,14)) + [31, 52]),
    'univariate': set(range(14, 31)),
    'multivariate': set(range(32, 41)),
    'research_cohort': set(range(41,52)),
    'final_validation': set(range(53,58)),
    'E01_completed': {1},
}
CANDIDATE_A = {
    'foundation': ['eda/input_data.py', 'eda/index.py', 'eda/univariate.py', 'eda/__main__.py', 'structure_audit.py', 'quality_audit.py'],
    'univariate': ['eda/univariate.py', 'eda/stats.py', 'eda/plots.py', 'eda/report.py'],
    'multivariate': ['eda/multivariate.py', 'eda/products.py', 'eda/plots.py', 'eda/report.py'],
    'research_cohort': ['eda/products.py', 'eda/index.py', 'eda/__main__.py'],
    'final_validation': ['tests/test_eda.py', 'tests/test_e01_contract.py', 'eda/report.py'],
    'E01_completed': ['eda/research_contract.py', 'tests/test_e01_contract.py'],
}
CANDIDATE_B = {
    'foundation': ['candidate_B/Univariate_EDA.py', 'candidate_B/audit_shared.py', 'candidate_B/eda_extensions.py'],
    'univariate': ['candidate_B/Univariate_EDA.py', 'candidate_B/eda_extensions.py'],
    'multivariate': ['candidate_B/eda_extensions.py'],
    'research_cohort': [],
    'final_validation': ['candidate_B/Univariate_EDA.py'],
    'E01_completed': ['candidate_B/Univariate_EDA.py'],
}


def load_requirements(path):
    text = Path(path).read_text(encoding='utf-8')
    matches = list(HEAD.finditer(text))
    if [m.group(1) for m in matches] != [f'E{i:02d}' for i in range(1, 58)]:
        raise ValueError('Requirements are missing, reordered or duplicated')
    entries = []
    for i, match in enumerate(matches):
        block = text[match.end():matches[i+1].start()] if i+1 < len(matches) else text[match.end():].split('\n## ', 1)[0]
        row = {'id': match.group(1), 'title': match.group(2)}
        for fa, field in PROPERTIES.items():
            pattern = rf'^\*\*{re.escape(fa)}:\*\*\s*(.+)$'
            found = re.search(pattern, block, re.M)
            if not found:
                raise ValueError(f'{row["id"]} property missing: {fa}')
            row[field] = found.group(1).strip()
        n = int(row['id'][1:])
        group = next(g for g, codes in GROUPS.items() if n in codes)
        row.update(group=group, candidate_A_locations=CANDIDATE_A[group], candidate_B_locations=CANDIDATE_B[group],
                   audit_verdict='not_yet_audited', acceptance_verified=False, evidence_ids=[],
                   notes='Initial coverage is a code inventory, not a scientific pass.')
        if n == 1:
            row.update(audit_verdict='passed_synthetic_evidence_pending_full_real',
                       evidence_ids=['E01_VALIDATION', 'E01_23_tests'],
                       notes='E01 contract implemented in prior bundle; real four-category EDA not rerun.')
        entries.append(row)
    return entries


def main():
    root = Path(__file__).resolve().parents[1]
    records = load_requirements(root/'EDA_PHASE1_REQUIREMENTS_E01_E57_FA.md')
    output = root/'audit_lab/evidence/REQUIREMENTS_TRIAGE_57.json'
    output.write_text(json.dumps({'schema_version': 1, 'total_requirements': len(records),
                                  'decision_policy': 'Only separately tested requirements may be marked accepted.',
                                  'items': records}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Written 57-row requirement ledger to', output)


if __name__ == '__main__':
    main()
