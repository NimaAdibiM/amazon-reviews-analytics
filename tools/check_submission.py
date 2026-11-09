from pathlib import Path
import json,zipfile,hashlib,sys
r=Path(__file__).resolve().parents[1]
necessary=['README.md','PROJECT_CARD.md','AI_USAGE.md','reports/phase1.pdf','reports/phase1.md','quality_audit.py','structure_audit.py','phase1_eda_delivery/eda/__main__.py','reports/phase1/eda/RUN_MANIFEST.json','reports/phase1/eda/full/RUN_STATUS.json','reports/phase1/eda/validation/real_validation.json']
missing=[p for p in necessary if not (r/p).is_file()]
try:
 m=json.loads((r/'reports/phase1/eda/RUN_MANIFEST.json').read_text(encoding='utf-8'))
 g=json.loads((r/'reports/phase1/eda/validation/real_validation.json').read_text(encoding='utf-8'))
 s=json.loads((r/'reports/phase1/eda/full/RUN_STATUS.json').read_text(encoding='utf-8'))
 valid=(m.get('validation_status')==g.get('status')=='PASS_REAL_Q_MATCHED' and s.get('status')=='complete' and g.get('all_checks_passed') is True)
except Exception: valid=False
has_quality=(r/'reports/phase1/quality/QUALITY_SEND_TO_CHAT_20261106T043348Z_4216fe6a.zip').is_file()
full_structure=bool(list((r/'reports/phase1/structure').glob('*.zip')))
result={'core_files_present':not missing,'missing':missing,'eda_real_q_matched':valid,'quality_evidence_present':has_quality,'independent_structure_archive_present':full_structure,'technical_package_sane':not missing and valid and has_quality,'scientific_signoff_requires_human_review':True,'note':'Independent structural evidence is not inside this version; course site instructions may add constraints.'}
print(json.dumps(result,ensure_ascii=False,indent=2))
sys.exit(0 if result['technical_package_sane'] else 2)
