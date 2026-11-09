"""Import an independently validated real four-category EDA result ZIP.

The import is deliberately strict: a completed run with a matched real quality
reference is necessary, but not sufficient for scientific sign-off of the PDF.
Only report artifacts are imported. Raw JSONL and scratch SQLite files are never
accepted. This script works with the output of run_eda_all.py.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"All_Beauty", "Appliances", "Beauty_and_Personal_Care", "Electronics"}
EXPECTED_RUN_ID = "20261108T131934Z_6e74f578"
MAX_FILE_BYTES = 90 * 1024 * 1024  # avoid files rejected by GitHub


def fail(reason):
    raise SystemExit("EDA_IMPORT_NOT_ACCEPTED: " + reason)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        fail("Usage: python tools/import_eda_full_archive.py EDA_SEND_TO_CHAT_....zip")
    source = Path(sys.argv[1]).resolve()
    if not source.is_file():
        fail("The EDA ZIP file does not exist: " + str(source))
    with zipfile.ZipFile(source) as archive:
        broken = archive.testzip()
        if broken:
            fail("CRC failure: " + broken)
        members = archive.infolist()
        names = [x.filename for x in members]
        if len(names) != len(set(names)):
            fail("Duplicate archive member names")
        for x in members:
            p = PurePosixPath(x.filename)
            if x.filename.startswith('/') or '\\' in x.filename or '..' in p.parts or any(part == '' for part in p.parts):
                fail("Unsafe archive path: " + x.filename)
            if x.file_size > MAX_FILE_BYTES:
                fail("An individual result is larger than GitHub's comfortable limit: " + x.filename)
            if p.suffix.lower() in ('.sqlite', '.sqlite3', '.gz') or '_working.sqlite' in x.filename:
                fail("Raw/scratch data found in EDA ZIP: " + x.filename)
        needed = ['RUN_MANIFEST.json', 'outputs/full/RUN_STATUS.json', 'outputs/validation/real_validation.json']
        for name in needed:
            if name not in names:
                fail("Missing required evidence: " + name)
        receipt = json.loads(archive.read('RUN_MANIFEST.json'))
        status = json.loads(archive.read('outputs/full/RUN_STATUS.json'))
        gate = json.loads(archive.read('outputs/validation/real_validation.json'))
        if receipt.get('runner_id') != EXPECTED_RUN_ID:
            fail("Unexpected run id: " + str(receipt.get('runner_id')))
        if not (receipt.get('mode') == 'full' and receipt.get('completed_eda') is True
                and receipt.get('real_validation') is True and receipt.get('validation_status') == 'PASS_REAL_Q_MATCHED'):
            fail("Runner manifest does not confirm a verified full EDA run")
        if not (status.get('status') == 'complete' and status.get('scope') == 'entire_supplied_dataset_files'):
            fail("EDA engine status/scope does not confirm a complete real run")
        if gate.get('status') != 'PASS_REAL_Q_MATCHED':
            fail("Independent quality reference gate did not pass")
        cats = {p.parts[2] for p in map(PurePosixPath, names)
                if len(p.parts) >= 4 and p.parts[:2] == ('outputs', 'full') and p.parts[2] in CATEGORIES}
        if cats != CATEGORIES:
            fail("Missing category results: " + ', '.join(sorted(CATEGORIES - cats)))
        accepted = [m for m in members if not m.is_dir() and m.filename.startswith('outputs/')]
        if not accepted:
            fail("No result artifacts")
        target = ROOT/'reports'/'phase1'/'eda'
        if (target/'REAL_RUN_RECEIPT.json').exists():
            fail("Results already imported. Refusing silent replacement")
        # All validations are complete before making any changes.
        for item in accepted:
            rel = PurePosixPath(item.filename).relative_to('outputs')
            dest = target.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as src, dest.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src, dst, length=1024*1024)
        (target/'RUN_MANIFEST.json').write_bytes(archive.read('RUN_MANIFEST.json'))
        imported = sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file())
        evidence = {
            'run_id': receipt['runner_id'],
            'run_scope': status['scope'],
            'validation_status': gate['status'],
            'source_zip_name': source.name,
            'source_zip_sha256': digest_file(source),
            'imported_files': imported,
            'scientific_report_review': 'PENDING_UPDATE_REPORT_AND_MANUAL_REVIEW',
        }
        (target/'REAL_RUN_RECEIPT.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('REAL_EDA_ARTIFACTS_IMPORTED:', len(imported))
        print('The existing reports/phase1.pdf still needs scientific revision and is NOT approved for submission.')

if __name__ == '__main__':
    main()
