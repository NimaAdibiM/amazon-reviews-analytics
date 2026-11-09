"""Attach a genuine full EDA archive without treating it as a validated analysis."""
from pathlib import Path
import json, zipfile, hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)!=2:
    raise SystemExit('Usage: python tools/add_eda_archive.py PATH_TO_EDA_ZIP')
src=Path(sys.argv[1]).resolve()
if not src.is_file(): raise SystemExit(f'Not a file: {src}')
with zipfile.ZipFile(src) as z:
    if z.testzip() is not None: raise SystemExit('Corrupt ZIP')
    names=z.namelist()
    if any(x.startswith('/') or '..' in Path(x).parts for x in names): raise SystemExit('Unsafe path')
    if not any('manifest' in x.lower() for x in names): raise SystemExit('Cannot find a manifest; manual verification required')
    if not any('All_Beauty/' in x for x in names) or not any('Electronics/' in x for x in names):
        raise SystemExit('Missing at least some expected category folders; manual verification required')
dst=ROOT/'reports/phase1/eda';dst.mkdir(parents=True,exist_ok=True)
target=dst/src.name
if target.exists(): raise SystemExit('Target exists; refusing overwrite')
import shutil
shutil.copy2(src,target)
(dst/'ARCHIVE_RECEIPT.json').write_text(json.dumps({'filename':src.name,'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'status':'ATTACHED_NOT_YET_SCIENTIFICALLY_REVIEWED'},indent=2),encoding='utf-8')
print('Archive added; revise reports/phase1.md and rebuild PDF after scientific review. NOT FINAL.')
