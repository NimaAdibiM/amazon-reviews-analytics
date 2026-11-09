"""Small deterministic tests for the outer one-click launcher only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_eda_all as runner


class DeliveryRunnerTests(unittest.TestCase):
    def test_reference_code_hashes_match_historical_quality_manifest(self):
        archive, manifest=runner.quality_reference()
        self.assertEqual(set(manifest['full']),set(runner.CATEGORIES))
        self.assertTrue(archive.is_file())

    def test_raw_preflight_rejects_missing_inputs(self):
        _,q=runner.quality_reference()
        with tempfile.TemporaryDirectory() as tmp:
            paths,bytes_count,guards,errors=runner.raw_preflight(Path(tmp),{'raw_dir':'data/raw'},q)
            self.assertEqual(len(errors),8)
            self.assertEqual(bytes_count,0)

    def test_raw_preflight_compares_all_eight_stored_sizes(self):
        _,q=runner.quality_reference()
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'data/raw';raw.mkdir(parents=True)
            for cat in runner.CATEGORIES:
                for role in ('reviews','metadata'):
                    name=('meta_' if role=='metadata' else '')+cat+'.jsonl.gz'
                    # Sparse test files simulate byte sizes; they must NEVER be parsed.
                    with (raw/name).open('wb') as f:
                        f.truncate(q['input_fingerprints'][cat][role]['bytes'])
            _,size,guards,errors=runner.raw_preflight(Path(tmp),{'raw_dir':'data/raw'},q)
            self.assertFalse(errors)
            self.assertEqual(len(guards),8)
            self.assertGreater(size,10*1024**3)
            with (raw/'Electronics.jsonl.gz').open('r+b') as f:
                f.truncate(24)
            _,_,_,errors=runner.raw_preflight(Path(tmp),{'raw_dir':'data/raw'},q)
            self.assertEqual(len(errors),1)
            self.assertIn('SIZE MISMATCH',errors[0])

    def test_packaging_failed_validation_cannot_be_reported_as_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);run=base/'run';run.mkdir()
            (run/'runner.log').write_text('example',encoding='utf-8')
            (run/'_working.sqlite').write_text('scratch',encoding='utf-8')
            (run/'full').mkdir();(run/'full'/'RUN_STATUS.json').write_text('{}',encoding='utf-8')
            dest=base/'out.zip'
            runner.pack_send_zip(dest,run,'unit-test','full',{'status':'FAIL'},runner.HERE/runner.Q_NAME,False)
            with zipfile.ZipFile(dest) as z:
                self.assertIsNone(z.testzip())
                manifest=json.loads(z.read('RUN_MANIFEST.json'))
                self.assertFalse(manifest['real_validation'])
                self.assertNotIn('outputs/_working.sqlite',z.namelist())
                self.assertIn('outputs/full/RUN_STATUS.json',z.namelist())


if __name__=='__main__':unittest.main()
