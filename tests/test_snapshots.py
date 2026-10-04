from unittest.mock import patch

from access_fixtures import AccessFixture
from aeroroute.normalization import pipeline
from aeroroute.paths import ROOT
from aeroroute.storage.artifacts import verify_dataset
from aeroroute.storage.identity import file_hash


class SnapshotCompatibilityTests(AccessFixture):
    def test_stored_transform_verification_does_not_require_current_code(self):
        legacy = {**pipeline.transform_identity(0), "code_sha256": "a" * 64}
        with patch.object(pipeline, "transform_identity", return_value=legacy):
            path = self.snapshot()
        files = {item: (file_hash(item), item.stat().st_mtime_ns)
                 for item in self.root.rglob("*") if item.is_file()}
        verified = verify_dataset(path)
        self.assertTrue(all(item["manifest"]["base"]["transform"] == legacy for item in verified["partitions"]))
        entry = {**verified["partitions"][0]["source"], "status": "validated"}
        with self.assertRaisesRegex(ValueError, "No compatible"):
            pipeline.normalize_month(self.root / "dataset/raw", entry, verified["root"], verify_only=True)
        self.assertEqual(files, {item: (file_hash(item), item.stat().st_mtime_ns)
                                 for item in self.root.rglob("*") if item.is_file()})

    def test_fingerprint_hashes_implementations_not_facades_or_cli(self):
        before = pipeline.transform_identity(0)
        seen = []
        def changed(path):
            relative = path.relative_to(ROOT / "aeroroute").as_posix()
            seen.append(relative)
            return "f" * 64 if relative == "normalization/rows.py" else file_hash(path)
        with patch.object(pipeline, "file_hash", side_effect=changed):
            after = pipeline.transform_identity(0)
        self.assertNotEqual(before["code_sha256"], after["code_sha256"])
        self.assertEqual(set(seen), set(pipeline.TRANSFORM_MODULES))
        self.assertTrue(all("/" in path for path in seen))
        self.assertFalse(any(path.startswith(("cli/", "catalog/", "outcomes/")) for path in seen))
        self.assertEqual(before.keys(), after.keys())
        self.assertEqual(pipeline.transform_identity(0), before)
