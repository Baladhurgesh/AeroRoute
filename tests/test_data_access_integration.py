import contextlib
import io
from pathlib import Path

from access_fixtures import AccessFixture
from aeroroute.data_snapshots import load_json, verify_derived
from scripts.build_data_access import main


class PilotPathTests(AccessFixture):
    def test_complete_verified_sample_and_replay_path(self):
        snapshot = self.snapshot()
        output = self.root / "access"
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            result = main(["--snapshot", str(snapshot), "--output-dir", str(output), "--stage", "pilot",
                           "--review-reference", "synthetic-fixture-review", "--schedule-periods", "2025-01",
                           "--fitting-years", "2020", "--min-support", "1", "--benchmark-draws", "3"])
        self.assertEqual(result, 0)
        report_path = Path(stream.getvalue().splitlines()[-1])
        manifest = verify_derived(report_path, "pilot")
        report = load_json(report_path / "pilot_report.json")
        self.assertEqual(report["sample"]["flight_date"][:4], "2020")
        self.assertEqual(report["replay"]["flight_date"][:4], "2025")
        self.assertTrue(report["scenario_stable_across_steps"])
        self.assertIn("cold_application_cache_seconds", report["benchmark"])
        self.assertEqual(report["benchmark"]["warm_draws"], 3)
        self.assertEqual(manifest["status"], "complete")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--snapshot", str(snapshot), "--verify-only", "--artifact", str(report_path)]), 0)

    def test_dry_run_creates_nothing(self):
        snapshot = self.snapshot()
        output = self.root / "access"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--snapshot", str(snapshot), "--output-dir", str(output), "--dry-run"]), 0)
        self.assertFalse(output.exists())

    def test_pilot_requires_explicit_review(self):
        snapshot = self.snapshot()
        with self.assertRaisesRegex(ValueError, "review"):
            main(["--snapshot", str(snapshot), "--output-dir", str(self.root / "access"), "--stage", "pilot",
                  "--schedule-periods", "2025-01", "--fitting-years", "2020"])
