import ast
import importlib
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeroroute.paths import ACCESS_DIR, PROCESSED_DIR, RAW_DIR, ROOT


class ModuleBoundaryTests(unittest.TestCase):
    def test_dependency_directions(self):
        allowed = {
            "domain": {"domain"},
            "acquisition": {"acquisition", "paths"},
            "reference": {"reference", "storage"},
            "storage": {"storage", "domain"},
            "normalization": {"normalization", "domain", "acquisition", "reference", "storage"},
            "catalog": {"catalog", "domain", "storage", "reference"},
            "outcomes": {"outcomes", "domain", "storage", "catalog"},
            "simulation": {"simulation", "domain", "outcomes"},
            "verification": {"verification", "simulation", "domain"},
            "reporting": {"reporting", "domain", "storage", "catalog", "outcomes", "reference"},
        }
        for package, dependencies in allowed.items():
            for path in (ROOT / "aeroroute" / package).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    imports = []
                    if isinstance(node, ast.ImportFrom):
                        name = "." * node.level + (node.module or "")
                        imports = [importlib.util.resolve_name(name, f"aeroroute.{package}") if node.level else name]
                    elif isinstance(node, ast.Import):
                        imports = [item.name for item in node.names]
                    for name in imports:
                        with self.subTest(path=path.name, dependency=name):
                            if name.startswith("aeroroute."):
                                self.assertIn(name.split(".")[1], dependencies)
                            if package == "domain" and not name.startswith("aeroroute."):
                                self.assertIn(name.split(".")[0], sys.stdlib_module_names)

    def test_lightweight_imports_do_not_load_optional_libraries(self):
        code = '''
import importlib
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"pyarrow", "gymnasium", "airportsdata", "tzdata"}:
            raise AssertionError("Unexpected optional import: " + fullname)
sys.meta_path.insert(0, Block())
for name in ("aeroroute", "aeroroute.domain.records", "aeroroute.domain.evidence",
             "aeroroute.domain.scalars", "aeroroute.storage.identity", "aeroroute.storage.fields",
             "aeroroute.normalization.parsing", "aeroroute.normalization.times",
             "aeroroute.reference.timezones", "aeroroute.acquisition.bts", "aeroroute.cli.download"):
    importlib.import_module(name)
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_does_not_import_normalization(self):
        code = '''
import importlib
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("aeroroute.normalization", "aeroroute.preprocessing",
                                "aeroroute.acquisition", "aeroroute.cli", "gymnasium")):
            raise AssertionError("Unexpected runtime dependency: " + fullname)
sys.meta_path.insert(0, Block())
for name in ("aeroroute.storage.dataset", "aeroroute.catalog.services", "aeroroute.catalog.schedule",
             "aeroroute.outcomes.providers", "aeroroute.outcomes.replay",
             "aeroroute.simulation.environment", "aeroroute.verification.verifier"):
    importlib.import_module(name)
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_simulation_cli_reports_missing_rl_extra_without_importing_it_for_help(self):
        code = '''
import contextlib
import importlib.abc
import io
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"gymnasium", "numpy"}:
            raise ModuleNotFoundError("Optional RL dependency blocked", name=fullname.split(".")[0])
sys.meta_path.insert(0, Block())
from aeroroute.cli.simulate import main
with contextlib.redirect_stdout(io.StringIO()):
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0
try:
    main(["--snapshot", "missing", "--pilot", "missing", "--interface", "gymnasium"])
except ValueError as error:
    assert "RL extra" in str(error), str(error)
else:
    raise AssertionError("Missing RL dependencies should fail before artifact access")
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compatibility_exports_are_original_objects(self):
        pairs = {
            "aeroroute.records": ("aeroroute.domain.records",),
            "aeroroute.airport_timezones": ("aeroroute.reference.airports", "aeroroute.reference.timezones"),
            "aeroroute.flight_times": ("aeroroute.normalization.times", "aeroroute.domain.scalars"),
            "aeroroute.preprocessing_schema": ("aeroroute.normalization.parsing", "aeroroute.storage.schemas", "aeroroute.storage.fields"),
            "aeroroute.data_snapshots": ("aeroroute.storage.identity",),
            "aeroroute.data_access": ("aeroroute.storage.lookup", "aeroroute.storage.dataset", "aeroroute.storage.parquet", "aeroroute.domain.evidence"),
            "aeroroute.service_resolution": ("aeroroute.catalog.services",),
            "aeroroute.schedule": ("aeroroute.catalog.schedule",),
            "aeroroute.sampling": ("aeroroute.outcomes.draws", "aeroroute.outcomes.pools", "aeroroute.outcomes.providers", "aeroroute.outcomes.replay"),
            "aeroroute.pilot_report": ("aeroroute.reporting.pilot",),
        }
        for legacy, owners in pairs.items():
            facade = importlib.import_module(legacy)
            for owner in owners:
                module = importlib.import_module(owner)
                for name, value in vars(module).items():
                    if ((inspect.isfunction(value) or inspect.isclass(value)) and value.__module__ == owner
                            or name.isupper() and not name.startswith("_")):
                        with self.subTest(legacy=legacy, name=name):
                            self.assertIs(getattr(facade, name), value)
        import aeroroute
        from aeroroute.domain import records
        for name in aeroroute.__all__:
            self.assertIs(getattr(aeroroute, name), getattr(records, name))

    def test_preprocessing_and_verification_facades(self):
        from aeroroute import data_snapshots, preprocessing
        from aeroroute.acquisition import archives
        from aeroroute.normalization import datasets, pipeline, quality, rows
        from aeroroute.storage import artifacts
        for name in ("artifact_metadata", "contained_path", "verify_artifacts", "verify_dataset", "verify_derived", "verify_partition"):
            self.assertIs(getattr(data_snapshots, name), getattr(artifacts, name))
        for module, names in (
            (archives, ("read_source_rows",)),
            (datasets, ("publish_dataset", "split_configuration")),
            (pipeline, ("normalize_month", "flush_batch", "peak_rss_bytes", "transform_identity", "source_identity")),
            (quality, ("collect_quality", "new_quality", "fact_summary")),
            (rows, ("normalize_row", "mapping_zone", "outcome_category")),
        ):
            for name in names:
                self.assertIs(getattr(preprocessing, name), getattr(module, name))


class CliCompatibilityTests(unittest.TestCase):
    PAIRS = (
        (("-m", "aeroroute.cli.download"), (str(ROOT / "scripts/download_bts.py"),)),
        (("-m", "aeroroute.cli.normalize"), ("-m", "aeroroute.preprocessing")),
        (("-m", "aeroroute.cli.pilot_report"), ("-m", "aeroroute.pilot_report")),
        (("-m", "aeroroute.cli.build_data_access"), (str(ROOT / "scripts/build_data_access.py"),)),
    )

    def invoke(self, command, args, cwd, module_path=True):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if module_path:
            env["PYTHONPATH"] = str(ROOT)
        return subprocess.run([sys.executable, *command, *args], cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=30)

    def test_legacy_and_new_help_and_argument_errors(self):
        with tempfile.TemporaryDirectory() as cwd:
            for new, old in self.PAIRS:
                with self.subTest(command=new):
                    left = self.invoke(new, ["--help"], cwd)
                    right = self.invoke(old, ["--help"], cwd, module_path=old[0] == "-m")
                    self.assertEqual(left.returncode, 0, left.stderr)
                    self.assertEqual(right.returncode, 0, right.stderr)
                    self.assertEqual(left.stderr, "")
                    self.assertEqual(right.stderr, "")
                    self.assertEqual(set(re.findall(r"--[\w-]+", left.stdout)), set(re.findall(r"--[\w-]+", right.stdout)))
                    for command in (new, old):
                        result = self.invoke(command, ["--not-a-real-option"], cwd, module_path=command[0] == "-m")
                        self.assertEqual(result.returncode, 2)
                        self.assertIn("error:", result.stderr)
            self.assertEqual(list(Path(cwd).iterdir()), [])

    def test_error_boundaries_do_not_create_output(self):
        cases = (
            (self.PAIRS[0], ["--start-year", "2017"], "Error:"),
            (self.PAIRS[1], ["--periods", "2020-01", "--raw-dir", "missing"], "Preprocessing failed:"),
            (self.PAIRS[3], ["--snapshot", "missing.json"], "Data access failed:"),
        )
        with tempfile.TemporaryDirectory() as cwd:
            for commands, args, message in cases:
                for command in commands:
                    with self.subTest(command=command):
                        result = self.invoke(command, args, cwd, module_path=command[0] == "-m")
                        self.assertEqual(result.returncode, 1)
                        self.assertIn(message, result.stderr)
                        self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(list(Path(cwd).iterdir()), [])

    def test_cli_defaults_are_repository_anchored(self):
        class Parsed(Exception):
            pass
        from aeroroute.cli import build_data_access, download, normalize
        for module, defaults in (
            (download, {"output_dir": RAW_DIR}),
            (normalize, {"raw_dir": RAW_DIR, "output_dir": PROCESSED_DIR}),
            (build_data_access, {"output_dir": ACCESS_DIR}),
        ):
            def capture(parser, *args, **kwargs):
                for name, value in defaults.items():
                    self.assertEqual(parser.get_default(name), value)
                    self.assertTrue(value.is_absolute())
                raise Parsed
            with patch("argparse.ArgumentParser.parse_args", autospec=True, side_effect=capture):
                with self.assertRaises(Parsed):
                    module.main([])

    def test_script_wrappers_export_same_entry_points(self):
        from scripts import build_data_access, download_bts
        from aeroroute.cli import build_data_access as new_access, download
        from aeroroute.acquisition import bts
        self.assertIs(build_data_access.main, new_access.main)
        self.assertIs(download_bts.main, download.main)
        self.assertIs(download_bts.process_month, bts.process_month)
