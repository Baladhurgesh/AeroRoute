from dataclasses import replace
from unittest.mock import patch

from access_fixtures import AccessFixture, flight_row
from aeroroute.storage.dataset import DatasetStore
from aeroroute.storage.lookup import build_lookup
from aeroroute.storage.artifacts import contained_path, verify_dataset, verify_derived
from aeroroute.storage.identity import load_json


class DataAccessTests(AccessFixture):
    def open_store(self, rows=None, **options):
        snapshot = self.snapshot(rows)
        lookup = build_lookup(snapshot, self.root / "lookups")
        store = DatasetStore(snapshot, lookup, **options)
        self.addCleanup(store.close)
        return store

    def test_inventory_identity_is_not_normalized_snapshot_identity(self):
        store = self.open_store()
        ref = next(store.source_refs("historical"))
        self.assertNotEqual(ref.dataset_version, store.snapshot["manifest"]["dataset_version"])
        evidence = store.get(ref)
        self.assertEqual(evidence.flight["marketing_flight_number"], "0012")
        self.assertEqual(evidence.source, ref)
        with self.assertRaises(TypeError):
            evidence.flight["cancelled"] = True
        self.assertIsNotNone(evidence.flight["actual_arrival"].value)
        with self.assertRaises(ValueError):
            store.get(replace(ref, dataset_version="a" * 64))

    def test_point_lookup_and_cache_do_not_scan_month(self):
        store = self.open_store()
        refs = tuple(store.source_refs("historical"))
        with patch("pyarrow.parquet.ParquetFile.read", side_effect=AssertionError("monthly scan")), \
                patch("pyarrow.parquet.ParquetFile.iter_batches", side_effect=AssertionError("batch scan")):
            store.get(refs[0])
            store.get(refs[1])
            store.get(refs[0])
        self.assertEqual(store.metrics["row_groups_read"], 1)
        self.assertEqual(store.metrics["cache_hits"], 2)

    def test_zero_byte_cache_and_handle_limit(self):
        store = self.open_store(cache_bytes=0, max_handles=1)
        refs = tuple(store.source_refs("all"))
        for ref in refs:
            store.get(ref)
        self.assertEqual(store.cache_size_bytes, 0)
        self.assertLessEqual(len(store.handles), 1)
        self.assertEqual(store.metrics["row_groups_read"], len(refs))

    def test_stops_are_indexed_and_original_ordinals_preserved(self):
        rows = [flight_row(), flight_row("2020-01-02", Diverted="1", DivReachedDest="1",
                 DivAirportLandings="2", Div1Airport="ORD", Div1AirportID="13930",
                 Div1WheelsOn="1200", Div2Airport="PHL", Div2AirportID="14100", Div2WheelsOn="1500",
                 DivActualElapsedTime="300", DivArrDelay="15")]
        store = self.open_store(rows)
        first, second = tuple(store.source_refs("historical"))
        self.assertEqual(store.get(first).stops, ())
        evidence = store.get(second)
        self.assertEqual(evidence.source.source_row_id, 2)
        self.assertEqual([stop["stop_index"] for stop in evidence.stops], [1, 2])
        self.assertEqual([stop["airport"] for stop in evidence.stops], ["ORD", "PHL"])

    def test_unavailable_split_and_source_fail(self):
        store = self.open_store([flight_row()])
        with self.assertRaises(ValueError):
            tuple(store.source_refs("evaluation"))
        with self.assertRaises(ValueError):
            store.get(replace(next(store.source_refs("historical")), source_row_id=9))

    def test_corrupt_lookup_is_not_reused(self):
        snapshot = self.snapshot()
        lookup = build_lookup(snapshot, self.root / "lookups")
        with (lookup / "lookup.sqlite").open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaises(ValueError):
            DatasetStore(snapshot, lookup)
        with self.assertRaises(ValueError):
            build_lookup(snapshot, self.root / "lookups")

    def test_unchanged_lookup_is_reused(self):
        snapshot = self.snapshot()
        first = build_lookup(snapshot, self.root / "lookups")
        before = (first / "lookup.sqlite").stat().st_mtime_ns
        self.assertEqual(build_lookup(snapshot, self.root / "lookups"), first)
        self.assertEqual((first / "lookup.sqlite").stat().st_mtime_ns, before)
        self.assertEqual(load_json(first / "manifest.json")["status"], "complete")

    def test_wrong_physical_row_is_rejected(self):
        store = self.open_store()
        first, second = tuple(store.source_refs("historical"))
        other, _ = store.raw(store.get(second).flight["source_record_id"])
        with patch.object(store.cache, "row", return_value=other):
            with self.assertRaisesRegex(ValueError, "locator"):
                store.get(first)

    def test_row_groups_not_csv_ordinals_are_physical_locations(self):
        snapshot = self.snapshot([flight_row(f"2020-01-{day:02d}") for day in range(1, 6)], batch_size=2)
        store = DatasetStore(snapshot, build_lookup(snapshot, self.root / "lookups"))
        self.addCleanup(store.close)
        refs = tuple(store.source_refs("historical"))
        for index in (4, 0, 3, 1, 2):
            self.assertEqual(store.get(refs[index]).source.source_row_id, index + 1)
        self.assertEqual(store.metrics["row_groups_read"], 3)
        self.assertEqual(store.metrics["cache_hits"], 2)

    def test_five_stops_and_group_eviction(self):
        changes = {"Diverted": "1", "DivReachedDest": "0", "DivAirportLandings": "5"}
        for index in range(1, 6):
            changes.update({f"Div{index}Airport": "ORD", f"Div{index}AirportID": "13930"})
        store = self.open_store([flight_row(**changes)], cache_bytes=1, max_handles=1)
        evidence = store.get(next(store.source_refs("historical")))
        self.assertEqual([stop["stop_index"] for stop in evidence.stops], list(range(1, 6)))
        self.assertTrue(all(stop["landing"].value is None for stop in evidence.stops))
        self.assertEqual(store.cache_size_bytes, 0)
        self.assertEqual(len(store.handles), 1)

    def test_unknown_lookup_version_is_rejected(self):
        snapshot = self.snapshot()
        lookup = build_lookup(snapshot, self.root / "lookups")
        manifest = load_json(lookup / "manifest.json")
        manifest["identity"]["lookup_version"] = 999
        with patch("aeroroute.storage.artifacts.load_json", return_value=manifest), \
                patch("aeroroute.storage.artifacts.digest", return_value=lookup.name):
            with self.assertRaises(ValueError):
                verify_derived(lookup, "lookup")

    def test_path_cannot_escape_artifact_root(self):
        with self.assertRaises(ValueError):
            contained_path(self.root, "../outside")
        with self.assertRaises(ValueError):
            contained_path(self.root, str(self.root / "absolute"))

    def test_corrupt_snapshot_fails_verification(self):
        snapshot = self.snapshot()
        inventory = snapshot.parent / "historical_source_inventory.json"
        with inventory.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaises(ValueError):
            verify_dataset(snapshot)
