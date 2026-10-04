import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from aeroroute.acquisition.archives import read_source_rows
from aeroroute.storage.identity import digest, file_hash
from aeroroute.storage.artifacts import verify_partition
from aeroroute.normalization.pipeline import normalize_month
from aeroroute.normalization.datasets import publish_dataset, split_configuration
from aeroroute.reporting.pilot import build_report


class PilotIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "processed"

    def source(self, changes=None, extra_rows=None):
        row = {
            "Year": "2020", "Month": "1", "FlightDate": "2020-01-01",
            "Origin": "SFO", "OriginAirportID": "14771", "OriginStateName": "California",
            "Dest": "JFK", "DestAirportID": "12478", "DestStateName": "New York",
            "Marketing_Airline_Network": "UA", "Operating_Airline ": "UA",
            "Flight_Number_Marketing_Airline": "0012", "Flight_Number_Operating_Airline": "0012",
            "CRSDepTime": "0800", "CRSArrTime": "1600", "CRSElapsedTime": "300.00",
            "DepDelay": "15.00", "ActualElapsedTime": "300.00", "ArrDelay": "15.00",
            "DepTime": "0815", "ArrTime": "1615", "Cancelled": "0.00", "Diverted": "0.00",
            "DivAirportLandings": "0", "Duplicate": "N", "": "",
        }
        row.update(changes or {})
        rows = [row, *(extra_rows or [])]
        columns = list(dict.fromkeys(key for current in rows for key in current))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        path = self.raw / "2020-01.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("flights.csv", stream.getvalue())
        return {
            "year": 2020, "month": 1, "path": path.name, "sha256": file_hash(path),
            "csv_member": "flights.csv", "row_count": len(rows), "size_bytes": path.stat().st_size,
            "cancelled_rows": sum(current.get("Cancelled") in {"1", "1.00"} for current in rows),
            "diverted_rows": sum(current.get("Diverted") in {"1", "1.00"} for current in rows),
            "status": "validated", "columns": columns,
        }

    def run_month(self, entry):
        return normalize_month(self.raw, entry, self.output, batch_size=1)

    def flights(self, result):
        return pq.ParquetFile(result["path"] / "flights.parquet").read().to_pylist()

    def test_preservation_and_independent_arrival(self):
        entry = self.source({"ActualElapsedTime": "", "DepDelay": "", "DepTime": ""})
        result = self.run_month(entry)
        rows = self.flights(result)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_row_id"], 1)
        self.assertEqual(rows[0]["marketing_flight_number"], "0012")
        self.assertIsNone(rows[0]["actual_departure"]["value"])
        self.assertIsNotNone(rows[0]["actual_arrival"]["value"])
        self.assertEqual(rows[0]["actual_arrival"]["method"], "arrival_delay_only")
        verify_partition(result["path"])

    def test_invalid_cell_preserved_as_diagnostic(self):
        entry = self.source({"ActualElapsedTime": "15.7"})
        row = self.flights(self.run_month(entry))[0]
        self.assertIsNone(row["actual_elapsed_minutes"])
        self.assertIsNotNone(row["actual_arrival"]["value"])
        self.assertTrue(any(issue["raw_value"] == "15.7" for issue in row["cell_issues"]))

    def test_cancellation_and_source_ordinals_preserved_across_batches(self):
        cancelled = {"Year": "2020", "Month": "1", "FlightDate": "2020-01-01", "Cancelled": "1", "Diverted": "0",
                     "Origin": "SFO", "OriginAirportID": "14771", "Dest": "JFK", "DestAirportID": "12478",
                     "CRSDepTime": "0800", "CRSArrTime": "1600", "CRSElapsedTime": "300"}
        entry = self.source(extra_rows=[cancelled])
        result = self.run_month(entry)
        rows = self.flights(result)
        self.assertEqual([row["source_row_id"] for row in rows], [1, 2])
        self.assertTrue(rows[1]["cancelled"])
        self.assertIsNone(rows[1]["actual_arrival"]["value"])
        self.assertIsNotNone(rows[1]["scheduled_departure"]["value"])
        self.assertNotEqual(rows[0]["source_record_id"], rows[1]["source_record_id"])

    def test_resolved_diverted_arrival_does_not_require_stop_timing(self):
        entry = self.source({"Diverted": "1", "DivReachedDest": "1", "DivAirportLandings": "1",
                             "Div1Airport": "ORD", "Div1AirportID": "13930", "Div1WheelsOn": "",
                             "DivActualElapsedTime": "", "DivArrDelay": "15", "ActualElapsedTime": "", "ArrDelay": ""})
        result = self.run_month(entry)
        self.assertIsNotNone(self.flights(result)[0]["actual_arrival"]["value"])
        stops = pq.ParquetFile(result["path"] / "diversion_stops.parquet").read().to_pylist()
        self.assertEqual(len(stops), 1)
        self.assertIsNone(stops[0]["landing"]["value"])

    def test_stranded_diversion_endpoint_not_confused_with_arrival(self):
        entry = self.source({"Diverted": "1", "DivReachedDest": "0", "DivAirportLandings": "2",
                             "Div1Airport": "ORD", "Div1AirportID": "13930", "Div1WheelsOn": "1200",
                             "Div2Airport": "PHL", "Div2AirportID": "14100", "Div2WheelsOn": "1500"})
        row = self.flights(self.run_month(entry))[0]
        self.assertEqual(row["endpoint_airport_id"], 14100)
        self.assertIsNone(row["actual_arrival"]["value"])
        self.assertEqual(row["outcome_category"], "diverted_stranded")

    def test_rerun_reuses_without_modifying_partition(self):
        entry = self.source()
        first = self.run_month(entry)
        artifact = first["path"] / "flights.parquet"
        before = artifact.stat().st_mtime_ns
        second = self.run_month(entry)
        self.assertTrue(second["reused"])
        self.assertEqual(first["path"], second["path"])
        self.assertEqual(artifact.stat().st_mtime_ns, before)

    def test_changed_source_creates_separate_partition(self):
        first = self.run_month(self.source())
        second = self.run_month(self.source({"ArrDelay": "25", "ActualElapsedTime": "310", "ArrTime": "1625"}))
        self.assertNotEqual(first["path"], second["path"])
        self.assertTrue((first["path"] / "flights.parquet").exists())

    def test_blank_record_fails_instead_of_renumbering(self):
        entry = self.source()
        path = self.raw / entry["path"]
        with zipfile.ZipFile(path) as archive:
            content = archive.read("flights.csv")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("flights.csv", content + b"\n")
        with self.assertRaisesRegex(ValueError, "fields"):
            list(read_source_rows(path, "flights.csv"))

    def test_bad_partition_is_not_silently_reused(self):
        entry = self.source()
        first = self.run_month(entry)
        report = first["path"] / "partition_quality.json"
        report.write_bytes(report.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_month(entry)

    def test_reuse_does_not_read_csv_records_again(self):
        entry = self.source()
        self.run_month(entry)
        with patch("aeroroute.normalization.pipeline.read_source_rows", side_effect=AssertionError("CSV unexpectedly rescanned")):
            self.assertTrue(self.run_month(entry)["reused"])

    def test_split_changes_do_not_change_normalized_partition(self):
        result = self.run_month(self.source())
        before = file_hash(result["path"] / "flights.parquet")
        first = publish_dataset([result], self.output)
        second = publish_dataset([result], self.output, historical_years=(), evaluation_years=(2020,))
        self.assertNotEqual(first, second)
        self.assertEqual(file_hash(result["path"] / "flights.parquet"), before)
        with self.assertRaises(ValueError):
            split_configuration([2020], [2020])

    def test_unrelated_airport_override_does_not_invalidate_month(self):
        entry = self.source()
        first = self.run_month(entry)
        override = self.root / "overrides.json"
        override.write_text(json.dumps([{"airport_id": 13930, "code": "ORD", "time_zone": "America/Chicago",
                                         "source": "test-reviewed-reference", "valid_from": "2020-01-01", "valid_to": "2020-12-31"}]))
        second = normalize_month(self.raw, entry, self.output, overrides=override)
        self.assertTrue(second["reused"])
        self.assertEqual(first["path"], second["path"])

    def test_changed_relevant_mapping_creates_new_partition(self):
        entry = self.source()
        first = self.run_month(entry)
        override = self.root / "overrides.json"
        override.write_text(json.dumps([{"airport_id": 14771, "code": "SFO", "time_zone": "America/Los_Angeles",
                                         "source": "test-reviewed-historical-reference", "valid_from": "2020-01-01", "valid_to": "2020-12-31"}]))
        second = normalize_month(self.raw, entry, self.output, overrides=override)
        self.assertFalse(second["reused"])
        self.assertNotEqual(first["path"], second["path"])

    def test_multiline_csv_field_does_not_change_record_ordinal(self):
        entry = self.source({"OriginCityName": "San\nFrancisco"})
        rows = list(read_source_rows(self.raw / entry["path"], entry["csv_member"]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[0][1]["OriginCityName"], "San\nFrancisco")

    def test_verify_only_never_creates_missing_partition(self):
        entry = self.source()
        with self.assertRaisesRegex(ValueError, "No compatible"):
            normalize_month(self.raw, entry, self.output, verify_only=True)
        self.assertFalse(self.output.exists())

    def test_pilot_report_independently_audits_parquet(self):
        result = self.run_month(self.source())
        dataset = publish_dataset([result], self.output)
        report = build_report(dataset)
        self.assertEqual(report["totals"]["row_count"], 1)
        self.assertTrue(report["months"][0]["audit"]["source_ordinals_contiguous"])
        self.assertEqual(report["months"][0]["outcome_resolution"]["ordinary"]["actual_arrival"]["resolved_pct_of_applicable"], 100)
        self.assertEqual(len(report["source_schema_groups"]), 1)
        self.assertEqual(report["experiment_split"]["evaluation_years"], [2025])

    def test_immutable_snapshot_hash_is_order_independent(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
