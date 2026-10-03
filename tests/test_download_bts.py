import csv
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import download_bts as bts


COLUMNS = [
    "Year", "Month", "FlightDate", "Origin", "Dest", "OriginAirportID",
    "DestAirportID", "Marketing_Airline_Network", "Operating_Airline",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepDelay", "ArrDelay",
    "ActualElapsedTime", "Cancelled", "Diverted", "DivReachedDest",
    "DivActualElapsedTime", "DivArrDelay", "DivAirportLandings",
    "Flight_Number_Marketing_Airline", "Flight_Number_Operating_Airline",
    "DepTime", "ArrTime", "CancellationCode",
] + [f"Div{stop}{field}" for stop in range(1, 6)
     for field in ("Airport", "AirportID", "WheelsOn", "WheelsOff")]


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "sample.zip"

    def make_zip(self, path=None, rows=None, columns=None):
        path = path or self.archive
        columns = COLUMNS if columns is None else columns
        rows = [{"Year": "2020", "Month": "1", "FlightDate": "2020-01-02",
                 "Cancelled": "0.00", "Diverted": "0.00"}] if rows is None else rows
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("flights.csv", buffer.getvalue())
            archive.writestr("readme.txt", "BTS fixture")
        return path.stat().st_size

    def empty_manifest(self):
        return bts.load_manifest(self.root / "manifest.json")

    def fake_transfer(self, url, path):
        self.make_zip(path)
        return {"etag": '"fixture"', "last_modified": "fixture-date"}

    def test_month_range(self):
        result = bts.requested_months(2020, 2025)
        self.assertEqual(len(result), 72)
        self.assertEqual(result[0], (2020, 1))
        self.assertEqual(result[-1], (2025, 12))
        self.assertEqual(bts.requested_months(2020, 2020, 1), [(2020, 1)])

    def test_invalid_ranges(self):
        for args in [(2017, 2020), (2025, 2020), (2020, 2021, 13)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                bts.requested_months(*args)

    def test_listing_parsing(self):
        name = bts.archive_name(2020, 1)
        html = f'1/1/2020 1:00 PM 12345 <a href="/PREZIP/{name}">{name}</a><br>'
        self.assertEqual(bts.parse_listing(html), {name: 12345})

    def test_empty_listing_rejected(self):
        with self.assertRaises(ValueError):
            bts.parse_listing("<html>Service unavailable</html>")

    def test_valid_archive(self):
        self.make_zip()
        result = bts.validate_archive(self.archive, 2020, 1)
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["first_flight_date"], "2020-01-02")
        self.assertEqual(result["columns"], COLUMNS)

    def test_real_bts_header_whitespace_and_trailing_empty_column(self):
        columns = ["Operating_Airline " if column == "Operating_Airline" else column
                   for column in COLUMNS] + [""]
        self.make_zip(columns=columns)
        result = bts.validate_archive(self.archive, 2020, 1)
        self.assertEqual(result["columns"], columns)
        self.assertEqual(result["row_count"], 1)

    def test_duplicate_normalized_columns_rejected(self):
        self.make_zip(columns=COLUMNS + ["Operating_Airline "])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bts.validate_archive(self.archive, 2020, 1)

    def test_zip_crc_error_rejected(self):
        self.make_zip()
        with zipfile.ZipFile(self.archive) as archive:
            content = archive.read("flights.csv")
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("flights.csv", content)
        original = self.archive.read_bytes()
        self.archive.write_bytes(original.replace(b"2020", b"2021", 1))
        with self.assertRaises(zipfile.BadZipFile):
            bts.validate_archive(self.archive, 2020, 1)

    def test_cancelled_and_diverted_rows_retained(self):
        self.make_zip(rows=[
            {"Year": "2020", "Month": "1", "FlightDate": "2020-01-02",
             "Cancelled": "1", "Diverted": "0"},
            {"Year": "2020", "Month": "1", "FlightDate": "2020-01-03",
             "Cancelled": "0", "Diverted": "1"},
        ])
        result = bts.validate_archive(self.archive, 2020, 1)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["cancelled_rows"], 1)
        self.assertEqual(result["diverted_rows"], 1)

    def test_wrong_month_rejected(self):
        self.make_zip()
        with self.assertRaisesRegex(ValueError, "month"):
            bts.validate_archive(self.archive, 2020, 2)

    def test_invalid_date_rejected(self):
        self.make_zip(rows=[{"Year": "2020", "Month": "1", "FlightDate": "2020-01-99"}])
        with self.assertRaises(ValueError):
            bts.validate_archive(self.archive, 2020, 1)

    def test_missing_column_rejected(self):
        self.make_zip(columns=[column for column in COLUMNS if column != "Diverted"])
        with self.assertRaisesRegex(ValueError, "columns"):
            bts.validate_archive(self.archive, 2020, 1)

    def test_flight_identity_and_all_diversion_stop_columns_required(self):
        additions = [
            "Flight_Number_Marketing_Airline", "Flight_Number_Operating_Airline",
            "DivAirportLandings", "DepTime", "ArrTime", "CancellationCode",
        ] + [f"Div{stop}{field}" for stop in range(1, 6)
             for field in ("Airport", "AirportID", "WheelsOn", "WheelsOff")]
        for missing in additions:
            with self.subTest(missing=missing):
                self.make_zip(columns=[column for column in COLUMNS if column != missing])
                with self.assertRaisesRegex(ValueError, missing):
                    bts.validate_archive(self.archive, 2020, 1)

    def test_stranded_multistop_diversion_allows_missing_arrival_duration(self):
        self.make_zip(rows=[{
            "Year": "2020", "Month": "1", "FlightDate": "2020-01-02",
            "Cancelled": "0", "Diverted": "1", "DivReachedDest": "0",
            "DivAirportLandings": "2", "Div1Airport": "ORD", "Div1WheelsOn": "2300",
            "Div2Airport": "PHL", "Div2WheelsOn": "0200",
        }])
        original = self.archive.read_bytes()
        result = bts.validate_archive(self.archive, 2020, 1)
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["diverted_rows"], 1)
        self.assertEqual(self.archive.read_bytes(), original)

    def test_extra_columns_retained(self):
        self.make_zip(columns=COLUMNS + ["WeatherDelay"])
        result = bts.validate_archive(self.archive, 2020, 1)
        self.assertEqual(result["columns"], COLUMNS + ["WeatherDelay"])

    def test_empty_csv_rejected(self):
        self.make_zip(rows=[])
        with self.assertRaisesRegex(ValueError, "rows"):
            bts.validate_archive(self.archive, 2020, 1)

    def test_html_and_truncated_zip_rejected(self):
        self.archive.write_bytes(b"<html>Error</html>")
        with self.assertRaises(zipfile.BadZipFile):
            bts.validate_archive(self.archive, 2020, 1)
        self.make_zip()
        self.archive.write_bytes(self.archive.read_bytes()[:20])
        with self.assertRaises(zipfile.BadZipFile):
            bts.validate_archive(self.archive, 2020, 1)

    def test_malformed_row_rejected(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("flights.csv", ",".join(COLUMNS) + "\n2020,1\n")
        with self.assertRaisesRegex(ValueError, "fields"):
            bts.validate_archive(self.archive, 2020, 1)

    def test_hash(self):
        self.archive.write_bytes(b"example")
        self.assertEqual(bts.sha256_file(self.archive), hashlib.sha256(b"example").hexdigest())

    def test_download_and_skip(self):
        size = self.make_zip()
        manifest = self.empty_manifest()
        with patch.object(bts, "fetch_archive", side_effect=self.fake_transfer) as transfer:
            self.assertEqual(bts.process_month(2020, 1, self.root, size, manifest), "downloaded")
            self.assertEqual(bts.process_month(2020, 1, self.root, size, manifest), "skipped")
            transfer.assert_called_once()
        saved = bts.load_manifest(self.root / "manifest.json")
        entry = saved["files"]["2020-01"]
        self.assertEqual(entry["row_count"], 1)
        self.assertEqual(entry["source_etag"], '"fixture"')
        self.assertFalse((self.root / "2020" / (bts.archive_name(2020, 1) + ".part")).exists())

    def test_adopt_existing_archive(self):
        path = self.root / "2020" / bts.archive_name(2020, 1)
        size = self.make_zip(path)
        with patch.object(bts, "fetch_archive") as transfer:
            self.assertEqual(bts.process_month(2020, 1, self.root, size, self.empty_manifest()), "adopted")
            transfer.assert_not_called()

    def test_size_mismatch_leaves_no_completed_file(self):
        manifest = self.empty_manifest()
        with patch.object(bts, "fetch_archive", side_effect=self.fake_transfer):
            with self.assertRaisesRegex(ValueError, "size"):
                bts.process_month(2020, 1, self.root, 1, manifest)
        self.assertFalse((self.root / "2020" / bts.archive_name(2020, 1)).exists())
        self.assertEqual(manifest["files"], {})

    def test_interruption_leaves_no_completed_file(self):
        def interrupt(url, path):
            path.write_bytes(b"partial")
            raise KeyboardInterrupt

        with patch.object(bts, "fetch_archive", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                bts.process_month(2020, 1, self.root, 123, self.empty_manifest())
        self.assertFalse((self.root / "2020" / bts.archive_name(2020, 1)).exists())
        self.assertFalse((self.root / "manifest.json").exists())

    def test_corrupt_existing_archive_not_overwritten(self):
        path = self.root / "2020" / bts.archive_name(2020, 1)
        size = self.make_zip(path)
        manifest = self.empty_manifest()
        bts.process_month(2020, 1, self.root, size, manifest)
        content = path.read_bytes()
        changed = content[:-1] + bytes([content[-1] ^ 1])
        path.write_bytes(changed)
        with patch.object(bts, "fetch_archive") as transfer:
            with self.assertRaisesRegex(ValueError, "checksum"):
                bts.process_month(2020, 1, self.root, size, manifest)
            transfer.assert_not_called()
        self.assertEqual(path.read_bytes(), changed)

    def test_verify_only_does_not_download_or_write(self):
        path = self.root / "2020" / bts.archive_name(2020, 1)
        size = self.make_zip(path)
        manifest = self.empty_manifest()
        bts.process_month(2020, 1, self.root, size, manifest)
        original = (self.root / "manifest.json").read_bytes()
        with patch.object(bts, "fetch_archive") as transfer:
            self.assertEqual(bts.process_month(2020, 1, self.root, size, manifest, True), "verified")
            with self.assertRaises(FileNotFoundError):
                bts.process_month(2020, 2, self.root, size, manifest, True)
            transfer.assert_not_called()
        self.assertEqual((self.root / "manifest.json").read_bytes(), original)

    def test_conflicting_manifest_rejected(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"version": 99, "files": {}}))
        with self.assertRaises(ValueError):
            bts.load_manifest(path)

    def test_disk_space_margin(self):
        with patch.object(bts.shutil, "disk_usage", return_value=(100, 95, 5)):
            with self.assertRaisesRegex(OSError, "space"):
                bts.check_disk_space(self.root, 10, reserve=2)

    def test_curl_http_error_and_retry_flags(self):
        error = subprocess.CalledProcessError(22, ["curl"], stderr="HTTP 404")
        with patch.object(bts.subprocess, "run", side_effect=error) as run:
            with self.assertRaisesRegex(RuntimeError, "404"):
                bts.fetch_archive("https://example.invalid/archive.zip", self.archive)
        command = run.call_args.args[0]
        self.assertIn("--fail", command)
        self.assertIn("--retry", command)
        self.assertNotIn("--insecure", command)


if __name__ == "__main__":
    unittest.main()
