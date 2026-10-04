import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from aeroroute.storage.identity import file_hash
from aeroroute.normalization.pipeline import normalize_month
from aeroroute.normalization.datasets import publish_dataset


def flight_row(day="2020-01-01", **changes):
    row = {
        "FlightDate": day, "Origin": "SFO", "OriginAirportID": "14771", "OriginStateName": "California",
        "Dest": "JFK", "DestAirportID": "12478", "DestStateName": "New York",
        "Marketing_Airline_Network": "UA", "Operating_Airline": "UA",
        "Flight_Number_Marketing_Airline": "0012", "Flight_Number_Operating_Airline": "0012",
        "CRSDepTime": "0800", "CRSArrTime": "1600", "CRSElapsedTime": "300",
        "DepDelay": "15", "ActualElapsedTime": "300", "ArrDelay": "15",
        "DepTime": "0815", "ArrTime": "1615", "Cancelled": "0", "Diverted": "0",
        "DivAirportLandings": "0", "Duplicate": "N",
    }
    return {**row, **changes}


class AccessFixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def snapshot(self, rows=None, name="dataset", batch_size=2):
        rows = rows or [flight_row(), flight_row("2020-01-02"), flight_row("2025-01-01")]
        root = self.root / name
        raw = root / "raw"
        raw.mkdir(parents=True)
        processed = root / "processed"
        months = sorted({row["FlightDate"][:7] for row in rows})
        results = []
        for period in months:
            subset = [row for row in rows if row["FlightDate"].startswith(period)]
            columns = sorted({key for row in subset for key in row})
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(subset)
            path = raw / f"{period}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("flights.csv", stream.getvalue())
            year, month = map(int, period.split("-"))
            entry = {"year": year, "month": month, "path": path.name, "sha256": file_hash(path),
                     "csv_member": "flights.csv", "row_count": len(subset), "size_bytes": path.stat().st_size,
                     "status": "validated", "columns": columns}
            results.append(normalize_month(raw, entry, processed, batch_size=batch_size))
        return publish_dataset(results, processed) / "dataset_manifest.json"
