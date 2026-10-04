import csv
import io
import re
import zipfile
from datetime import date


REQUIRED_COLUMNS = {
    "Year", "Month", "FlightDate", "Origin", "Dest", "OriginAirportID",
    "DestAirportID", "Marketing_Airline_Network", "Operating_Airline",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepDelay", "ArrDelay",
    "ActualElapsedTime", "Cancelled", "Diverted", "DivReachedDest",
    "DivActualElapsedTime", "DivArrDelay", "DivAirportLandings",
    "Flight_Number_Marketing_Airline", "Flight_Number_Operating_Airline",
    "DepTime", "ArrTime", "CancellationCode",
} | {f"Div{stop}{field}" for stop in range(1, 6)
     for field in ("Airport", "AirportID", "WheelsOn", "WheelsOff")}
VALIDATION_VERSION = 2


def normalize_column(column):
    return re.sub(r"[^a-z0-9]", "", column.lower())


def validate_archive(path, year, month):
    dates = set()
    row_count = cancelled_rows = diverted_rows = 0
    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.infolist()
                   if not member.is_dir() and member.filename.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected exactly one flight CSV member in the ZIP")
        with archive.open(members[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            reader = csv.reader(text, strict=True)
            columns = next(reader, [])
            indexes = {normalize_column(column): index for index, column in enumerate(columns) if column.strip()}
            if len(indexes) != sum(bool(column.strip()) for column in columns):
                raise ValueError("Duplicate CSV columns")
            missing = sorted(column for column in REQUIRED_COLUMNS if normalize_column(column) not in indexes)
            if missing:
                raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
            year_index, month_index, date_index, cancel_index, divert_index = (
                indexes[normalize_column(column)]
                for column in ("Year", "Month", "FlightDate", "Cancelled", "Diverted")
            )
            for row in reader:
                if len(row) != len(columns):
                    raise ValueError(f"CSV row {reader.line_num} has {len(row)} fields, expected {len(columns)}")
                if row[year_index] != str(year) or row[month_index] not in {str(month), f"{month:02d}"}:
                    raise ValueError(f"CSV row {reader.line_num} is outside expected year/month {year}-{month:02d}")
                flight_date = row[date_index]
                if flight_date not in dates:
                    parsed = date.fromisoformat(flight_date)
                    if (parsed.year, parsed.month) != (year, month):
                        raise ValueError(f"FlightDate outside expected month: {flight_date}")
                    dates.add(flight_date)
                cancelled, diverted = float(row[cancel_index]), float(row[divert_index])
                if cancelled not in (0, 1) or diverted not in (0, 1):
                    raise ValueError(f"Invalid outcome flags at CSV row {reader.line_num}")
                cancelled_rows += int(cancelled)
                diverted_rows += int(diverted)
                row_count += 1
        if not row_count:
            raise ValueError("Flight CSV has no data rows")
        for member in archive.infolist():
            if member != members[0] and not member.is_dir():
                with archive.open(member) as handle:
                    while handle.read(1024 * 1024):
                        pass
    return {
        "csv_member": members[0].filename,
        "columns": columns,
        "row_count": row_count,
        "cancelled_rows": cancelled_rows,
        "diverted_rows": diverted_rows,
        "first_flight_date": min(dates),
        "last_flight_date": max(dates),
        "flight_dates": sorted(dates),
    }


def read_source_rows(path, member):
    with zipfile.ZipFile(path) as archive, archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
        reader = csv.reader(text, strict=True)
        original = next(reader, [])
        headers = [column.strip() for column in original]
        named = [column for column in headers if column]
        if not named or len(named) != len(set(named)):
            raise ValueError("Missing or duplicate normalized CSV headers")
        for ordinal, cells in enumerate(reader, 1):
            if len(cells) != len(headers):
                raise ValueError(f"CSV data record {ordinal} has {len(cells)} fields; expected {len(headers)}")
            if any(value.strip() for field, value in zip(headers, cells) if not field):
                raise ValueError(f"Nonempty unnamed CSV field at record {ordinal}")
            yield ordinal, {field: value for field, value in zip(headers, cells) if field}
