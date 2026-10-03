import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path


BASE_URL = "https://transtats.bts.gov/PREZIP/"
PREFIX = "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/raw/bts_marketing"
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


def archive_name(year, month):
    return f"{PREFIX}_{year}_{month}.zip"


def requested_months(start_year, end_year, month=None):
    if start_year < 2018 or end_year < start_year:
        raise ValueError("Years must start at 2018 or later and be in ascending order")
    if month is not None and not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")
    return [(year, selected) for year in range(start_year, end_year + 1)
            for selected in ([month] if month is not None else range(1, 13))]


def curl(*args):
    command = [
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", "20", "--max-time", "300",
        "--retry", "3", "--retry-connrefused", "--retry-max-time", "600",
        *map(str, args),
    ]
    try:
        return subprocess.run(command, check=True, capture_output=True,
                              text=True, timeout=1000).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"BTS request failed: {(error.stderr or '')[-1000:].strip()}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("BTS request exceeded the total retry timeout") from error


def parse_listing(html):
    pattern = rf'(\d+)\s+<a href="[^\"]*/({re.escape(PREFIX)}_\d{{4}}_\d{{1,2}}\.zip)"'
    matches = re.findall(pattern, html, re.IGNORECASE)
    result = {name: int(size) for size, name in matches}
    if not result:
        raise ValueError("BTS directory did not contain the expected monthly archive listing")
    return result


def fetch_archive(url, path):
    headers = curl("--dump-header", "-", "--output", path, url)
    metadata = {}
    for line in headers.splitlines():
        if line.startswith("HTTP/"):
            metadata = {}
        name, separator, value = line.partition(":")
        if separator and name.lower() in {"etag", "last-modified"}:
            metadata[name.lower().replace("-", "_")] = value.strip()
    return metadata


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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


def load_manifest(path):
    if not path.exists():
        return {"version": 1, "dataset": PREFIX, "files": {}}
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or manifest.get("dataset") != PREFIX or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Unrecognized manifest format; existing manifest will not be overwritten")
    if any(not isinstance(entry, dict) for entry in manifest["files"].values()):
        raise ValueError("Invalid manifest entries")
    return manifest


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def check_disk_space(root, required_bytes, reserve=1024 ** 3):
    existing = root.resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing)[2]
    if free < required_bytes + reserve:
        raise OSError(f"Insufficient disk space: {free:,} bytes free; need {required_bytes + reserve:,} including reserve")
    return free


def process_month(year, month, root, expected_size, manifest, verify_only=False):
    key = f"{year}-{month:02d}"
    relative = Path(str(year)) / archive_name(year, month)
    path = root / relative
    url = BASE_URL + path.name
    entry = manifest["files"].get(key)
    if entry:
        expected = {"year": year, "month": month, "source_url": url,
                    "path": relative.as_posix(), "size_bytes": expected_size,
                    "status": "validated", "validation_version": VALIDATION_VERSION}
        if any(entry.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Conflicting manifest/source metadata for {key}; refusing to overwrite")
        if not isinstance(entry.get("row_count"), int) or entry["row_count"] <= 0:
            raise ValueError(f"Invalid row count in manifest for {key}")
    if verify_only and (not path.is_file() or not entry):
        raise FileNotFoundError(f"Missing validated archive or manifest entry for {key}")
    if path.exists():
        if path.stat().st_size != expected_size:
            raise ValueError(f"Existing archive size mismatch for {key}; refusing to overwrite")
        checksum = sha256_file(path)
        if entry and checksum != entry.get("sha256"):
            raise ValueError(f"Existing archive checksum mismatch for {key}; refusing to overwrite")
        if entry and not verify_only:
            return "skipped"
        validation = validate_archive(path, year, month)
        if verify_only:
            if any(entry.get(field) != value for field, value in validation.items()):
                raise ValueError(f"Archive validation differs from manifest for {key}")
            return "verified"
        metadata = {}
        action = "adopted"
    else:
        check_disk_space(root, expected_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".zip.part")
        metadata = fetch_archive(url, partial)
        if partial.stat().st_size != expected_size:
            raise ValueError(f"Downloaded archive size mismatch for {key}")
        validation = validate_archive(partial, year, month)
        checksum = sha256_file(partial)
        if entry and checksum != entry.get("sha256"):
            raise ValueError(f"Downloaded archive checksum differs from existing manifest for {key}")
        if path.exists():
            raise FileExistsError(f"Archive appeared during download: {path}; refusing to overwrite")
        partial.replace(path)
        action = "downloaded"
    manifest["files"][key] = {
        "year": year, "month": month, "path": relative.as_posix(),
        "source_url": url, "size_bytes": expected_size, "sha256": checksum,
        "source_etag": metadata.get("etag"),
        "source_last_modified": metadata.get("last_modified"),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "validated", "validation_version": VALIDATION_VERSION,
        **validation,
    }
    save_manifest(root / "manifest.json", manifest)
    return action


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and validate original monthly BTS marketing-carrier ZIPs.")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--month", type=int, help="Limit to this month in each requested year (1–12).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Check source coverage and storage without writing files.")
    mode.add_argument("--verify-only", action="store_true", help="Fully validate local archives offline without changing files.")
    args = parser.parse_args(argv)
    months = requested_months(args.start_year, args.end_year, args.month)
    root = args.output_dir.resolve()
    manifest = load_manifest(root / "manifest.json")
    if args.verify_only:
        sizes = {archive_name(year, month): manifest["files"].get(f"{year}-{month:02d}", {}).get("size_bytes")
                 for year, month in months}
    else:
        sizes = parse_listing(curl(BASE_URL))
    missing = [f"{year}-{month:02d}" for year, month in months
               if not isinstance(sizes.get(archive_name(year, month)), int)
               or sizes[archive_name(year, month)] <= 0]
    if missing:
        raise ValueError(f"Missing monthly metadata: {', '.join(missing)}")
    total = sum(sizes[archive_name(year, month)] for year, month in months)
    print(f"Requested: {len(months)} months; {total:,} compressed bytes ({total / 1024 ** 3:.3f} GiB)", flush=True)
    print(f"Output: {root}", flush=True)
    if not args.verify_only:
        needed = sum(sizes[archive_name(year, month)] for year, month in months
                     if not (root / str(year) / archive_name(year, month)).exists())
        free = check_disk_space(root, needed)
        print(f"Free disk: {free / 1024 ** 3:.2f} GiB; remaining downloads: {needed / 1024 ** 3:.3f} GiB", flush=True)
    if args.dry_run:
        return 0
    failed, complete = [], []
    for index, (year, month) in enumerate(months, 1):
        key = f"{year}-{month:02d}"
        print(f"[{index}/{len(months)}] {key}: {'verifying' if args.verify_only else 'checking/downloading'}", flush=True)
        try:
            action = process_month(year, month, root, sizes[archive_name(year, month)], manifest, args.verify_only)
        except PermissionError:
            raise
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, csv.Error, EOFError) as error:
            failed.append(key)
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)
            continue
        entry = manifest["files"][key]
        complete.append(entry)
        print(f"  {action}: {entry['row_count']:,} rows; {entry['size_bytes']:,} bytes", flush=True)
    for year in range(args.start_year, args.end_year + 1):
        entries = [entry for entry in complete if entry["year"] == year]
        print(f"{year}: {len(entries)} months, {sum(entry['row_count'] for entry in entries):,} rows, "
              f"{sum(entry['size_bytes'] for entry in entries):,} bytes", flush=True)
    print(f"Complete: {len(complete)}/{len(months)}; total rows: {sum(entry['row_count'] for entry in complete):,}", flush=True)
    if failed:
        print(f"Incomplete months: {', '.join(failed)}", file=sys.stderr, flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed archives are preserved; rerun to continue.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
