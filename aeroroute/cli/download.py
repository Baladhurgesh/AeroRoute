import argparse
import csv
import sys
import zipfile
from pathlib import Path

from ..acquisition.bts import (
    BASE_URL, DEFAULT_OUTPUT, archive_name, check_disk_space, curl, load_manifest,
    parse_listing, process_month, requested_months,
)


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


def run():
    try:
        return main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed archives are preserved; rerun to continue.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
