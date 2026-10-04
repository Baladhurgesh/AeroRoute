import argparse
import csv
import sys
import zipfile
from datetime import date
from pathlib import Path

from ..normalization.datasets import publish_dataset, split_configuration
from ..normalization.pipeline import normalize_month
from ..paths import PROCESSED_DIR, RAW_DIR
from ..storage.identity import load_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Normalize explicitly selected BTS months without filtering source records.")
    parser.add_argument("--periods", nargs="+", required=True, help="Explicit source months, e.g. 2020-01 2020-03 2025-01")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--tolerance-minutes", type=int, default=0)
    parser.add_argument("--airport-overrides", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--historical-years", nargs="*", type=int, default=list(range(2020, 2025)))
    parser.add_argument("--evaluation-years", nargs="*", type=int, default=[2025])
    args = parser.parse_args(argv)
    split_configuration(args.historical_years, args.evaluation_years)
    manifest = load_json(args.raw_dir / "manifest.json")
    periods = sorted(set(args.periods))
    for period in periods:
        day = date.fromisoformat(period + "-01")
        if day.year < 2018 or period != day.strftime("%Y-%m") or period not in manifest["files"]:
            raise ValueError(f"Requested month is unavailable or invalid: {period}")
    results = []
    output_dir = args.output_dir.resolve()
    for period in periods:
        print(f"{period}: {'verifying' if args.verify_only else 'normalizing/checking reusable partition'}", flush=True)
        result = normalize_month(args.raw_dir, manifest["files"][period], output_dir, args.batch_size,
                                 args.tolerance_minutes, args.airport_overrides, args.verify_only)
        results.append(result)
        quality = result["quality"]
        print(f"  {'reused' if result['reused'] else 'normalized'} {quality['row_count']:,} records; "
              f"{quality['parquet_bytes']:,} Parquet bytes; operation {result['elapsed_seconds']:.2f}s", flush=True)
        print(f"  cancelled={quality['cancelled_rows']:,}; diverted={quality['diverted_rows']:,}; "
              f"timezone-affected rows={quality['airport_mapping_affected_flight_rows']:,}", flush=True)
        print(f"  {result['path']}", flush=True)
    if not args.verify_only:
        path = publish_dataset(results, output_dir, args.historical_years, args.evaluation_years)
        print(f"Dataset quality report: {path / 'quality_report.json'}", flush=True)
    return 0


def run():
    try:
        return main()
    except (OSError, ValueError, KeyError, csv.Error, zipfile.BadZipFile) as error:
        print(f"Preprocessing failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
