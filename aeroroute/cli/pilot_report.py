import argparse
from pathlib import Path

from ..reporting.pilot import build_report
from ..storage.identity import load_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independently audit pilot Parquet counts and summarize field-resolution evidence.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.dataset_dir)
    output = args.output or args.dataset_dir.resolve().parents[2] / "reports" / report["dataset_version"] / "pilot_review.json"
    if output.exists():
        if load_json(output) != report:
            raise ValueError("Existing report differs; use a new output path")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    for month in report["months"]:
        audit = month["audit"]
        print(f"{month['period']}: {audit['row_count']:,} rows; cancelled={audit['cancelled_rows']:,}; diverted={audit['diverted_rows']:,}")
        for name, metrics in month["schedule_resolution"].items():
            print(f"  {name}: {metrics['resolved_pct_all_rows']:.3f}% resolved")
        for category, fields in month["outcome_resolution"].items():
            metric = fields["actual_arrival"]
            rate = metric["resolved_pct_of_applicable"]
            label = "N/A" if rate is None else f"{rate:.3f}%"
            print(f"  {category} arrival: {label} of {metric['applicable_rows']:,} applicable rows")
        print(f"  timezone-affected={month['timezone_mapping']['affected_flight_rows']:,}; "
              f"normalization={month['runtime_seconds']:.2f}s; peak RSS={month['peak_rss_bytes'] / 1024**2:.2f} MiB; "
              f"Parquet={month['parquet_bytes'] / 1024**2:.2f} MiB")
    print(f"Totals: {report['totals']}")
    print(f"Raw source schema variants: {len(report['source_schema_groups'])}")
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
