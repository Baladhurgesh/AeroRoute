import argparse
import json
import shutil
import sqlite3
import sys
from contextlib import ExitStack
from datetime import date
from pathlib import Path

from ..catalog.schedule import ScheduleStore, build_schedule
from ..catalog.services import ResolutionPolicy, ServiceCatalog, build_services
from ..outcomes.pools import MatchingPolicy, PreparedPools, prepare_pools
from ..outcomes.replay import ExactReplayProvider, prepare_replay
from ..paths import ACCESS_DIR
from ..reporting.access import run_pilot, verify_access
from ..storage.artifacts import verify_dataset
from ..storage.dataset import DatasetStore
from ..storage.lookup import build_lookup


def periods(values):
    result = []
    for value in values:
        day = date.fromisoformat(value + "-01")
        if day.strftime("%Y-%m") != value:
            raise ValueError("Periods must use YYYY-MM")
        result.append((day.year, day.month))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build immutable, service-aware BTS data access artifacts on an explicit pilot snapshot.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ACCESS_DIR)
    parser.add_argument("--stage", choices=("inspect", "pilot"), default="inspect")
    parser.add_argument("--review-reference")
    parser.add_argument("--schedule-periods", nargs="+", default=[])
    parser.add_argument("--fitting-years", nargs="+", type=int, default=[])
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--bucket-hours", type=int, default=6)
    parser.add_argument("--retain-time-of-day", action="store_true")
    parser.add_argument("--validation-reference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario-id", default="pilot-scenario")
    parser.add_argument("--service-id")
    parser.add_argument("--benchmark-draws", type=int, default=20)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args(argv)
    snapshot = verify_dataset(args.snapshot)
    if args.verify_only:
        if not args.artifact:
            raise ValueError("--verify-only requires --artifact")
        verify_access(args.snapshot, args.artifact)
        print(f"Verified: {args.artifact}")
        return 0
    if args.dry_run:
        print(json.dumps({"snapshot": snapshot["manifest"]["dataset_version"],
                          "periods": [[p["source"]["year"], p["source"]["month"]] for p in snapshot["partitions"]],
                          "source_counts": {key: len(value["sources"]) for key, value in snapshot["inventories"].items()},
                          "stage": args.stage, "writes": False}, sort_keys=True))
        return 0
    if args.benchmark_draws < 1:
        raise ValueError("Benchmark draws must be positive")
    if args.stage == "pilot" and (not args.review_reference or not args.schedule_periods or not args.fitting_years):
        raise ValueError("Pilot requires an explicit review reference, schedule periods, and fitting years")
    output = args.output_dir.resolve()
    existing = output
    while not existing.exists():
        existing = existing.parent
    required = 3 * 1024 ** 3
    free = shutil.disk_usage(existing).free
    if free < required:
        raise OSError(f"Post-processing needs {required:,} free bytes of scratch space; only {free:,} available. The normalized Parquet is already stored and is not copied.")
    print("Preparing verified source and stop lookup", flush=True)
    lookup = build_lookup(args.snapshot, output / "lookups")
    with ExitStack() as stack:
        store = stack.enter_context(DatasetStore(args.snapshot, lookup))
        print("Inspecting service identities and supporting evidence", flush=True)
        path = build_services(store, output / "services", ResolutionPolicy(review_reference=args.review_reference))
        catalog = stack.enter_context(ServiceCatalog(store, path))
        print(json.dumps(catalog.report, sort_keys=True), flush=True)
        if args.stage == "inspect":
            print(path)
            return 0
        schedule = stack.enter_context(ScheduleStore(catalog, build_schedule(catalog, output / "schedules", periods(args.schedule_periods))))
        policy = MatchingPolicy(bucket_hours=args.bucket_hours, min_support=args.min_support, validation_reference=args.validation_reference)
        if args.retain_time_of_day:
            policy = MatchingPolicy(levels=(("carrier", "quarter", "weekday", "bucket"), ("carrier", "quarter", "bucket"),
                                             ("carrier", "bucket"), ("carrier",), ()),
                                    bucket_hours=args.bucket_hours, min_support=args.min_support, validation_reference=args.validation_reference)
        pools = stack.enter_context(PreparedPools(catalog, schedule, prepare_pools(catalog, schedule, output / "pools", policy, args.fitting_years)))
        replay = stack.enter_context(ExactReplayProvider(schedule, prepare_replay(schedule, output / "replay")))
        path = run_pilot(store, catalog, schedule, pools, replay, output / "pilots", args.seed,
                         args.scenario_id, args.benchmark_draws, args.service_id)
        print(path)
    return 0


def run():
    try:
        return main()
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        print(f"Data access failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
