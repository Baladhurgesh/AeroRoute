import argparse
import json
import resource
import sqlite3
import statistics
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aeroroute.data_access import DatasetStore, build_lookup, publish, staging
from aeroroute.data_snapshots import contained_path, load_json, verify_dataset, verify_derived, write_json
from aeroroute.sampling import (
    EmpiricalProvider, ExactReplayProvider, MatchingPolicy, PreparedPools, Scenario, prepare_pools, prepare_replay,
)
from aeroroute.schedule import ScheduleStore, build_schedule
from aeroroute.service_resolution import ResolutionPolicy, ServiceCatalog, build_services


def periods(values):
    result = []
    for value in values:
        day = date.fromisoformat(value + "-01")
        if day.strftime("%Y-%m") != value:
            raise ValueError("Periods must use YYYY-MM")
        result.append((day.year, day.month))
    return result


def verify_access(snapshot, artifact):
    artifact = Path(artifact).resolve()
    manifest = verify_derived(artifact)
    kind, identity = manifest["identity"]["kind"], manifest["identity"]
    root = artifact.parent.parent
    with ExitStack() as stack:
        if kind == "pilot":
            refs = identity["references"]
            store = stack.enter_context(DatasetStore(snapshot, contained_path(root, "lookups/" + refs["lookup"])))
            catalog = stack.enter_context(ServiceCatalog(store, contained_path(root, "services/" + refs["services"])))
            schedule = stack.enter_context(ScheduleStore(catalog, contained_path(root, "schedules/" + refs["schedule"])))
            pools = stack.enter_context(PreparedPools(catalog, schedule, contained_path(root, "pools/" + refs["pools"])))
            replay = stack.enter_context(ExactReplayProvider(schedule, contained_path(root, "replay/" + refs["replay"])))
            scenario = Scenario(**load_json(artifact / "scenario.json"))
            flight = schedule.get_flight(identity["service_instance_id"])
            empirical = EmpiricalProvider(pools, scenario)
            report = load_json(artifact / "pilot_report.json")
            if (empirical.sample(flight).historical_sample.to_dict() != report["sample"]
                    or replay.sample(flight).historical_sample.to_dict() != report["replay"]):
                raise ValueError("Pilot report does not reproduce")
        elif kind in ("lookup", "services"):
            lookup = artifact if kind == "lookup" else contained_path(root, "lookups/" + identity["lookup"])
            store = stack.enter_context(DatasetStore(snapshot, lookup))
            if kind == "services":
                stack.enter_context(ServiceCatalog(store, artifact))
        else:
            raise ValueError("Verify an inspect-stage lookup/services artifact or a complete pilot artifact")
    return manifest


def run_pilot(store, catalog, schedule, pools, replay, output, seed, scenario_id, draws, service_id=None):
    if service_id is None:
        flight = next((flight for flight in schedule.all_flights() if catalog.get(flight.flight_id)["split"] == "evaluation"), None)
        if flight is None:
            raise ValueError("Pilot needs a scheduled held-out departure")
    else:
        flight = schedule.get_flight(service_id)
    scenario = Scenario.create(scenario_id, seed, pools)
    references = {"lookup": store.lookup_path.name, "services": catalog.path.name, "schedule": schedule.path.name,
                  "pools": pools.path.name, "replay": replay.path.name}
    identity = {"kind": "pilot", "schema_version": 1, "references": references,
                "scenario_sha256": scenario.sha256, "service_instance_id": flight.flight_id, "benchmark_draws": draws}
    final, stage = staging(output, identity)
    if stage is None:
        verify_access(store.snapshot["path"], final)
        return final
    query = schedule.departures(flight.origin.airport_id, flight.scheduled_departure_at, flight.scheduled_departure_at + timedelta(seconds=1))
    if flight not in query.flights:
        raise ValueError("Selected pilot departure is absent from schedule query")
    provider = EmpiricalProvider(pools, scenario)
    store.cache.close()
    schedule.cache.close()
    store.metrics.clear()
    started = time.perf_counter()
    sampled = provider.sample(flight, episode_id="policy-a", step_index=1)
    cold = time.perf_counter() - started
    cold_metrics = dict(store.metrics)
    elapsed = []
    for _ in range(draws):
        started = time.perf_counter()
        repeated = provider.sample(flight, episode_id="policy-b", step_index=3)
        elapsed.append(time.perf_counter() - started)
        if sampled.evidence != repeated.evidence or sampled.receipt.draw != repeated.receipt.draw:
            raise ValueError("Scenario outcome changed with decision metadata")
    replayed = replay.sample(flight)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"sample": sampled.historical_sample.to_dict(), "replay": replayed.historical_sample.to_dict(),
              "sample_receipt": asdict(sampled.receipt), "replay_receipt": asdict(replayed.receipt),
              "scenario_stable_across_steps": True, "query_complete": query.complete,
              "scope": "pilot plumbing; historical replay under future traveler/recovery assumptions",
              "benchmark": {"cold_application_cache_seconds": cold, "cold_metrics": cold_metrics,
                            "warm_draws": draws, "warm_median_seconds": statistics.median(elapsed),
                            "warm_p95_seconds": sorted(elapsed)[min(len(elapsed) - 1, int(len(elapsed) * .95))],
                            "cumulative_evidence_metrics": dict(store.metrics),
                            "schedule_metrics": dict(schedule.cache.metrics),
                            "evidence_cache_bytes": store.cache_size_bytes, "open_evidence_handles": len(store.handles),
                            "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                            "cache_note": "Snapshot verification excluded; OS filesystem cache not controlled"}}
    write_json(stage / "scenario.json", asdict(scenario))
    write_json(stage / "pilot_report.json", report)
    return publish(stage, final, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build immutable, service-aware BTS data access artifacts on an explicit pilot snapshot.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/access")
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


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        print(f"Data access failed: {error}", file=sys.stderr)
        sys.exit(1)
