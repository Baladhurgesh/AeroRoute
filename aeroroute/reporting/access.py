import resource
import statistics
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from ..catalog.schedule import ScheduleStore
from ..catalog.services import ServiceCatalog
from ..outcomes.draws import Scenario
from ..outcomes.pools import PreparedPools
from ..outcomes.providers import EmpiricalProvider
from ..outcomes.replay import ExactReplayProvider
from ..storage.artifacts import contained_path, publish, staging, verify_derived
from ..storage.dataset import DatasetStore
from ..storage.identity import load_json, write_json


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
