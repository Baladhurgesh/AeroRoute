import argparse
import sqlite3
import sys
from contextlib import ExitStack
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from ..catalog.schedule import ScheduleStore
from ..catalog.services import ServiceCatalog
from ..domain.records import EpisodeRecord, EpisodeSettings, TripRequest
from ..outcomes.draws import Scenario
from ..outcomes.pools import PreparedPools
from ..outcomes.providers import EmpiricalProvider
from ..outcomes.replay import ExactReplayProvider
from ..paths import PROCESSED_DIR
from ..simulation.environment import FlightEnvironment
from ..simulation.transitions import TransitionPolicy
from ..storage.artifacts import contained_path, publish, staging, verify_derived
from ..storage.dataset import DatasetStore
from ..storage.identity import digest, load_json, write_json
from ..verification.verifier import EpisodeVerifier


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or verify a sequential flight episode against a prepared pilot bundle.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--mode", choices=("replay", "empirical"), default="replay")
    parser.add_argument("--interface", choices=("core", "gymnasium"), default="core")
    parser.add_argument("--agent", choices=("earliest-arrival", "random"), default="earliest-arrival")
    parser.add_argument("--agent-seed", type=int, default=0,
                        help="Agent/adapter RNG seed; does not change the frozen outcome scenario.")
    parser.add_argument("--max-flights", type=int, default=1000,
                        help="Gymnasium action capacity; overflow raises rather than hiding flights.")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--flight-id")
    parser.add_argument("--max-duration-minutes", type=int, default=1440)
    parser.add_argument("--max-decisions", type=int, default=8)
    parser.add_argument("--cancellation-recovery-minutes", type=int, default=60)
    parser.add_argument("--missed-boarding-buffer-minutes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "episodes")
    parser.add_argument("--verify-episode", type=Path)
    args = parser.parse_args(argv)
    if not args.verify_episode:
        if args.agent_seed < 0 or args.max_flights < 1:
            parser.error("--agent-seed must be nonnegative and --max-flights must be positive")
        if args.interface == "core" and args.agent != "earliest-arrival":
            parser.error("--agent random requires --interface gymnasium")
        if args.interface == "gymnasium":
            try:
                from ..simulation.agents import EarliestArrivalAgent, RandomAgent, run_episode
                from ..simulation.gymnasium_env import FlightGymEnv
            except ModuleNotFoundError as error:
                if error.name not in ("gymnasium", "numpy"):
                    raise
                raise ValueError("Gymnasium examples require the RL extra: uv sync --extra rl; run with uv run --extra rl") from error
    pilot = args.pilot.resolve()
    pilot_manifest = verify_derived(pilot, "pilot")
    refs, root = pilot_manifest["identity"]["references"], pilot.parent.parent
    experiment = None
    if args.verify_episode:
        episode_manifest = verify_derived(args.verify_episode, "episode")
        experiment = load_json(args.verify_episode / "experiment.json")
        if (episode_manifest["identity"]["experiment_sha256"] != digest(experiment)
                or experiment["pilot"] != pilot.name):
            raise ValueError("Episode experiment identity/pilot mismatch")
        mode = experiment["mode"]
    else:
        mode = args.mode
    with ExitStack() as stack:
        store = stack.enter_context(DatasetStore(args.snapshot, contained_path(root, "lookups/" + refs["lookup"])))
        catalog = stack.enter_context(ServiceCatalog(store, contained_path(root, "services/" + refs["services"])))
        schedule = stack.enter_context(ScheduleStore(catalog, contained_path(root, "schedules/" + refs["schedule"])))
        if mode == "empirical":
            pools = stack.enter_context(PreparedPools(catalog, schedule, contained_path(root, "pools/" + refs["pools"])))
            provider = EmpiricalProvider(pools, Scenario(**load_json(pilot / "scenario.json")))
        elif mode == "replay":
            provider = stack.enter_context(ExactReplayProvider(schedule, contained_path(root, "replay/" + refs["replay"])))
        else:
            raise ValueError("Unknown experiment provider mode")
        if experiment is not None:
            request = TripRequest.from_dict(experiment["request"])
            settings = EpisodeSettings.from_dict(experiment["settings"])
            policy = TransitionPolicy(**experiment["policy"])
            record = EpisodeRecord.from_dict(load_json(args.verify_episode / "episode.json"))
            if episode_manifest["identity"]["trace_sha256"] != digest(record.to_dict()):
                raise ValueError("Episode trace identity mismatch")
            report = EpisodeVerifier(schedule, provider, settings, policy).verify(record, request)
            if report.to_dict() != load_json(args.verify_episode / "verification.json"):
                raise ValueError("Stored verifier report does not reproduce")
            print(report.to_json())
            return 0
        settings = EpisodeSettings(max_duration_minutes=args.max_duration_minutes, max_decisions=args.max_decisions)
        policy = TransitionPolicy(args.cancellation_recovery_minutes, args.missed_boarding_buffer_minutes)
        if args.request:
            request = TripRequest.from_dict(load_json(args.request))
        else:
            flight = schedule.get_flight(args.flight_id or pilot_manifest["identity"]["service_instance_id"])
            request = TripRequest(request_id="smoke-" + flight.flight_id, origin=flight.origin, destination=flight.destination,
                start_at=flight.scheduled_departure_at - timedelta(minutes=settings.initial_boarding_buffer_minutes),
                arrival_deadline=flight.scheduled_arrival_at + timedelta(minutes=120))
        environment = FlightEnvironment(schedule, provider, settings, policy)
        episode_id = f"{mode}:{request.request_id}"
        rollout = None
        if args.interface == "gymnasium":
            agent = RandomAgent(seed=args.agent_seed) if args.agent == "random" else EarliestArrivalAgent()
            adapter = FlightGymEnv(environment, request, max_flights=args.max_flights, episode_id=episode_id)
            try:
                rollout = run_episode(adapter, agent, seed=args.agent_seed)
            finally:
                adapter.close()
            record = rollout.record
        else:
            observation = environment.reset(request, episode_id=episode_id)
            while not observation.done:
                flight = min(observation.flights, key=lambda option: (option.destination != request.destination,
                             option.scheduled_arrival_at, option.scheduled_departure_at, option.flight_id))
                observation = environment.step(flight.flight_id).observation
            record = environment.record
        report = EpisodeVerifier(schedule, provider, settings, policy).verify(record, request)
        if rollout is not None and rollout.total_reward != report.terminal_reward:
            raise ValueError("Gymnasium reward differs from verified terminal reward")
        experiment = {"pilot": pilot.name, "mode": mode, "request": request.to_dict(), "settings": settings.to_dict(),
                      "policy": asdict(policy), "baseline": "direct-destination-then-earliest-scheduled-arrival-v1"}
        if rollout is not None:
            experiment.update(interface="gymnasium", adapter_version="padded-schedule-v1", agent=args.agent,
                              agent_seed=args.agent_seed, max_flights=args.max_flights,
                              baseline=("masked-random-v1" if args.agent == "random"
                                        else "direct-destination-then-earliest-arrival-offer-order-v1"))
        identity = {"kind": "episode", "schema_version": 1, "builder_version": 1,
                    "experiment_sha256": digest(experiment), "trace_sha256": digest(record.to_dict())}
        final, stage = staging(args.output_dir, identity)
        if stage is not None:
            write_json(stage / "experiment.json", experiment)
            write_json(stage / "episode.json", record.to_dict())
            write_json(stage / "verification.json", report.to_dict())
            publish(stage, final, identity)
        print(report.to_json())
        print(final)
    return 0


def run():
    try:
        return main()
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        print(f"Episode failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
