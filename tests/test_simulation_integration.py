import contextlib
import io
import importlib.util
import unittest
from pathlib import Path

from access_fixtures import AccessFixture, flight_row
from aeroroute.cli.build_data_access import main as build
from aeroroute.cli.simulate import main as simulate
from aeroroute.domain.records import EpisodeRecord
from aeroroute.storage.identity import load_json


class SimulationIntegrationTests(AccessFixture):
    def test_both_providers_produce_independently_reproducible_episode_artifacts(self):
        snapshot = self.snapshot([flight_row("2020-01-01"), flight_row("2025-01-02")])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            build(["--snapshot", str(snapshot), "--output-dir", str(self.root / "access"), "--stage", "pilot",
                   "--review-reference", "synthetic-review", "--schedule-periods", "2025-01", "--fitting-years", "2020",
                   "--min-support", "1", "--benchmark-draws", "2"])
        pilot = Path(output.getvalue().splitlines()[-1])
        for mode in ("replay", "empirical"):
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(simulate(["--snapshot", str(snapshot), "--pilot", str(pilot), "--mode", mode,
                                           "--output-dir", str(self.root / "episodes")]), 0)
                artifact = Path(output.getvalue().splitlines()[-1])
                record = EpisodeRecord.from_dict(load_json(artifact / "episode.json"))
                self.assertEqual(record.arrival_at_destination.year, 2025)
                self.assertEqual(record.steps[0].outcome.historical_sample.flight_date.year, 2025 if mode == "replay" else 2020)
                self.assertEqual(load_json(artifact / "verification.json")["terminal_reward"], 1)
                self.assertEqual(simulate(["--snapshot", str(snapshot), "--pilot", str(pilot),
                                           "--verify-episode", str(artifact)]), 0)

    @unittest.skipUnless(importlib.util.find_spec("gymnasium") and importlib.util.find_spec("numpy"), "requires RL extra")
    def test_gymnasium_agents_publish_and_reverify_both_provider_modes(self):
        snapshot = self.snapshot([flight_row("2020-01-01"), flight_row("2025-01-02")])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            build(["--snapshot", str(snapshot), "--output-dir", str(self.root / "access"), "--stage", "pilot",
                   "--review-reference", "synthetic-review", "--schedule-periods", "2025-01", "--fitting-years", "2020",
                   "--min-support", "1", "--benchmark-draws", "2"])
        pilot = Path(output.getvalue().splitlines()[-1])
        for mode in ("replay", "empirical"):
            for agent in ("earliest-arrival", "random"):
                with self.subTest(mode=mode, agent=agent):
                    arguments = ["--snapshot", str(snapshot), "--pilot", str(pilot), "--mode", mode,
                                 "--interface", "gymnasium", "--agent", agent, "--agent-seed", "42", "--max-flights", "4",
                                 "--output-dir", str(self.root / "episodes")]
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(simulate(arguments), 0)
                    artifact = Path(output.getvalue().splitlines()[-1])
                    record = EpisodeRecord.from_dict(load_json(artifact / "episode.json"))
                    experiment = load_json(artifact / "experiment.json")
                    self.assertEqual(experiment["interface"], "gymnasium")
                    self.assertEqual(experiment["agent"], agent)
                    self.assertEqual(experiment["agent_seed"], 42)
                    self.assertEqual(experiment["max_flights"], 4)
                    expected_seed = load_json(pilot / "scenario.json")["seed"] if mode == "empirical" else 0
                    self.assertEqual(record.provenance.seed, expected_seed)
                    self.assertEqual(record.steps[0].outcome.historical_sample.flight_date.year,
                                     2025 if mode == "replay" else 2020)
                    self.assertEqual(load_json(artifact / "verification.json")["terminal_reward"], 1)
                    with contextlib.redirect_stdout(io.StringIO()) as repeated:
                        self.assertEqual(simulate(arguments), 0)
                    self.assertEqual(Path(repeated.getvalue().splitlines()[-1]), artifact)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(simulate(["--snapshot", str(snapshot), "--pilot", str(pilot),
                                                   "--verify-episode", str(artifact)]), 0)

    def test_invalid_agent_options_fail_before_loading_artifacts(self):
        for arguments in (["--agent", "random"], ["--agent-seed", "-1"], ["--max-flights", "0"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    simulate(["--snapshot", "missing", "--pilot", "missing", *arguments])
                self.assertEqual(caught.exception.code, 2)
