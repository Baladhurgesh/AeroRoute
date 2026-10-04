from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from importlib.util import find_spec
import unittest
from unittest.mock import Mock, patch

from access_fixtures import flight_row
from test_environment import EnvironmentFixture
from aeroroute.domain.records import TerminationReason
from aeroroute.verification.verifier import EpisodeVerifier


HAS_NUMPY = find_spec("numpy") is not None
HAS_GYMNASIUM = find_spec("gymnasium") is not None

if HAS_NUMPY:
    import numpy as np
    from aeroroute.simulation.agents import EarliestArrivalAgent, RandomAgent, make_agent, run_episode


def encoded(flights, mask):
    return {
        "flights": np.asarray(flights, dtype=np.float64),
        "state": np.zeros(8, dtype=np.float64),
        "time_known": np.ones(2, dtype=np.int8),
        "action_mask": np.asarray(mask, dtype=np.int8),
    }


@unittest.skipUnless(HAS_NUMPY, "Install the rl extra to test encoded-observation agents")
class AgentTests(unittest.TestCase):
    def test_random_is_seeded_and_selects_only_masked_slots(self):
        observation = encoded([[1, 2, 60, 120, 1]] * 5, [0, 1, 0, 1, 0, 0])
        first, second = RandomAgent(seed=27), RandomAgent(seed=27)
        sequence = [first(observation) for _ in range(100)]
        self.assertEqual(sequence, [second(observation) for _ in range(100)])
        self.assertEqual(set(sequence), {1, 3})
        different = RandomAgent(seed=28)
        self.assertNotEqual(sequence, [different(observation) for _ in range(100)])
        self.assertTrue(all(type(action) is int for action in sequence))

    def test_random_does_not_use_global_random_state(self):
        observation = encoded([[1, 2, 60, 120, 1]] * 2, [1, 1, 0])
        with patch("random.choice", side_effect=AssertionError("global random state")):
            self.assertIn(RandomAgent()(observation), (0, 1))

    def test_earliest_prefers_direct_destination_before_arrival(self):
        observation = encoded([[1, 3, 1, 2, 0], [1, 2, 60, 120, 1]], [1, 1, 0])
        self.assertEqual(EarliestArrivalAgent()(observation), 1)

    def test_earliest_orders_by_arrival_departure_then_row_order(self):
        observation = encoded([
            [1, 2, 30, 120, 1],
            [1, 2, 60, 100, 1],
            [1, 2, 40, 100, 1],
            [1, 2, 40, 100, 1],
        ], [1, 1, 1, 1, 0])
        self.assertEqual(EarliestArrivalAgent()(observation), 2)
        observation["flights"][:, 4] = 0
        self.assertEqual(EarliestArrivalAgent()(observation), 2)

    def test_policies_ignore_masked_padding_without_mutating_observations(self):
        observation = encoded([[0, 0, 0, 0, 0], [1, 2, 60, 120, 1], [0, 0, -100, -100, 1]], [0, 1, 0, 0])
        observation["flights"][0] = np.nan
        before = {key: value.copy() for key, value in observation.items()}
        for agent in (RandomAgent(), EarliestArrivalAgent()):
            self.assertEqual(agent(observation), 1)
        for key in before:
            np.testing.assert_array_equal(observation[key], before[key])

    def test_policies_accept_only_encoded_observations_and_acknowledgement(self):
        observation = encoded([[0, 0, 0, 0, 0]] * 3, [0, 0, 0, 1])
        observation["state"][7] = 1
        for agent in (RandomAgent(), EarliestArrivalAgent()):
            self.assertEqual(agent(observation), 3)
            with self.assertRaisesRegex(ValueError, "encoded observation"):
                agent(object())

    def test_invalid_observations_fail_clearly(self):
        cases = [
            {},
            encoded([[1, 2, 60, 120, 1]], [0, 0]),
            encoded([[1, 2, 60, 120, 1]], [2, 0]),
            encoded([[1, 2, 60, 120, 1]], [1]),
            encoded([[1, 2, 60, 120, 1]], [1, 1]),
            encoded([[1, 2, 60, float("inf"), 1]], [1, 0]),
            encoded([[1, 2, 60, 120, 2]], [1, 0]),
            encoded([[1, 2, 60, 120]], [1, 0]),
        ]
        for observation in cases:
            for agent in (RandomAgent(), EarliestArrivalAgent()):
                with self.subTest(observation=observation, agent=type(agent).__name__), self.assertRaises(ValueError):
                    agent(observation)

    def test_factory_and_seed_validation(self):
        self.assertIsInstance(make_agent("random", seed=12), RandomAgent)
        self.assertIsInstance(make_agent("earliest-arrival"), EarliestArrivalAgent)
        with self.assertRaisesRegex(ValueError, "Unknown agent"):
            make_agent("trained")
        for seed in (True, "12", 1.5):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                RandomAgent(seed=seed)


@unittest.skipUnless(HAS_NUMPY and HAS_GYMNASIUM, "Install the rl extra to test Gymnasium rollouts")
class RolloutTests(EnvironmentFixture):
    def adapter(self, environment, request):
        from aeroroute.simulation.gymnasium_env import FlightGymEnv
        return FlightGymEnv(environment, request, max_flights=8, episode_id="agent-test")

    def verify(self, environment, request, result):
        self.assertIs(result.record, environment.record)
        report = EpisodeVerifier(environment.schedule, environment.provider, environment.settings, environment.policy).verify(result.record, request)
        self.assertTrue(report.valid)
        self.assertEqual(result.total_reward, report.terminal_reward)
        return report

    def test_both_policies_complete_verified_artifact_backed_episodes(self):
        rows = [flight_row("2025-01-02"), flight_row("2025-01-02", CRSDepTime="0900", CRSArrTime="1700",
            DepTime="0915", ArrTime="1715", Flight_Number_Operating_Airline="0013", Flight_Number_Marketing_Airline="0013")]
        environment, request, _ = self.environment(rows)
        for name in ("random", "earliest-arrival"):
            with self.subTest(agent=name):
                env = self.adapter(environment, request)
                with patch.object(env, "reset", wraps=env.reset) as reset:
                    result = run_episode(env, make_agent(name, seed=5), seed=5)
                reset.assert_called_once_with(seed=5, options=None)
                self.assertTrue(result.terminated)
                self.assertFalse(result.truncated)
                self.assertTrue(self.verify(environment, request, result).destination_reached)
                self.assertEqual(len(result.record.steps), 1)
                with self.assertRaises(FrozenInstanceError):
                    result.total_reward = -1
                with self.assertRaises(ValueError):
                    env.step(0)

    def test_cancellation_recovery_is_verified_for_both_agents(self):
        rows = [flight_row("2025-01-02", Cancelled="1", DepTime="", DepDelay=""),
                flight_row("2025-01-02", CRSDepTime="1000", CRSArrTime="1800", DepTime="1000", ArrTime="1800",
                           DepDelay="0", ArrDelay="0", Flight_Number_Operating_Airline="0013",
                           Flight_Number_Marketing_Airline="0013")]
        environment, request, _ = self.environment(rows)
        for name in ("random", "earliest-arrival"):
            with self.subTest(agent=name):
                result = run_episode(self.adapter(environment, request), make_agent(name, seed=1), seed=1)
                report = self.verify(environment, request, result)
                self.assertEqual(report.cancellations, 1)
                self.assertEqual(report.decisions, 2)
                self.assertTrue(report.destination_reached)

    def test_unresolved_diversion_terminates_and_verifies_without_new_action(self):
        environment, request, _ = self.environment([flight_row(
            "2025-01-02", Diverted="1", DivReachedDest="0", DivAirportLandings="1",
            Div1Airport="PHL", Div1AirportID="14100")])
        for name in ("random", "earliest-arrival"):
            with self.subTest(agent=name):
                result = run_episode(self.adapter(environment, request), make_agent(name))
                report = self.verify(environment, request, result)
                self.assertTrue(result.terminated)
                self.assertFalse(result.truncated)
                self.assertEqual(report.diversions, 1)
                self.assertEqual(report.decisions, 1)
                self.assertEqual(result.record.termination_reason, TerminationReason.UNRESOLVED_DIVERSION_TIMING)
                self.assertIsNone(report.elapsed_minutes)
                self.assertIsNone(result.record.terminal.next_decision_at)
                self.assertEqual(result.total_reward, 0)

    def test_initial_terminal_acknowledgement_does_not_sample_or_add_a_decision(self):
        environment, request, selected = self.environment()
        request = replace(request, start_at=selected.scheduled_departure_at + timedelta(minutes=1))
        for name in ("random", "earliest-arrival"):
            env = self.adapter(environment, request)
            with patch.object(environment.provider, "sample", side_effect=AssertionError("hidden evidence")), \
                    patch.object(env, "step", wraps=env.step) as step:
                result = run_episode(env, make_agent(name))
            step.assert_called_once_with(8)
            self.assertTrue(result.terminated)
            self.assertFalse(result.truncated)
            self.assertEqual(result.record.steps, ())
            self.assertEqual(result.record.termination_reason, TerminationReason.NO_FEASIBLE_FLIGHTS)
            self.verify(environment, request, result)

    def test_initial_time_limit_acknowledgement(self):
        environment, request, _ = self.environment(duration=30)
        for name in ("random", "earliest-arrival"):
            result = run_episode(self.adapter(environment, request), make_agent(name))
            self.assertFalse(result.terminated)
            self.assertTrue(result.truncated)
            self.assertEqual(result.record.steps, ())
            self.assertEqual(result.record.termination_reason, TerminationReason.TIME_LIMIT)
            self.verify(environment, request, result)

    def test_time_limit_stops_rollout_and_verifies_censored_record(self):
        environment, request, _ = self.environment(duration=120)
        for name in ("random", "earliest-arrival"):
            result = run_episode(self.adapter(environment, request), make_agent(name))
            self.assertFalse(result.terminated)
            self.assertTrue(result.truncated)
            self.assertEqual(result.record.termination_reason, TerminationReason.TIME_LIMIT)
            self.assertIsNone(result.record.steps[0].outcome.actual_arrival_at)
            self.verify(environment, request, result)

    def test_decision_limit_stops_rollout(self):
        environment, request, _ = self.environment([flight_row("2025-01-02", DepDelay="-5", DepTime="0755",
            ArrDelay="-5", ArrTime="1555")], decisions=1)
        for name in ("random", "earliest-arrival"):
            result = run_episode(self.adapter(environment, request), make_agent(name))
            self.assertFalse(result.terminated)
            self.assertTrue(result.truncated)
            self.assertEqual(result.record.termination_reason, TerminationReason.DECISION_LIMIT)
            self.verify(environment, request, result)

    def test_runner_forwards_reset_arguments_accumulates_rewards_and_hides_info(self):
        environment, request, selected = self.environment()
        environment.reset(request)
        environment.step(selected.flight_id)
        first = encoded([[1, 2, 60, 120, 1]], [1, 0])
        second = encoded([[1, 2, 70, 130, 1]], [1, 0])
        hidden_info = {"provider": object(), "historical_evidence": object()}
        env = Mock()
        env.record = environment.record
        env.reset.return_value = (first, hidden_info)
        env.step.side_effect = [(second, 0.25, False, False, hidden_info), (second, 0.75, False, True, hidden_info)]
        agent = Mock(side_effect=EarliestArrivalAgent())
        options = {"episode_id": "custom"}
        result = run_episode(env, agent, seed=91, options=options)
        env.reset.assert_called_once_with(seed=91, options=options)
        self.assertEqual(env.step.call_count, 2)
        self.assertIs(agent.call_args_list[0].args[0], first)
        self.assertIs(agent.call_args_list[1].args[0], second)
        self.assertTrue(all(len(call.args) == 1 and not call.kwargs for call in agent.call_args_list))
        self.assertEqual(result.total_reward, 1.0)
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertIs(result.record, environment.record)

    def test_runner_rejects_missing_terminal_record(self):
        env = Mock()
        env.reset.return_value = (encoded([[1, 2, 60, 120, 1]], [1, 0]), {})
        env.step.return_value = ({}, 0, True, False, {})
        env.record = None
        with self.assertRaisesRegex(ValueError, "EpisodeRecord"):
            run_episode(env, EarliestArrivalAgent())
