import unittest
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

from access_fixtures import flight_row
from test_environment import EnvironmentFixture
from aeroroute.domain.records import OutcomeStatus
from aeroroute.outcomes.draws import Scenario
from aeroroute.outcomes.pools import MatchingPolicy, PreparedPools, prepare_pools
from aeroroute.outcomes.providers import EmpiricalProvider
from aeroroute.simulation.environment import FlightEnvironment

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium.utils.env_checker import check_env
except ModuleNotFoundError:
    gym = None

if gym is not None:
    from aeroroute.simulation.gymnasium_env import FlightGymEnv


@unittest.skipIf(gym is None, "Install the rl extra to test the Gymnasium adapter")
class GymnasiumEnvironmentTests(EnvironmentFixture):
    def assert_observations_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def recovery_rows(self, **changes):
        return [
            flight_row("2025-01-02", **changes),
            flight_row("2025-01-02", CRSDepTime="1000", CRSArrTime="1800", DepTime="1000",
                       ArrTime="1800", DepDelay="0", ArrDelay="0",
                       Flight_Number_Operating_Airline="0013", Flight_Number_Marketing_Airline="0013"),
        ]

    def connection_rows(self):
        return [
            flight_row("2025-01-02", Dest="ORD", DestAirportID="13930", DestStateName="Illinois",
                       CRSArrTime="1400", CRSElapsedTime="240", ArrTime="1415", ActualElapsedTime="240"),
            flight_row("2025-01-02", Origin="ORD", OriginAirportID="13930", OriginStateName="Illinois",
                       CRSDepTime="1500", CRSArrTime="1800", CRSElapsedTime="120", DepTime="1500",
                       ArrTime="1800", ActualElapsedTime="120", DepDelay="0", ArrDelay="0"),
        ]

    def test_spaces_schedule_only_encoding_and_normal_terminal(self):
        core, request, flight = self.environment()
        env = FlightGymEnv(core, request, max_flights=3, episode_id="normal")
        self.assertIs(env.environment, core)
        self.assertIsNone(env.record)
        with patch.object(core.provider, "sample", side_effect=AssertionError("future evidence read")):
            observation, info = env.reset(seed=7)
        self.assertIsInstance(env, gym.Env)
        self.assertEqual(env.action_space.n, 4)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(set(observation), {"flights", "state", "time_known", "action_mask"})
        self.assertEqual(observation["flights"].dtype, np.dtype("float64"))
        self.assertEqual(observation["state"].dtype, np.dtype("float64"))
        np.testing.assert_array_equal(observation["flights"][0], [14771, 12478, 60, 360, 1])
        np.testing.assert_array_equal(observation["flights"][1:], np.zeros((2, 5)))
        np.testing.assert_array_equal(observation["state"], [14771, 12478, 0, 60, 480, 1440, 0, 0])
        np.testing.assert_array_equal(observation["time_known"], [1, 1])
        np.testing.assert_array_equal(observation["action_mask"], [1, 0, 0, 0])
        self.assertEqual(info, {
            "flight_ids": (flight.flight_id,), "termination_reason": None, "provider_mode": "replay",
            "scenario_id": None, "scenario_seed": None,
            "reset_seed_scope": "adapter_rng_and_action_space_only", "provider_reseeded": False,
        })
        observation, reward, terminated, truncated, info = env.step(np.int64(0))
        self.assertEqual((reward, terminated, truncated), (1.0, True, False))
        self.assertTrue(env.observation_space.contains(observation))
        np.testing.assert_array_equal(observation["action_mask"], [0, 0, 0, 0])
        np.testing.assert_array_equal(observation["time_known"], [1, 0])
        np.testing.assert_array_equal(observation["state"], [12478, 12478, 375, 0, 480, 1440, 1, 1])
        self.assertEqual(info["termination_reason"], "destination_reached")
        self.assertEqual(info["flight_ids"], ())
        self.assertNotIn("record", info)
        self.assertNotIn("outcome", info)
        self.assertIs(env.record, core.record)
        self.assertEqual(env.record.episode_id, "normal")
        before = core.__dict__.copy()
        for action in (0, 3):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(core.__dict__, before)

    def test_observation_space_sampling_is_finite_and_repeatable(self):
        core, request, _ = self.environment()
        env = FlightGymEnv(core, request, max_flights=3)
        env.observation_space.seed(41)
        first = env.observation_space.sample()
        env.observation_space.seed(41)
        second = env.observation_space.sample()
        self.assert_observations_equal(first, second)
        self.assertTrue(env.observation_space.contains(first))
        self.assertTrue(all(np.isfinite(value).all() for value in first.values()))

    def test_offer_order_and_returned_arrays_do_not_control_actions(self):
        core, request, _ = self.environment(self.recovery_rows())
        env = FlightGymEnv(core, request, max_flights=2)
        observation, info = env.reset()
        expected = tuple(flight.flight_id for flight in core.observation.flights)
        self.assertEqual(info["flight_ids"], expected)
        self.assertEqual(observation["flights"][:, 2].tolist(), [60, 180])
        observation["flights"].fill(0)
        observation["action_mask"].fill(0)
        env.step(1)
        self.assertEqual(env.record.steps[0].selected_flight.flight_id, expected[1])

    def test_invalid_actions_leave_state_and_rng_unchanged(self):
        core, request, _ = self.environment()
        env = FlightGymEnv(core, request, max_flights=3)
        with self.assertRaises(ValueError):
            env.step(0)
        env.reset(seed=8)
        before = core.__dict__.copy()
        rng = env.np_random.bit_generator.state
        action_rng = env.action_space.np_random.bit_generator.state
        with patch.object(core.provider, "sample", side_effect=AssertionError("invalid action sampled")):
            for action in (True, False, np.bool_(True), 0.0, "0", None, [], np.array(0), -1, 1, 2, 3, 4, np.int64(99)):
                with self.subTest(action=repr(action)), self.assertRaises(ValueError):
                    env.step(action)
                self.assertEqual(core.__dict__, before)
                self.assertEqual(env.np_random.bit_generator.state, rng)
                self.assertEqual(env.action_space.np_random.bit_generator.state, action_rng)
        self.assertEqual(env.step(0)[1], 1.0)

    def test_cancellation_recovers_with_reindexed_mask(self):
        core, request, flight = self.environment(self.recovery_rows(Cancelled="1", DepTime="", DepDelay=""))
        env = FlightGymEnv(core, request, max_flights=3)
        observation, info = env.reset()
        np.testing.assert_array_equal(observation["action_mask"], [1, 1, 0, 0])
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        self.assertEqual(observation["state"][2:4].tolist(), [120, 120])
        self.assertEqual(core.steps[0].outcome.status, OutcomeStatus.CANCELLED)
        np.testing.assert_array_equal(observation["action_mask"], [1, 0, 0, 0])
        self.assertNotIn(flight.flight_id, info["flight_ids"])
        self.assertIsNone(env.record)
        self.assertEqual(env.step(0)[1:4], (0.9, True, False))
        self.assertEqual(env.record.cancellation_count, 1)
        self.assertEqual(len(env.record.steps), 2)

    def test_missed_boarding_recovers(self):
        core, request, _ = self.environment(self.recovery_rows(DepTime="0755", DepDelay="-5", ArrTime="1555", ArrDelay="-5"))
        env = FlightGymEnv(core, request, max_flights=2)
        env.reset()
        observation, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        self.assertEqual(observation["state"][0], request.origin.airport_id)
        self.assertEqual(observation["state"][2:4].tolist(), [60, 60])
        self.assertEqual(env.step(0)[1], 0.9)
        self.assertEqual(env.record.missed_boarding_count, 1)

    def test_connection_applies_one_buffer(self):
        core, request, _ = self.environment(self.connection_rows())
        onward = list(core.schedule.all_flights())[1]
        env = FlightGymEnv(core, replace(request, destination=onward.destination), max_flights=1)
        observation, _ = env.reset()
        self.assertEqual(observation["flights"][0, 4], 0)
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        self.assertEqual(observation["state"][3] - observation["state"][2], 45)
        self.assertEqual(info["flight_ids"], (onward.flight_id,))
        self.assertEqual(observation["flights"][0, 4], 1)
        self.assertEqual(env.step(0)[1], 0.75)

    def test_cutoff_is_truncated_without_future_arrival(self):
        core, request, _ = self.environment(duration=120)
        env = FlightGymEnv(core, request, max_flights=1)
        env.reset()
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual((reward, terminated, truncated), (0.15, False, True))
        self.assertEqual(info["termination_reason"], "time_limit")
        self.assertIsNone(env.record.steps[0].outcome.actual_arrival_at)
        np.testing.assert_array_equal(observation["action_mask"], [0, 0])
        np.testing.assert_array_equal(observation["time_known"], [0, 0])
        self.assertEqual(observation["state"][0], 0)
        self.assertEqual(observation["state"][2:4].tolist(), [0, 0])
        self.assertTrue(env.observation_space.contains(observation))

    def test_arrival_at_cutoff_is_terminated_and_late_arrival_still_rewarded(self):
        core, request, _ = self.environment(duration=375, deadline_offset=-10)
        env = FlightGymEnv(core, request, max_flights=1)
        env.reset()
        self.assertEqual(env.step(0)[1:4], (0.75, True, False))
        self.assertEqual(env.record.trip_lateness_minutes, 25)
        self.assertEqual(env.record.arrival_at_destination, env.record.episode_end_at)

    def test_decision_limit_is_truncated(self):
        core, request, _ = self.environment(self.recovery_rows(Cancelled="1", DepTime="", DepDelay=""), decisions=1)
        env = FlightGymEnv(core, request, max_flights=2)
        env.reset()
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual((reward, terminated, truncated), (0.05, False, True))
        self.assertEqual(info["termination_reason"], "decision_limit")
        np.testing.assert_array_equal(observation["action_mask"], [0, 0, 0])

    def test_unresolved_arrival_has_unknown_times(self):
        core, request, _ = self.environment([flight_row("2025-01-02", ArrDelay="", ActualElapsedTime="")])
        env = FlightGymEnv(core, request, max_flights=1)
        env.reset()
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual((reward, terminated, truncated), (0.1, True, False))
        self.assertEqual(info["termination_reason"], "unresolved_outcome")
        np.testing.assert_array_equal(observation["time_known"], [0, 0])
        self.assertEqual(observation["state"][2:4].tolist(), [0, 0])
        self.assertTrue(env.observation_space.contains(observation))

    def test_reset_terminal_acknowledgement_preserves_empty_trace(self):
        core, request, flight = self.environment()
        request = replace(request, start_at=flight.scheduled_departure_at + timedelta(minutes=1))
        env = FlightGymEnv(core, request, max_flights=2)
        observation, info = env.reset(seed=4)
        np.testing.assert_array_equal(observation["action_mask"], [0, 0, 1])
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(observation["state"][-1], 1)
        self.assertEqual(info["termination_reason"], "no_feasible_flights")
        self.assertEqual(info["flight_ids"], ())
        record = env.record
        before = core.__dict__.copy()
        with patch.object(core.provider, "sample", side_effect=AssertionError("acknowledgement sampled")):
            for action in (0, 1, True, 2.0, -1, 3):
                with self.assertRaises(ValueError):
                    env.step(action)
                self.assertEqual(core.__dict__, before)
            observation, reward, terminated, truncated, after_info = env.step(np.int64(2))
        self.assertEqual((reward, terminated, truncated), (0.15, True, False))
        np.testing.assert_array_equal(observation["action_mask"], [0, 0, 0])
        self.assertEqual(after_info, info)
        self.assertIs(env.record, record)
        self.assertEqual(env.record.steps, ())
        self.assertEqual(core.__dict__, before)
        with self.assertRaises(ValueError):
            env.step(2)
        self.assertEqual(env.reset()[0]["action_mask"].tolist(), [0, 0, 1])

    def test_reset_cutoff_acknowledgement_is_truncated(self):
        core, request, _ = self.environment(duration=30)
        env = FlightGymEnv(core, request, max_flights=1)
        observation, info = env.reset()
        self.assertEqual(info["termination_reason"], "time_limit")
        np.testing.assert_array_equal(observation["action_mask"], [0, 1])
        self.assertEqual(env.step(1)[1:4], (0.15, False, True))
        self.assertEqual(env.record.steps, ())

    def test_seed_controls_adapter_and_action_space_not_replay(self):
        core, request, _ = self.environment()
        env = FlightGymEnv(core, request, max_flights=2)
        first_observation, first_info = env.reset(seed=17)
        first_random = env.np_random.integers(1000000, size=10)
        first_actions = [env.action_space.sample() for _ in range(20)]
        env.step(0)
        first_record = env.record
        second_observation, second_info = env.reset(seed=17)
        self.assert_observations_equal(first_observation, second_observation)
        self.assertEqual(first_info, second_info)
        np.testing.assert_array_equal(first_random, env.np_random.integers(1000000, size=10))
        self.assertEqual(first_actions, [env.action_space.sample() for _ in range(20)])
        env.step(0)
        self.assertEqual(env.record, first_record)
        env.reset(seed=18)
        self.assertNotEqual(first_actions, [env.action_space.sample() for _ in range(20)])
        env.step(0)
        self.assertEqual(env.record, first_record)
        rng = env.np_random.bit_generator.state
        action_rng = env.action_space.np_random.bit_generator.state
        env.reset()
        self.assertEqual(env.np_random.bit_generator.state, rng)
        self.assertEqual(env.action_space.np_random.bit_generator.state, action_rng)

    def test_empirical_scenario_remains_frozen_across_reset_seeds(self):
        core, request, _ = self.environment([flight_row("2020-01-02"), flight_row("2025-01-02")])
        catalog = core.schedule.catalog
        pools = PreparedPools(catalog, core.schedule, prepare_pools(
            catalog, core.schedule, self.root / "pools", MatchingPolicy(min_support=1), (2020,),
        ))
        self.addCleanup(pools.close)
        scenario = Scenario.create("frozen-scenario", 42, pools)
        provider = EmpiricalProvider(pools, scenario)
        core = FlightEnvironment(core.schedule, provider, core.settings, core.policy)
        env = FlightGymEnv(core, request, max_flights=1)
        _, info = env.reset(seed=1)
        self.assertEqual(info["provider_mode"], "empirical")
        self.assertEqual(info["scenario_seed"], 42)
        self.assertEqual(info["scenario_id"], "frozen-scenario")
        self.assertFalse(info["provider_reseeded"])
        env.step(0)
        first = env.record
        env.reset(seed=999)
        env.step(0)
        self.assertEqual(env.record, first)
        self.assertIs(provider.scenario, scenario)

    def test_reset_options_are_strict_and_nonsticky(self):
        core, request, flight = self.environment()
        env = FlightGymEnv(core, request, max_flights=1, episode_id="default")
        empty_request = replace(request, start_at=flight.scheduled_departure_at + timedelta(minutes=1))
        env.reset(options={"request": empty_request, "episode_id": "override"})
        self.assertEqual(env.record.request, empty_request)
        self.assertEqual(env.record.episode_id, "override")
        env.reset(seed=3)
        self.assertEqual(core.request, request)
        self.assertEqual(core.episode_id, "default")
        before = core.__dict__.copy()
        rng = env.np_random.bit_generator.state
        for options in ([], False, "", {"unknown": 1}, {"request": None}, {"request": {}},
                        {"episode_id": None}, {"episode_id": ""}, {"episode_id": "  "}, {"episode_id": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                env.reset(seed=99, options=options)
            self.assertEqual(core.__dict__, before)
            self.assertEqual(env.np_random.bit_generator.state, rng)
        for seed in (-1, True, 1.5, "1", np.int64(1)):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                env.reset(seed=seed)
            self.assertEqual(core.__dict__, before)
            self.assertEqual(env.np_random.bit_generator.state, rng)

    def test_constructor_validation(self):
        core, request, _ = self.environment()
        for capacity in (0, -1, True, 1.5, "1", np.int64(1)):
            with self.subTest(max_flights=capacity), self.assertRaises(ValueError):
                FlightGymEnv(core, request, max_flights=capacity)
        for episode_id in (None, "", "  ", 1):
            with self.assertRaises(ValueError):
                FlightGymEnv(core, request, episode_id=episode_id)
        with self.assertRaises(ValueError):
            FlightGymEnv(core, {})
        with self.assertRaises(ValueError):
            FlightGymEnv(object(), request)

    def test_reset_overflow_restores_uninitialized_and_active_core(self):
        core, request, flight = self.environment(self.recovery_rows())
        env = FlightGymEnv(core, request, max_flights=1)
        before = core.__dict__.copy()
        with self.assertRaisesRegex(ValueError, "max_flights=1"):
            env.reset(seed=5)
        self.assertEqual(core.__dict__, before)
        with self.assertRaises(ValueError):
            env.step(0)
        later = replace(request, start_at=flight.scheduled_departure_at + timedelta(minutes=1))
        env.reset(seed=7, options={"request": later, "episode_id": "active"})
        before = core.__dict__.copy()
        rng = env.np_random.bit_generator.state
        with self.assertRaisesRegex(ValueError, "Offered 2 flights exceeds max_flights=1"):
            env.reset(seed=55)
        self.assertEqual(core.__dict__, before)
        self.assertEqual(env.np_random.bit_generator.state, rng)
        self.assertEqual(env.step(0)[1], 1.0)
        self.assertEqual(env.record.episode_id, "active")

    def test_step_overflow_rolls_back_selected_flight_and_record(self):
        rows = self.connection_rows()
        rows.append({**rows[1], "CRSDepTime": "1600", "CRSArrTime": "1900", "DepTime": "1600",
                     "ArrTime": "1900", "Flight_Number_Operating_Airline": "0013",
                     "Flight_Number_Marketing_Airline": "0013"})
        core, request, _ = self.environment(rows)
        request = replace(request, destination=list(core.schedule.all_flights())[1].destination)
        env = FlightGymEnv(core, request, max_flights=1)
        env.reset(seed=8)
        before = core.__dict__.copy()
        rng = env.np_random.bit_generator.state
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "Offered 2 flights exceeds max_flights=1"):
                env.step(0)
            self.assertEqual(core.__dict__, before)
            self.assertIsNone(env.record)
            self.assertEqual(env.np_random.bit_generator.state, rng)
        larger = FlightGymEnv(core, request, max_flights=2)
        larger.reset()
        observation, reward, terminated, truncated, _ = larger.step(0)
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        np.testing.assert_array_equal(observation["action_mask"], [1, 1, 0])
        self.assertEqual(larger.step(0)[1], 0.75)

    def test_encoding_failure_rolls_back_successful_core_step(self):
        core, request, _ = self.environment()
        env = FlightGymEnv(core, request, max_flights=1)
        env.reset()
        before = core.__dict__.copy()
        with patch.object(env, "_encode", side_effect=ValueError("encoding failure")):
            with self.assertRaisesRegex(ValueError, "encoding failure"):
                env.step(0)
        self.assertEqual(core.__dict__, before)
        self.assertEqual(env.step(0)[1], 1.0)

    def test_close_does_not_close_borrowed_resources(self):
        core, request, _ = self.environment()
        env = FlightGymEnv(core, request, max_flights=1)
        with patch.object(core.schedule, "close") as schedule_close, patch.object(core.provider, "close") as provider_close:
            env.close()
            env.close()
            schedule_close.assert_not_called()
            provider_close.assert_not_called()
        env.reset()
        self.assertEqual(env.step(0)[1], 1.0)

    def test_gymnasium_checker_with_valid_masked_sampling(self):
        core, request, _ = self.environment()
        env = FlightGymEnv(core, request, max_flights=1)
        sample = env.action_space.sample

        def sample_valid(mask=None, probability=None):
            return sample(mask=np.array([1, 0], dtype=np.int8))

        with patch.object(env.action_space, "sample", side_effect=sample_valid):
            check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=19)
        for _ in range(10):
            self.assertEqual(env.action_space.sample(mask=observation["action_mask"]), 0)
        with self.assertRaises(ValueError):
            env.step(1)
