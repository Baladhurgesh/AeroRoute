from dataclasses import replace
from unittest.mock import patch

from access_fixtures import AccessFixture, flight_row
from aeroroute.storage.dataset import DatasetStore
from aeroroute.storage.lookup import build_lookup
from aeroroute.catalog.services import ResolutionPolicy, ServiceCatalog, build_services
from aeroroute.catalog.schedule import ScheduleStore, build_schedule
from aeroroute.outcomes.draws import Scenario, uniform_index
from aeroroute.outcomes.pools import MatchingPolicy, NoMatchingPool, PreparedPools, prepare_pools
from aeroroute.outcomes.providers import EmpiricalProvider
from aeroroute.outcomes.replay import ExactReplayProvider, prepare_replay


class SamplingTests(AccessFixture):
    def providers(self, rows=None, policy=None):
        snapshot = self.snapshot(rows)
        store = DatasetStore(snapshot, build_lookup(snapshot, self.root / "lookups"))
        self.addCleanup(store.close)
        catalog = ServiceCatalog(store, build_services(store, self.root / "services",
                                 ResolutionPolicy(review_reference="synthetic-review")))
        self.addCleanup(catalog.close)
        schedule = ScheduleStore(catalog, build_schedule(catalog, self.root / "schedules", [(2025, 1)]))
        self.addCleanup(schedule.close)
        path = prepare_pools(catalog, schedule, self.root / "pools", policy or MatchingPolicy(min_support=1), (2020,))
        pools = PreparedPools(catalog, schedule, path)
        self.addCleanup(pools.close)
        scenario = Scenario.create("scenario-1", 42, pools)
        replay = ExactReplayProvider(schedule, prepare_replay(schedule, self.root / "replay"))
        self.addCleanup(replay.close)
        return EmpiricalProvider(pools, scenario), replay, schedule

    def test_same_service_same_scenario_across_decisions_and_policies(self):
        empirical, replay, schedule = self.providers()
        flight = next(schedule.all_flights())
        first = empirical.sample(flight, episode_id="policy-a", step_index=1)
        second = empirical.sample(flight, episode_id="policy-b", step_index=3)
        self.assertEqual(first.historical_sample, second.historical_sample)
        self.assertEqual(first.evidence, second.evidence)
        self.assertEqual(first.receipt.draw, second.receipt.draw)
        self.assertNotEqual(first.receipt.trace, second.receipt.trace)
        self.assertEqual(first.historical_sample.flight_date.year, 2020)
        self.assertEqual(first.historical_sample.source.dataset_version, empirical.pools.inventory["dataset_version"])

    def test_service_weighted_pool_retains_disruptions(self):
        rows = [flight_row(), flight_row(Marketing_Airline_Network="XX"),
                flight_row("2020-01-02", Cancelled="1"), flight_row("2025-01-01")]
        empirical, replay, schedule = self.providers(rows, MatchingPolicy(min_support=30))
        result = empirical.sample(next(schedule.all_flights()))
        self.assertEqual(result.historical_sample.matching_pool_size, 2)
        self.assertTrue(empirical.pools.report["low_support"]["count"])
        selected = set()
        for seed in range(30):
            provider = EmpiricalProvider(empirical.pools, Scenario.create("test", seed, empirical.pools))
            selected.add(provider.sample(next(schedule.all_flights())).evidence.flight["cancelled"])
        self.assertEqual(selected, {False, True})

    def test_replay_is_exact_and_keeps_incomplete_evidence(self):
        empirical, replay, schedule = self.providers([flight_row(), flight_row("2025-01-01", ArrDelay="", ActualElapsedTime="")])
        flight = next(schedule.all_flights())
        result = replay.sample(flight)
        self.assertEqual(result.historical_sample.flight_date.year, 2025)
        self.assertEqual(result.historical_sample.matching_pool_size, 1)
        self.assertEqual(result.historical_sample.fallback_level, "exact_replay")
        self.assertIsNone(result.evidence.flight["actual_arrival"].value)
        self.assertEqual(result.historical_sample.source.source_row_id, flight.schedule_source.source_row_id)
        self.assertEqual(replay.sample(flight).historical_sample, result.historical_sample)

    def test_tampered_flight_is_rejected(self):
        empirical, replay, schedule = self.providers()
        flight = replace(next(schedule.all_flights()), marketing_carrier="FAKE")
        with self.assertRaises(ValueError):
            empirical.sample(flight)
        with self.assertRaises(ValueError):
            replay.sample(flight)

    def test_no_route_fallback(self):
        empirical, replay, schedule = self.providers([flight_row(), flight_row("2025-01-01", Dest="PHL", DestAirportID="14100",
            DestStateName="Pennsylvania")])
        with self.assertRaises(NoMatchingPool):
            empirical.sample(next(schedule.all_flights()))
        self.assertEqual(empirical.pools.report["unsupported"]["fraction"], 1)

    def test_fitting_scope_cannot_include_heldout_year(self):
        empirical, replay, schedule = self.providers()
        with self.assertRaises(ValueError):
            prepare_pools(empirical.pools.catalog, schedule, self.root / "invalid", MatchingPolicy(), (2025,))

    def test_scenario_binding_cannot_be_changed_silently(self):
        empirical, replay, schedule = self.providers()
        with self.assertRaises(ValueError):
            EmpiricalProvider(empirical.pools, replace(empirical.scenario, pool_set="a" * 64))

    def test_all_fallback_levels_and_time_of_day_loss(self):
        cases = [
            (flight_row(), 0),
            (flight_row("2020-01-02"), 1),
            (flight_row(CRSDepTime="1400", CRSArrTime="2200"), 2),
            (flight_row("2020-04-01"), 3),
            (flight_row(Operating_Airline="XX"), 4),
        ]
        for row, level in cases:
            with self.subTest(level=level):
                fixture = SamplingTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                empirical, replay, schedule = fixture.providers([row, flight_row("2025-01-01")])
                result = empirical.sample(next(schedule.all_flights()))
                self.assertTrue(result.historical_sample.fallback_level.startswith(f"level_{level}:"))
                self.assertEqual(empirical.pools.report["lost_time_of_day"]["fraction"], float(level >= 2))
                self.assertEqual(empirical.pools.report["carrier_fallback"]["fraction"], float(level == 4))

    def test_retain_time_of_day_hierarchy(self):
        policy = MatchingPolicy(levels=(("carrier", "quarter", "weekday", "bucket"), ("carrier", "quarter", "bucket"),
                                       ("carrier", "bucket"), ("carrier",), ()), min_support=1)
        empirical, replay, schedule = self.providers([flight_row("2020-04-01"), flight_row("2025-01-01")], policy)
        result = empirical.sample(next(schedule.all_flights()))
        self.assertTrue(result.historical_sample.fallback_level.startswith("level_2:"))
        self.assertEqual(empirical.pools.report["lost_time_of_day"]["fraction"], 0)

    def test_validation_report_is_request_weighted_and_pre_holdout(self):
        empirical, replay, schedule = self.providers([flight_row(), flight_row("2024-01-01"), flight_row("2025-01-01")])
        with self.assertRaises(ValueError):
            empirical.pools.validation_report([next(schedule.all_flights()).flight_id])
        catalog = empirical.pools.catalog
        validation = ScheduleStore(catalog, build_schedule(catalog, self.root / "validation-schedule", [(2024, 1)]))
        self.addCleanup(validation.close)
        pools = PreparedPools(catalog, validation, prepare_pools(catalog, validation, self.root / "validation-pools", MatchingPolicy(min_support=1), (2020,)))
        self.addCleanup(pools.close)
        service_id = next(validation.all_flights()).flight_id
        report = pools.validation_report([service_id, service_id])
        self.assertEqual(report["request_count"], 2)
        self.assertEqual(report["unique_service_count"], 1)
        self.assertEqual(report["unsupported"]["count"], 0)

    def test_rejection_algorithm_retries_without_modulo_bias(self):
        class FakeHash:
            def __init__(self, value):
                self.value = value
            def digest(self):
                return self.value.to_bytes(32, "big")
        with patch("aeroroute.outcomes.draws.hashlib.sha256", side_effect=[FakeHash(2 ** 256 - 1), FakeHash(4)]) as mocked:
            self.assertEqual(uniform_index({"seed": 0}, 3), 1)
            self.assertEqual(mocked.call_count, 2)

    def test_uniform_index_bounds_and_repeatability(self):
        self.assertEqual(uniform_index({"seed": 4}, 1), 0)
        self.assertEqual(uniform_index({"seed": 4}, 37), 24)
        with self.assertRaises(ValueError):
            uniform_index({}, 0)
