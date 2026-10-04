from dataclasses import replace
from datetime import timedelta

from access_fixtures import AccessFixture, flight_row
from aeroroute.catalog.schedule import ScheduleStore, build_schedule
from aeroroute.catalog.services import ResolutionPolicy, ServiceCatalog, build_services
from aeroroute.domain.records import EpisodeRecord, EpisodeSettings, OutcomeStatus, TerminationReason, TripRequest
from aeroroute.outcomes.replay import ExactReplayProvider, prepare_replay
from aeroroute.simulation.environment import FlightEnvironment
from aeroroute.simulation.transitions import TransitionPolicy
from aeroroute.storage.dataset import DatasetStore
from aeroroute.storage.lookup import build_lookup


class EnvironmentFixture(AccessFixture):
    def environment(self, rows=None, duration=1440, decisions=8, deadline_offset=120):
        snapshot = self.snapshot(rows or [flight_row("2025-01-02")])
        store = DatasetStore(snapshot, build_lookup(snapshot, self.root / "lookups"))
        self.addCleanup(store.close)
        catalog = ServiceCatalog(store, build_services(store, self.root / "services", ResolutionPolicy(review_reference="synthetic-review")))
        self.addCleanup(catalog.close)
        schedule = ScheduleStore(catalog, build_schedule(catalog, self.root / "schedules", [(2025, 1)]))
        self.addCleanup(schedule.close)
        provider = ExactReplayProvider(schedule, prepare_replay(schedule, self.root / "replay"))
        self.addCleanup(provider.close)
        selected = next(schedule.all_flights())
        request = TripRequest(request_id="trip", origin=selected.origin, destination=selected.destination,
                              start_at=selected.scheduled_departure_at - timedelta(minutes=60),
                              arrival_deadline=selected.scheduled_arrival_at + timedelta(minutes=deadline_offset))
        env = FlightEnvironment(schedule, provider, EpisodeSettings(max_duration_minutes=duration, max_decisions=decisions), TransitionPolicy())
        return env, request, selected


class EnvironmentTests(EnvironmentFixture):
    def test_normal_flight_and_roundtrip(self):
        env, request, flight = self.environment()
        observation = env.reset(request, episode_id="normal")
        self.assertEqual(observation.ready_to_board_at, flight.scheduled_departure_at)
        result = env.step(flight.flight_id)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.reward, 1)
        self.assertEqual(env.record.termination_reason, TerminationReason.DESTINATION_REACHED)
        self.assertEqual(EpisodeRecord.from_json(env.record.to_json()), env.record)
        with self.assertRaises(ValueError):
            env.step(flight.flight_id)

    def test_cancelled_service_recovers_without_double_buffer(self):
        rows = [flight_row("2025-01-02", Cancelled="1", DepTime="", DepDelay=""),
                flight_row("2025-01-02", CRSDepTime="1000", CRSArrTime="1800", DepTime="1000", ArrTime="1800",
                           DepDelay="0", ArrDelay="0", Flight_Number_Operating_Airline="0013", Flight_Number_Marketing_Airline="0013")]
        env, request, flight = self.environment(rows)
        env.reset(request)
        result = env.step(flight.flight_id)
        self.assertFalse(result.terminated)
        self.assertEqual(result.observation.decision_at, flight.scheduled_departure_at + timedelta(minutes=60))
        self.assertEqual(result.observation.ready_to_board_at, result.observation.decision_at)
        self.assertNotIn(flight.flight_id, [f.flight_id for f in result.observation.flights])
        env.step(result.observation.flights[0].flight_id)
        self.assertEqual(env.record.cancellation_count, 1)

    def test_missed_boarding_retains_origin_and_recovers(self):
        rows = [flight_row("2025-01-02", DepDelay="-5", DepTime="0755", ArrDelay="-5", ArrTime="1555"),
                flight_row("2025-01-02", CRSDepTime="1000", CRSArrTime="1800", DepTime="1000", ArrTime="1800",
                           DepDelay="0", ArrDelay="0", Flight_Number_Operating_Airline="0013", Flight_Number_Marketing_Airline="0013")]
        env, request, flight = self.environment(rows)
        env.reset(request)
        result = env.step(flight.flight_id)
        self.assertEqual(result.observation.current_airport, request.origin)
        self.assertEqual(result.observation.decision_at, flight.scheduled_departure_at)
        self.assertEqual(env.steps[0].outcome.status, OutcomeStatus.MISSED_BOARDING)
        self.assertIsNone(env.steps[0].outcome.actual_departure_at)
        env.step(result.observation.flights[0].flight_id)
        self.assertEqual(env.record.missed_boarding_count, 1)
        with self.assertRaises(ValueError):
            replace(env.record, schema_version=1)

    def test_cutoff_censors_future_arrival(self):
        env, request, flight = self.environment(duration=120)
        env.reset(request)
        result = env.step(flight.flight_id)
        self.assertTrue(result.truncated)
        self.assertEqual(result.reward, 0.15)
        self.assertEqual(env.record.termination_reason, TerminationReason.TIME_LIMIT)
        self.assertIsNone(env.record.steps[0].outcome.actual_arrival_at)
        self.assertEqual(env.record.steps[0].outcome.unresolved_reason, "episode_cutoff")

    def test_arrival_exactly_at_cutoff_is_allowed(self):
        env, request, flight = self.environment(duration=375)
        env.reset(request)
        self.assertEqual(env.step(flight.flight_id).reward, 1)
        self.assertEqual(env.record.arrival_at_destination, env.record.episode_end_at)

    def test_missing_arrival_terminates_without_inventing_time(self):
        env, request, flight = self.environment([flight_row("2025-01-02", ArrDelay="", ActualElapsedTime="")])
        env.reset(request)
        env.step(flight.flight_id)
        self.assertEqual(env.record.termination_reason, TerminationReason.UNRESOLVED_OUTCOME)
        self.assertIsNone(env.record.terminal.next_decision_at)

    def test_stranded_diversion_keeps_endpoint_but_no_gate_time(self):
        env, request, flight = self.environment([flight_row("2025-01-02", Diverted="1", DivReachedDest="0",
                                      DivAirportLandings="1", Div1Airport="PHL", Div1AirportID="14100")])
        env.reset(request)
        env.step(flight.flight_id)
        self.assertEqual(env.record.termination_reason, TerminationReason.UNRESOLVED_DIVERSION_TIMING)
        self.assertEqual(env.record.terminal.current_airport.code, "PHL")
        self.assertIsNone(env.record.arrival_at_destination)

    def test_invalid_action_does_not_modify_state(self):
        env, request, flight = self.environment()
        before = env.reset(request)
        with self.assertRaises(ValueError):
            env.step("not-offered")
        self.assertEqual(env.observation, before)
        self.assertEqual(env.steps, ())

    def test_no_departures_is_a_terminal_episode(self):
        env, request, flight = self.environment()
        request = replace(request, start_at=flight.scheduled_departure_at + timedelta(minutes=1))
        observation = env.reset(request)
        self.assertTrue(observation.done)
        self.assertEqual(env.record.termination_reason, TerminationReason.NO_FEASIBLE_FLIGHTS)

    def test_connection_buffer_is_applied_once(self):
        rows = [flight_row("2025-01-02", Dest="ORD", DestAirportID="13930", DestStateName="Illinois",
                           CRSArrTime="1400", CRSElapsedTime="240", ArrTime="1415", ActualElapsedTime="240"),
                flight_row("2025-01-02", Origin="ORD", OriginAirportID="13930", OriginStateName="Illinois",
                           CRSDepTime="1500", CRSArrTime="1800", CRSElapsedTime="120", DepTime="1500",
                           ArrTime="1800", ActualElapsedTime="120", DepDelay="0", ArrDelay="0")]
        env, request, flight = self.environment(rows)
        onward = list(env.schedule.all_flights())[1]
        request = replace(request, destination=onward.destination)
        env.reset(request)
        result = env.step(flight.flight_id)
        self.assertEqual(result.observation.ready_to_board_at, onward.scheduled_departure_at)
        self.assertEqual(result.observation.ready_to_board_at - result.observation.decision_at, timedelta(minutes=45))
        self.assertEqual(env.step(onward.flight_id).reward, 0.75)

    def test_diversion_cutoff_keeps_only_observed_stop_events(self):
        row = flight_row("2025-01-02", Diverted="1", DivReachedDest="1", DivAirportLandings="1",
                         Div1Airport="ORD", Div1AirportID="13930", Div1WheelsOn="1200", Div1WheelsOff="1300",
                         DivActualElapsedTime="300", DivArrDelay="15")
        env, request, flight = self.environment([row], duration=210)
        env.reset(request)
        result = env.step(flight.flight_id)
        self.assertTrue(result.truncated)
        outcome = result.outcome
        self.assertEqual(outcome.status, OutcomeStatus.DIVERTED)
        self.assertIsNotNone(outcome.diversion_stops[0].landed_at)
        self.assertIsNone(outcome.diversion_stops[0].departed_at)
        self.assertIsNone(outcome.actual_arrival_at)
        self.assertIsNone(outcome.resulting_airport)

    def test_cutoff_preserves_known_stop_departure_with_unknown_landing(self):
        row = flight_row("2025-01-02", Diverted="1", DivReachedDest="1", DivAirportLandings="1",
                         Div1Airport="ORD", Div1AirportID="13930", Div1WheelsOff="1200",
                         DivActualElapsedTime="300", DivArrDelay="15")
        env, request, flight = self.environment([row], duration=210)
        env.reset(request)
        outcome = env.step(flight.flight_id).outcome
        self.assertEqual(outcome.status, OutcomeStatus.DIVERTED)
        self.assertIsNone(outcome.diversion_stops[0].landed_at)
        self.assertIsNotNone(outcome.diversion_stops[0].departed_at)

    def test_diversion_arrival_can_resolve_without_stop_clock(self):
        row = flight_row("2025-01-02", Diverted="1", DivReachedDest="1", DivAirportLandings="1",
                         Div1Airport="ORD", Div1AirportID="13930", DivActualElapsedTime="300", DivArrDelay="15")
        env, request, flight = self.environment([row])
        env.reset(request)
        self.assertEqual(env.step(flight.flight_id).reward, 0.9)
        self.assertEqual(env.record.diversion_count, 1)
        self.assertIsNone(env.record.steps[0].outcome.diversion_stops[0].landed_at)

    def test_cancellation_notification_beyond_cutoff_is_not_observed(self):
        env, request, flight = self.environment([flight_row("2025-01-02", Cancelled="1", DepDelay="", DepTime="")], duration=90)
        env.reset(request)
        result = env.step(flight.flight_id)
        self.assertTrue(result.truncated)
        self.assertEqual(result.outcome.status, OutcomeStatus.UNRESOLVED)
        self.assertEqual(env.record.cancellation_count, 0)

    def test_decision_limit_after_missed_boarding(self):
        env, request, flight = self.environment([flight_row("2025-01-02", DepDelay="-5", DepTime="0755",
                                                          ArrDelay="-5", ArrTime="1555")], decisions=1)
        env.reset(request)
        result = env.step(flight.flight_id)
        self.assertTrue(result.truncated)
        self.assertEqual(env.record.termination_reason, TerminationReason.DECISION_LIMIT)
        self.assertEqual(env.record.missed_boarding_count, 1)
        self.assertEqual(EpisodeRecord.from_json(env.record.to_json()), env.record)

    def test_readiness_after_cutoff_requires_no_outcome_draw(self):
        env, request, flight = self.environment(duration=30)
        self.assertTrue(env.reset(request).done)
        self.assertEqual(env.record.steps, ())
        self.assertEqual(env.record.terminal.last_known_at, env.record.episode_end_at)

    def test_deadline_is_a_metric_not_a_stopping_rule(self):
        env, request, flight = self.environment(deadline_offset=-10)
        env.reset(request)
        self.assertEqual(env.step(flight.flight_id).reward, 0.75)
        self.assertEqual(env.record.trip_lateness_minutes, 25)
