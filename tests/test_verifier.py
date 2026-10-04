from dataclasses import replace
from datetime import timedelta

from test_environment import EnvironmentFixture
from access_fixtures import flight_row
from unittest.mock import patch
from aeroroute.domain.records import EpisodeRecord, TerminalState, TerminationReason
from aeroroute.verification.verifier import EpisodeVerifier, VerificationError


class VerifierTests(EnvironmentFixture):
    def finished(self):
        env, request, flight = self.environment()
        env.reset(request)
        env.step(flight.flight_id)
        return env, EpisodeVerifier(env.schedule, env.provider, env.settings, env.policy)

    def test_valid_trace_reports_metrics_and_weighted_reward(self):
        env, verifier = self.finished()
        report = verifier.verify(env.record, env.request)
        self.assertTrue(report.valid)
        self.assertTrue(report.destination_reached)
        self.assertTrue(report.deadline_met)
        self.assertEqual([(item.name, item.weight, item.score) for item in report.criteria], [
            ("destination_reached", 60, 1.0), ("deadline_met", 25, 1.0),
            ("disruption_free", 10, 1.0), ("completed_within_limit", 5, 1.0),
        ])
        self.assertEqual(report.terminal_reward, 1)
        self.assertEqual(report.verifier_version, "reliability-rubric-v1")
        self.assertEqual(report.elapsed_minutes, 375)
        self.assertEqual(verifier.verify(env.record.to_json(), env.request), report)

    def test_changed_arrival_is_rejected_even_when_trace_consistent(self):
        env, verifier = self.finished()
        record = env.record
        original = record.steps[0]
        later = original.outcome.actual_arrival_at + timedelta(minutes=1)
        altered = replace(original, outcome=replace(original.outcome, actual_arrival_at=later, next_decision_at=later))
        record = replace(record, steps=(altered,), terminal=replace(record.terminal, last_known_at=later, next_decision_at=later))
        with self.assertRaisesRegex(VerificationError, "evidence|transition"):
            verifier.verify(record, env.request)

    def test_changed_source_or_schedule_is_rejected(self):
        env, verifier = self.finished()
        step = env.record.steps[0]
        sample = step.outcome.historical_sample
        tampered_source = replace(sample, source=replace(sample.source, source_row_id=999))
        cases = [replace(step, outcome=replace(step.outcome, historical_sample=tampered_source)),
                 replace(step, selected_flight=replace(step.selected_flight, marketing_carrier="XX"))]
        for tampered in cases:
            with self.subTest(tampered=tampered), self.assertRaises(VerificationError):
                verifier.verify(replace(env.record, steps=(tampered,)), env.request)

    def test_forged_no_flights_and_waiting_timeout_are_rejected(self):
        env, request, flight = self.environment()
        env.reset(request)
        verifier = EpisodeVerifier(env.schedule, env.provider, env.settings, env.policy)
        for reason in (TerminationReason.NO_FEASIBLE_FLIGHTS, TerminationReason.TIME_LIMIT):
            clock = request.start_at if reason is TerminationReason.NO_FEASIBLE_FLIGHTS else request.start_at + timedelta(minutes=env.settings.max_duration_minutes)
            record = EpisodeRecord(schema_version=2, episode_id="forged", request=request, settings=env.settings,
                provenance=env.provenance, steps=(), terminal=TerminalState(current_airport=request.origin,
                next_decision_at=clock, last_known_at=clock), termination_reason=reason)
            with self.assertRaisesRegex(VerificationError, "Premature"):
                verifier.verify(record, request)

    def test_omitted_alternative_flight_is_rejected(self):
        rows = [flight_row("2025-01-02"), flight_row("2025-01-02", CRSDepTime="0900", CRSArrTime="1700",
            DepTime="0915", ArrTime="1715", Flight_Number_Operating_Airline="0013", Flight_Number_Marketing_Airline="0013")]
        env, request, selected = self.environment(rows)
        env.reset(request)
        env.step(selected.flight_id)
        step = env.record.steps[0]
        self.assertEqual(len(step.offered_flight_ids), 2)
        altered = replace(env.record, steps=(replace(step, offered_flight_ids=(selected.flight_id,)),))
        with self.assertRaisesRegex(VerificationError, "Offered"):
            EpisodeVerifier(env.schedule, env.provider, env.settings, env.policy).verify(altered, request)

    def test_verifier_does_not_execute_the_environment(self):
        env, verifier = self.finished()
        with patch("aeroroute.simulation.environment.FlightEnvironment.step", side_effect=AssertionError("environment replay")):
            self.assertTrue(verifier.verify(env.record, env.request).valid)

    def test_unresolved_diversion_metrics_do_not_invent_elapsed_time(self):
        env, request, selected = self.environment([flight_row("2025-01-02", Diverted="1", DivReachedDest="0",
            DivAirportLandings="1", Div1Airport="PHL", Div1AirportID="14100")])
        env.reset(request)
        env.step(selected.flight_id)
        report = EpisodeVerifier(env.schedule, env.provider, env.settings, env.policy).verify(env.record, request)
        self.assertIsNone(report.elapsed_minutes)
        self.assertEqual([item.score for item in report.criteria], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(report.terminal_reward, 0)
        self.assertEqual(report.diversions, 1)

    def _verified(self, *args, **kwargs):
        env, request, flight = self.environment(*args, **kwargs)
        env.reset(request)
        env.step(flight.flight_id)
        return EpisodeVerifier(env.schedule, env.provider, env.settings, env.policy).verify(env.record, request)

    def test_late_clean_arrival_loses_only_the_deadline_weight(self):
        report = self._verified(deadline_offset=-10)
        self.assertEqual([item.score for item in report.criteria], [1.0, 0.0, 1.0, 1.0])
        self.assertEqual(report.terminal_reward, 0.75)

    def test_diverted_arrival_loses_only_the_disruption_weight(self):
        report = self._verified([flight_row(
            "2025-01-02", Diverted="1", DivReachedDest="1", DivAirportLandings="1",
            Div1Airport="ORD", Div1AirportID="13930", DivActualElapsedTime="300", DivArrDelay="15")])
        self.assertEqual([item.score for item in report.criteria], [1.0, 1.0, 0.0, 1.0])
        self.assertEqual(report.terminal_reward, 0.9)

    def test_cutoff_keeps_disruption_and_limit_credit(self):
        report = self._verified(duration=120)
        self.assertEqual([item.score for item in report.criteria], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(report.terminal_reward, 0.15)

    def test_changed_request_and_policy_are_rejected(self):
        env, verifier = self.finished()
        changed_request = replace(env.request, arrival_deadline=env.request.arrival_deadline + timedelta(days=1))
        with self.assertRaises(VerificationError):
            verifier.verify(replace(env.record, request=changed_request), env.request)
        changed_provenance = replace(env.record.provenance, transition_parameters=(("policy_version", "invented"),))
        with self.assertRaises(VerificationError):
            verifier.verify(replace(env.record, provenance=changed_provenance), env.request)
