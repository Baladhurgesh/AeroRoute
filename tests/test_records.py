import json
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone

from aeroroute.records import (
    AirportRef, DecisionStep, DiversionStop, EpisodeProvenance, EpisodeRecord,
    EpisodeSettings, FlightOption, FlightOutcome, HistoricalSampleRef,
    OutcomeStatus, ScheduleRef, SourceRecordRef, TerminalState, TerminationReason,
    TripRequest,
)


UTC = timezone.utc
START = datetime(2025, 1, 1, 8, tzinfo=UTC)
SFO = AirportRef(airport_id=14771, code="SFO")
ORD = AirportRef(airport_id=13930, code="ORD")
JFK = AirportRef(airport_id=12478, code="JFK")
PHL = AirportRef(airport_id=14100, code="PHL")
SCHEDULE = ScheduleRef(snapshot_id="schedule-2025-01", sha256="a" * 64)


def at(minutes):
    return START + timedelta(minutes=minutes)


def source(row=1):
    return SourceRecordRef(
        dataset_version="b" * 64, source_file="2020/flight-data.zip",
        source_sha256="c" * 64, csv_member="flights.csv", source_row_id=row,
    )


def sample():
    return HistoricalSampleRef(
        source=source(42), flight_date=date(2020, 1, 10),
        matching_pool_id="route-carrier-winter-morning", matching_pool_sha256="d" * 64,
        matching_pool_size=420, fallback_level="route_carrier_season",
    )


def flight(flight_id="f1", origin=SFO, destination=JFK, departure=60, arrival=360):
    return FlightOption(
        flight_id=flight_id, origin=origin, destination=destination,
        marketing_carrier="UA", marketing_flight_number="0012",
        operating_carrier="UA", operating_flight_number="0012",
        scheduled_departure_at=at(departure), scheduled_arrival_at=at(arrival),
        schedule_ref=SCHEDULE, schedule_source=source(5),
    )


def outcome(airport=JFK, departure=60, arrival=360, **changes):
    result = FlightOutcome(
        status=OutcomeStatus.ARRIVED, resulting_airport=airport,
        reached_scheduled_destination=True,
        actual_departure_at=at(departure), actual_arrival_at=at(arrival),
        next_decision_at=at(arrival), historical_sample=sample(),
        timing_method="historical-resample-v1",
    )
    return replace(result, **changes) if changes else result


def step(index=0, origin=SFO, decision=0, ready=60, selected=None, result=None):
    selected = selected or flight()
    return DecisionStep(
        step_index=index, current_airport=origin, decision_at=at(decision),
        ready_to_board_at=at(ready), offered_flight_ids=(selected.flight_id,),
        selected_flight=selected, outcome=result or outcome(),
    )


def episode(steps=None, terminal=None, reason=TerminationReason.DESTINATION_REACHED,
            settings=None, deadline=600):
    return EpisodeRecord(
        schema_version=1, episode_id="episode-1",
        request=TripRequest(request_id="request-1", origin=SFO, destination=JFK,
                            start_at=START, arrival_deadline=at(deadline)),
        settings=settings or EpisodeSettings(max_duration_minutes=1440, max_decisions=6),
        provenance=EpisodeProvenance(
            simulator_version="sim-v1", sampler_version="sampler-v1", seed=7,
            schedule_ref=SCHEDULE, outcome_dataset_version="b" * 64,
            sampler_parameters=(("sampling", "whole_record"),),
        ),
        steps=(step(),) if steps is None else steps,
        terminal=terminal or TerminalState(current_airport=JFK, next_decision_at=at(360), last_known_at=at(360)),
        termination_reason=reason,
    )


class EpisodeRecordTests(unittest.TestCase):
    def test_nonstop_and_derived_metrics(self):
        record = episode()
        self.assertEqual(record.arrival_at_destination, at(360))
        self.assertEqual(record.trip_lateness_minutes, 0)
        self.assertEqual(record.cancellation_count, 0)
        self.assertEqual(record.diversion_count, 0)
        self.assertEqual(record.episode_end_at, at(1440))
        self.assertEqual(record.steps[0].outcome.arrival_delay_minutes(record.steps[0].selected_flight), 0)

    def test_connection_buffer_applied_exactly_once(self):
        first = step(selected=flight(destination=ORD, arrival=180), result=outcome(ORD, arrival=180))
        second = step(1, ORD, 180, 225,
                      flight("f2", ORD, JFK, 225, 360), outcome(JFK, 225, 360))
        record = episode(steps=(first, second))
        self.assertEqual(record.arrival_at_destination, at(360))
        with self.assertRaisesRegex(ValueError, "buffer"):
            episode(steps=(first, replace(second, ready_to_board_at=at(180))))
        with self.assertRaises(ValueError):
            episode(steps=(first, replace(second, ready_to_board_at=at(270))))

    def test_cancellation_recovery_and_separate_buffer(self):
        cancelled = FlightOutcome(
            status=OutcomeStatus.CANCELLED, resulting_airport=SFO,
            reached_scheduled_destination=False, next_decision_at=at(120),
            cancellation_code="B", historical_sample=sample(), timing_method="scheduled-plus-recovery-v1",
        )
        first = step(result=cancelled)
        second = step(1, SFO, 120, 120,
                      flight("f2", departure=150, arrival=400), outcome(departure=150, arrival=400))
        terminal = TerminalState(current_airport=JFK, next_decision_at=at(400), last_known_at=at(400))
        record = episode(steps=(first, second), terminal=terminal)
        self.assertEqual(record.cancellation_count, 1)
        self.assertIsNone(cancelled.arrival_delay_minutes(first.selected_flight))
        configured = EpisodeSettings(max_duration_minutes=1440, max_decisions=6, cancellation_boarding_buffer_minutes=15)
        episode(steps=(first, replace(second, ready_to_board_at=at(135))), terminal=terminal, settings=configured)
        with self.assertRaisesRegex(ValueError, "buffer"):
            episode(steps=(first, second), terminal=terminal, settings=configured)

    def test_late_trip_can_succeed_before_cutoff(self):
        record = episode(deadline=300)
        self.assertEqual(record.trip_lateness_minutes, 60)
        self.assertEqual(record.termination_reason, TerminationReason.DESTINATION_REACHED)

    def test_diversion_reaches_scheduled_destination(self):
        diverted = outcome(status=OutcomeStatus.DIVERTED,
                           diversion_stops=(DiversionStop(airport=ORD, landed_at=at(200), departed_at=at(240)),))
        record = episode(steps=(step(result=diverted),))
        self.assertEqual(record.diversion_count, 1)
        self.assertEqual(diverted.arrival_delay_minutes(flight()), 0)

    def test_stranded_diversion_has_no_scheduled_destination_delay(self):
        diverted = outcome(PHL, arrival=300, status=OutcomeStatus.DIVERTED,
                           reached_scheduled_destination=False,
                           diversion_stops=(DiversionStop(airport=ORD, landed_at=at(180), departed_at=at(220)),
                                            DiversionStop(airport=PHL, landed_at=at(290))))
        terminal = TerminalState(current_airport=PHL, next_decision_at=at(300), last_known_at=at(300))
        record = episode(steps=(step(result=diverted),), terminal=terminal,
                         reason=TerminationReason.NO_FEASIBLE_FLIGHTS)
        self.assertIsNone(diverted.arrival_delay_minutes(flight()))
        self.assertIsNone(record.trip_lateness_minutes)
        self.assertIsNone(record.arrival_at_destination)

    def test_diversion_to_trip_destination_is_trip_success(self):
        diverted = outcome(JFK, status=OutcomeStatus.DIVERTED, reached_scheduled_destination=False,
                           diversion_stops=(DiversionStop(airport=JFK, landed_at=at(350)),))
        record = episode(steps=(step(selected=flight(destination=ORD), result=diverted),))
        self.assertEqual(record.arrival_at_destination, at(360))
        self.assertIsNone(diverted.arrival_delay_minutes(record.steps[0].selected_flight))

    def unresolved_diversion(self):
        result = FlightOutcome(
            status=OutcomeStatus.DIVERTED, resulting_airport=PHL,
            reached_scheduled_destination=False, actual_departure_at=at(60),
            diversion_stops=(DiversionStop(airport=ORD, landed_at=at(180), departed_at=at(220)),
                             DiversionStop(airport=PHL)),
            historical_sample=sample(), timing_method="unresolved",
            unresolved_reason="missing_final_diversion_time",
        )
        return episode(steps=(step(result=result),),
                       terminal=TerminalState(current_airport=PHL, next_decision_at=None, last_known_at=at(220)),
                       reason=TerminationReason.UNRESOLVED_DIVERSION_TIMING)

    def test_unknown_terminal_time_preserves_last_known_time(self):
        record = self.unresolved_diversion()
        self.assertIsNone(record.terminal.next_decision_at)
        self.assertEqual(record.terminal.last_known_at, at(220))
        self.assertIsNone(record.arrival_at_destination)
        self.assertEqual(EpisodeRecord.from_json(record.to_json()), record)

    def test_no_continuation_after_unknown_time(self):
        record = self.unresolved_diversion()
        later = step(1, PHL, 300, 345, flight("f2", PHL, JFK, 360, 500), outcome(JFK, 360, 500))
        with self.assertRaisesRegex(ValueError, "unresolved"):
            replace(record, steps=record.steps + (later,))

    def test_general_unresolved_outcome(self):
        result = FlightOutcome(status=OutcomeStatus.UNRESOLVED, resulting_airport=None,
                               reached_scheduled_destination=None, historical_sample=sample(),
                               timing_method="unresolved", unresolved_reason="missing_outcome_fields")
        record = episode(steps=(step(result=result),),
                         terminal=TerminalState(current_airport=None, next_decision_at=None, last_known_at=START),
                         reason=TerminationReason.UNRESOLVED_OUTCOME)
        self.assertIsNone(record.arrival_at_destination)

    def test_empty_episode_no_flights(self):
        record = episode(steps=(), terminal=TerminalState(current_airport=SFO, next_decision_at=START, last_known_at=START),
                         reason=TerminationReason.NO_FEASIBLE_FLIGHTS)
        self.assertEqual(record.steps, ())
        self.assertIsNone(record.trip_lateness_minutes)

    def test_wait_until_hard_cutoff(self):
        record = episode(steps=(), terminal=TerminalState(current_airport=SFO, next_decision_at=at(1440), last_known_at=at(1440)),
                         reason=TerminationReason.TIME_LIMIT)
        self.assertEqual(record.terminal.next_decision_at, record.episode_end_at)

    def test_inflight_cutoff_is_censored_not_a_fake_arrival(self):
        result = FlightOutcome(status=OutcomeStatus.UNRESOLVED, resulting_airport=None,
                               reached_scheduled_destination=None, actual_departure_at=at(60),
                               historical_sample=sample(), timing_method="cutoff-censored-v1",
                               unresolved_reason="episode_cutoff")
        record = episode(steps=(step(result=result),),
                         terminal=TerminalState(current_airport=None, next_decision_at=None, last_known_at=at(60)),
                         reason=TerminationReason.TIME_LIMIT,
                         settings=EpisodeSettings(max_duration_minutes=180, max_decisions=6))
        self.assertIsNone(record.arrival_at_destination)
        self.assertEqual(record.episode_end_at, at(180))

    def test_diversion_in_progress_at_cutoff_keeps_known_stops(self):
        result = FlightOutcome(
            status=OutcomeStatus.DIVERTED, resulting_airport=None,
            reached_scheduled_destination=None, actual_departure_at=at(60),
            diversion_stops=(DiversionStop(airport=ORD, landed_at=at(180), departed_at=at(220)),),
            historical_sample=sample(), timing_method="cutoff-censored-v1", unresolved_reason="episode_cutoff",
        )
        record = episode(steps=(step(result=result),),
                         terminal=TerminalState(current_airport=None, next_decision_at=None, last_known_at=at(220)),
                         reason=TerminationReason.TIME_LIMIT,
                         settings=EpisodeSettings(max_duration_minutes=300, max_decisions=6))
        self.assertEqual(record.diversion_count, 1)
        self.assertIsNone(record.arrival_at_destination)
        self.assertEqual(EpisodeRecord.from_json(record.to_json()), record)

    def test_cutoff_cannot_be_relabelled_as_missing_data(self):
        result = FlightOutcome(status=OutcomeStatus.UNRESOLVED, resulting_airport=None,
                               reached_scheduled_destination=None, historical_sample=sample(),
                               timing_method="cutoff-censored-v1", unresolved_reason="episode_cutoff")
        with self.assertRaisesRegex(ValueError, "cutoff"):
            episode(steps=(step(result=result),),
                    terminal=TerminalState(current_airport=None, next_decision_at=None, last_known_at=START),
                    reason=TerminationReason.UNRESOLVED_OUTCOME)

    def test_cancelled_flight_instance_cannot_be_selected_again(self):
        cancelled = FlightOutcome(status=OutcomeStatus.CANCELLED, resulting_airport=SFO,
                                  reached_scheduled_destination=False, next_decision_at=at(30),
                                  historical_sample=sample(), timing_method="early-notification-v1")
        first = step(result=cancelled)
        second = step(1, SFO, 30, 30)
        with self.assertRaisesRegex(ValueError, "already attempted"):
            episode(steps=(first, second))

    def test_zero_duration_flight_rejected(self):
        with self.assertRaisesRegex(ValueError, "arrival must follow"):
            outcome(departure=60, arrival=60)

    def test_cutoff_cannot_hide_an_already_observed_arrival(self):
        with self.assertRaisesRegex(ValueError, "cutoff"):
            outcome(next_decision_at=None, unresolved_reason="episode_cutoff")

    def test_actual_arrival_after_cutoff_rejected(self):
        with self.assertRaisesRegex(ValueError, "cutoff"):
            episode(settings=EpisodeSettings(max_duration_minutes=300, max_decisions=6))

    def test_exact_cutoff_arrival_is_allowed(self):
        record = episode(settings=EpisodeSettings(max_duration_minutes=360, max_decisions=6))
        self.assertEqual(record.arrival_at_destination, record.episode_end_at)

    def test_decision_limit(self):
        first = step(selected=flight(destination=ORD, arrival=180), result=outcome(ORD, arrival=180))
        terminal = TerminalState(current_airport=ORD, next_decision_at=at(180), last_known_at=at(180))
        record = episode(steps=(first,), terminal=terminal, reason=TerminationReason.DECISION_LIMIT,
                         settings=EpisodeSettings(max_duration_minutes=1440, max_decisions=1))
        with self.assertRaisesRegex(ValueError, "decision limit"):
            replace(record, settings=EpisodeSettings(max_duration_minutes=1440, max_decisions=2))

    def test_early_historical_departure_before_readiness_rejected(self):
        with self.assertRaisesRegex(ValueError, "actual departure"):
            step(result=outcome(departure=59))

    def test_early_departure_after_readiness_is_allowed(self):
        record = episode(steps=(step(selected=flight(departure=90), result=outcome(departure=80)),))
        self.assertEqual(record.steps[0].outcome.actual_departure_at, at(80))

    def test_schedule_before_readiness_rejected(self):
        with self.assertRaisesRegex(ValueError, "scheduled departure"):
            step(selected=flight(departure=59))

    def test_wrong_origin_rejected(self):
        with self.assertRaisesRegex(ValueError, "origin"):
            step(selected=flight(origin=ORD))

    def test_unoffered_flight_rejected(self):
        with self.assertRaisesRegex(ValueError, "offered"):
            replace(step(), offered_flight_ids=("another-flight",))

    def test_normal_arrival_wrong_airport_rejected(self):
        with self.assertRaisesRegex(ValueError, "destination"):
            step(result=outcome(ORD))

    def test_terminal_location_cannot_claim_success(self):
        with self.assertRaises(ValueError):
            episode(terminal=TerminalState(current_airport=ORD, next_decision_at=at(360), last_known_at=at(360)))

    def test_step_index_and_decision_continuity(self):
        with self.assertRaisesRegex(ValueError, "index"):
            episode(steps=(replace(step(), step_index=1),))
        with self.assertRaisesRegex(ValueError, "decision"):
            episode(steps=(replace(step(), decision_at=at(1)),))

    def test_no_steps_after_destination_reached(self):
        later = step(1, JFK, 360, 405, flight("f2", JFK, ORD, 420, 600), outcome(ORD, 420, 600))
        with self.assertRaisesRegex(ValueError, "destination"):
            episode(steps=(step(), later))

    def test_normal_arrival_decision_has_no_hidden_buffer(self):
        with self.assertRaisesRegex(ValueError, "next_decision_at"):
            outcome(next_decision_at=at(405))

    def test_invalid_diversion_chronology(self):
        with self.assertRaises(ValueError):
            DiversionStop(airport=ORD, landed_at=at(240), departed_at=at(200))
        with self.assertRaises(ValueError):
            outcome(status=OutcomeStatus.DIVERTED,
                    diversion_stops=(DiversionStop(airport=ORD, landed_at=at(250)),
                                     DiversionStop(airport=PHL, landed_at=at(200))))

    def test_cancellation_cannot_claim_arrival(self):
        with self.assertRaises(ValueError):
            outcome(status=OutcomeStatus.CANCELLED, resulting_airport=SFO, reached_scheduled_destination=False)

    def test_utc_required(self):
        for timestamp in (START.replace(tzinfo=None), START.astimezone(timezone(timedelta(hours=-8)))):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                replace(episode().request, start_at=timestamp)

    def test_identity_and_settings_constraints(self):
        for build in (
            lambda: replace(episode().request, destination=SFO),
            lambda: replace(episode().request, arrival_deadline=START),
            lambda: EpisodeSettings(max_duration_minutes=0, max_decisions=6),
            lambda: EpisodeSettings(max_duration_minutes=1440, max_decisions=0),
            lambda: EpisodeSettings(max_duration_minutes=1440, max_decisions=6, connection_buffer_minutes=-1),
            lambda: replace(source(), source_row_id=0),
            lambda: replace(source(), source_sha256="not-a-checksum"),
            lambda: replace(sample(), matching_pool_size=0),
            lambda: replace(sample(), matching_pool_sha256="not-a-checksum"),
        ):
            with self.subTest(build=build), self.assertRaises(ValueError):
                build()

    def test_schedule_and_dataset_provenance_must_match(self):
        with self.assertRaisesRegex(ValueError, "schedule"):
            episode(steps=(replace(step(), selected_flight=replace(flight(), schedule_ref=replace(SCHEDULE, sha256="e" * 64))),))
        changed = replace(sample(), source=replace(source(), dataset_version="e" * 64))
        with self.assertRaisesRegex(ValueError, "dataset"):
            episode(steps=(step(result=outcome(historical_sample=changed)),))

    def test_records_are_immutable(self):
        record = episode()
        with self.assertRaises(FrozenInstanceError):
            record.episode_id = "changed"
        with self.assertRaises((TypeError, ValueError)):
            replace(record, steps=list(record.steps))

    def test_json_roundtrip_preserves_types_nulls_and_flight_numbers(self):
        record = episode()
        encoded = record.to_json()
        decoded = EpisodeRecord.from_json(encoded)
        self.assertEqual(decoded, record)
        self.assertIsInstance(decoded.steps[0].outcome.status, OutcomeStatus)
        self.assertEqual(decoded.steps[0].selected_flight.marketing_flight_number, "0012")
        self.assertEqual(json.loads(encoded)["request"]["start_at"], "2025-01-01T08:00:00Z")
        self.assertNotIn("success", json.loads(encoded))
        self.assertNotIn("score", json.loads(encoded))
        self.assertEqual(EpisodeRecord.from_dict(record.to_dict()), record)

    def test_unknown_schema_or_fields_rejected(self):
        for mutate in (lambda data: data.update(schema_version=2),
                       lambda data: data.update(success=True),
                       lambda data: data["request"].update(unexpected=1)):
            data = episode().to_dict()
            mutate(data)
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                EpisodeRecord.from_dict(data)

    def test_duplicate_json_keys_rejected(self):
        payload = episode().to_json().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            EpisodeRecord.from_json(payload)

    def test_missing_required_field_and_invalid_enum_rejected(self):
        payload = episode().to_dict()
        del payload["request"]["start_at"]
        with self.assertRaises((TypeError, ValueError)):
            EpisodeRecord.from_dict(payload)
        payload = episode().to_dict()
        payload["termination_reason"] = "invalid_action"
        with self.assertRaises(ValueError):
            EpisodeRecord.from_dict(payload)

    def test_early_arrival_preserves_negative_flight_delay(self):
        arrived = outcome(arrival=350)
        record = episode(steps=(step(result=arrived),),
                         terminal=TerminalState(current_airport=JFK, next_decision_at=at(350), last_known_at=at(350)),
                         deadline=350)
        self.assertEqual(arrived.arrival_delay_minutes(flight()), -10)
        self.assertEqual(record.trip_lateness_minutes, 0)

    def test_wrong_json_types_rejected_without_coercion(self):
        for field, value in (("max_decisions", True), ("max_decisions", "6"), ("max_duration_minutes", 2.5)):
            data = episode().to_dict()
            data["settings"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises((TypeError, ValueError)):
                EpisodeRecord.from_dict(data)


if __name__ == "__main__":
    unittest.main()
