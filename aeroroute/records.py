import json
import re
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta
from enum import Enum
from functools import cache
from types import UnionType
from typing import get_args, get_origin, get_type_hints


def require(condition, message):
    if not condition:
        raise ValueError(message)


@cache
def annotations(record_type):
    return get_type_hints(record_type)


def check_type(value, expected, name):
    origin, args = get_origin(expected), get_args(expected)
    if origin is UnionType:
        if value is None and type(None) in args:
            return
        expected = next(option for option in args if option is not type(None))
        return check_type(value, expected, name)
    if origin is tuple:
        require(type(value) is tuple, f"{name} must be an immutable tuple")
        if len(args) == 2 and args[1] is Ellipsis:
            for item in value:
                check_type(item, args[0], name)
        else:
            require(len(value) == len(args), f"{name} has the wrong tuple length")
            for item, item_type in zip(value, args):
                check_type(item, item_type, name)
        return
    require(type(value) is expected, f"{name} must be {expected.__name__}")
    if expected is datetime:
        require(value.tzinfo is not None and value.utcoffset() == timedelta(0), f"{name} must be timezone-aware UTC")
    if expected is str:
        require(bool(value.strip()), f"{name} must not be blank")


def encode(value):
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    return value


def decode(value, expected):
    origin, args = get_origin(expected), get_args(expected)
    if origin is UnionType:
        if value is None and type(None) in args:
            return None
        return decode(value, next(option for option in args if option is not type(None)))
    if origin is tuple:
        require(type(value) is list, "JSON tuple fields must be arrays")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(item, args[0]) for item in value)
        require(len(value) == len(args), "JSON tuple has the wrong length")
        return tuple(decode(item, item_type) for item, item_type in zip(value, args))
    if isinstance(expected, type) and issubclass(expected, Record):
        return expected.from_dict(value)
    if expected in (date, datetime):
        require(type(value) is str, "JSON dates and timestamps must be strings")
        return expected.fromisoformat(value)
    if isinstance(expected, type) and issubclass(expected, Enum):
        require(type(value) is str, "JSON enum values must be strings")
        return expected(value)
    check_type(value, expected, "JSON value")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


class Record:
    def __post_init__(self):
        for name, expected in annotations(type(self)).items():
            check_type(getattr(self, name), expected, name)

    def to_dict(self):
        return {field.name: encode(getattr(self, field.name)) for field in fields(self)}

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data):
        require(type(data) is dict, f"{cls.__name__} must be a JSON object")
        hints = annotations(cls)
        require(not set(data) - set(hints), f"Unknown fields in {cls.__name__}: {set(data) - set(hints)}")
        return cls(**{name: decode(value, hints[name]) for name, value in data.items()})

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text, object_pairs_hook=unique_object))


def check_sha256(value, name):
    require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{name} must be a lowercase SHA-256 digest")


class OutcomeStatus(str, Enum):
    ARRIVED = "arrived"
    CANCELLED = "cancelled"
    DIVERTED = "diverted"
    UNRESOLVED = "unresolved"


class TerminationReason(str, Enum):
    DESTINATION_REACHED = "destination_reached"
    NO_FEASIBLE_FLIGHTS = "no_feasible_flights"
    TIME_LIMIT = "time_limit"
    DECISION_LIMIT = "decision_limit"
    UNRESOLVED_DIVERSION_TIMING = "unresolved_diversion_timing"
    UNRESOLVED_OUTCOME = "unresolved_outcome"


@dataclass(frozen=True, kw_only=True)
class AirportRef(Record):
    airport_id: int
    code: str

    def __post_init__(self):
        super().__post_init__()
        require(self.airport_id > 0, "airport_id must be positive")
        require(re.fullmatch(r"[A-Z0-9]{3,4}", self.code) is not None, "Invalid airport code")


@dataclass(frozen=True, kw_only=True)
class SourceRecordRef(Record):
    dataset_version: str
    source_file: str
    source_sha256: str
    csv_member: str
    source_row_id: int

    def __post_init__(self):
        super().__post_init__()
        check_sha256(self.dataset_version, "dataset_version")
        check_sha256(self.source_sha256, "source_sha256")
        require(self.source_row_id > 0, "source_row_id must be a positive original data-record ordinal")


@dataclass(frozen=True, kw_only=True)
class ScheduleRef(Record):
    snapshot_id: str
    sha256: str

    def __post_init__(self):
        super().__post_init__()
        check_sha256(self.sha256, "schedule sha256")


@dataclass(frozen=True, kw_only=True)
class HistoricalSampleRef(Record):
    source: SourceRecordRef
    flight_date: date
    matching_pool_id: str
    matching_pool_sha256: str
    matching_pool_size: int
    fallback_level: str

    def __post_init__(self):
        super().__post_init__()
        check_sha256(self.matching_pool_sha256, "matching_pool_sha256")
        require(self.matching_pool_size > 0, "matching_pool_size must be positive")


@dataclass(frozen=True, kw_only=True)
class TripRequest(Record):
    request_id: str
    origin: AirportRef
    destination: AirportRef
    start_at: datetime
    arrival_deadline: datetime

    def __post_init__(self):
        super().__post_init__()
        require(self.origin.airport_id != self.destination.airport_id, "Trip origin and destination must differ")
        require(self.arrival_deadline > self.start_at, "arrival_deadline must follow start_at")


@dataclass(frozen=True, kw_only=True)
class EpisodeSettings(Record):
    max_duration_minutes: int
    max_decisions: int
    initial_boarding_buffer_minutes: int = 60
    connection_buffer_minutes: int = 45
    cancellation_boarding_buffer_minutes: int = 0

    def __post_init__(self):
        super().__post_init__()
        require(self.max_duration_minutes > 0 and self.max_decisions > 0, "Episode limits must be positive")
        require(min(self.initial_boarding_buffer_minutes, self.connection_buffer_minutes,
                    self.cancellation_boarding_buffer_minutes) >= 0, "Boarding buffers cannot be negative")


@dataclass(frozen=True, kw_only=True)
class EpisodeProvenance(Record):
    simulator_version: str
    sampler_version: str
    seed: int
    schedule_ref: ScheduleRef
    outcome_dataset_version: str
    sampler_parameters: tuple[tuple[str, str], ...] = ()
    transition_parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        super().__post_init__()
        require(self.seed >= 0, "seed must be nonnegative")
        check_sha256(self.outcome_dataset_version, "outcome_dataset_version")
        for parameters in (self.sampler_parameters, self.transition_parameters):
            keys = [key for key, value in parameters]
            require(len(keys) == len(set(keys)), "Duplicate configuration parameter")


@dataclass(frozen=True, kw_only=True)
class FlightOption(Record):
    flight_id: str
    origin: AirportRef
    destination: AirportRef
    marketing_carrier: str
    marketing_flight_number: str
    operating_carrier: str
    operating_flight_number: str
    scheduled_departure_at: datetime
    scheduled_arrival_at: datetime
    schedule_ref: ScheduleRef
    schedule_source: SourceRecordRef

    def __post_init__(self):
        super().__post_init__()
        require(self.origin.airport_id != self.destination.airport_id, "Flight origin and destination must differ")
        require(self.scheduled_arrival_at > self.scheduled_departure_at, "Scheduled arrival must follow departure")


@dataclass(frozen=True, kw_only=True)
class DiversionStop(Record):
    airport: AirportRef
    landed_at: datetime | None = None
    departed_at: datetime | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.landed_at is not None and self.departed_at is not None:
            require(self.departed_at >= self.landed_at, "Diversion departure precedes landing")


@dataclass(frozen=True, kw_only=True)
class FlightOutcome(Record):
    status: OutcomeStatus
    resulting_airport: AirportRef | None
    reached_scheduled_destination: bool | None
    historical_sample: HistoricalSampleRef
    timing_method: str
    actual_departure_at: datetime | None = None
    actual_arrival_at: datetime | None = None
    next_decision_at: datetime | None = None
    cancellation_code: str | None = None
    diversion_stops: tuple[DiversionStop, ...] = ()
    unresolved_reason: str | None = None

    def __post_init__(self):
        super().__post_init__()
        known = self.known_times()
        require(known == sorted(known), "Outcome timestamps are not chronological")
        if self.actual_departure_at is not None and self.actual_arrival_at is not None:
            require(self.actual_arrival_at > self.actual_departure_at, "Actual arrival must follow departure")
        require(len(self.diversion_stops) <= 5, "At most five diversion stops can be recorded")
        if self.next_decision_at is None:
            require(self.unresolved_reason is not None, "Unknown next_decision_at requires an unresolved_reason")
        else:
            require(self.unresolved_reason is None, "Resolved outcomes cannot have an unresolved_reason")
        if self.unresolved_reason == "episode_cutoff":
            require(self.actual_arrival_at is None and self.reached_scheduled_destination is not True
                    and self.status is not OutcomeStatus.ARRIVED,
                    "episode_cutoff cannot hide or claim a completed arrival")
        if self.status is OutcomeStatus.UNRESOLVED:
            require(self.next_decision_at is None and self.reached_scheduled_destination is None
                    and self.actual_arrival_at is None and self.resulting_airport is None,
                    "An unresolved outcome cannot claim a final location, arrival, or next decision")
        elif self.status is not OutcomeStatus.DIVERTED or self.next_decision_at is not None:
            require(self.resulting_airport is not None, "Known outcome requires a resulting airport")
            require(type(self.reached_scheduled_destination) is bool, "Known outcome requires destination-reached evidence")
        if self.status is OutcomeStatus.CANCELLED:
            require(self.reached_scheduled_destination is False and self.actual_arrival_at is None,
                    "Cancellation cannot claim arrival at the scheduled destination")
        else:
            require(self.cancellation_code is None, "Only cancellations can have a cancellation_code")
        if self.status is OutcomeStatus.DIVERTED:
            if self.next_decision_at is not None:
                require(bool(self.diversion_stops), "Resolved diversion requires its ordered stops")
            if self.reached_scheduled_destination is False and self.diversion_stops:
                require(self.resulting_airport == self.diversion_stops[-1].airport,
                        "Stranded diversion must end at its last recorded diversion airport")
        else:
            require(not self.diversion_stops, "Only diversions can have diversion stops")
        if self.status is OutcomeStatus.ARRIVED:
            require(self.reached_scheduled_destination is True, "Normal arrival must reach its scheduled destination")
        if self.status in (OutcomeStatus.ARRIVED, OutcomeStatus.DIVERTED) and self.next_decision_at is not None:
            require(self.actual_departure_at is not None and self.actual_arrival_at is not None,
                    "Resolved flown outcomes need actual departure and arrival times")
            require(self.next_decision_at == self.actual_arrival_at,
                    "next_decision_at must equal actual_arrival_at; apply boarding buffers separately")

    def known_times(self):
        times = [self.actual_departure_at]
        for stop in self.diversion_stops:
            times.extend((stop.landed_at, stop.departed_at))
        times.extend((self.actual_arrival_at, self.next_decision_at))
        return [timestamp for timestamp in times if timestamp is not None]

    def arrival_delay_minutes(self, flight: FlightOption):
        if self.reached_scheduled_destination is not True or self.actual_arrival_at is None:
            return None
        require(self.resulting_airport == flight.destination, "Arrival delay requires the flight's scheduled destination")
        return (self.actual_arrival_at - flight.scheduled_arrival_at).total_seconds() / 60


@dataclass(frozen=True, kw_only=True)
class DecisionStep(Record):
    step_index: int
    current_airport: AirportRef
    decision_at: datetime
    ready_to_board_at: datetime
    offered_flight_ids: tuple[str, ...]
    selected_flight: FlightOption
    outcome: FlightOutcome

    def __post_init__(self):
        super().__post_init__()
        require(self.step_index >= 0, "step_index cannot be negative")
        require(self.ready_to_board_at >= self.decision_at, "Boarding readiness precedes the decision")
        require(len(self.offered_flight_ids) == len(set(self.offered_flight_ids)), "Duplicate offered flight IDs")
        flight, outcome = self.selected_flight, self.outcome
        require(flight.flight_id in self.offered_flight_ids, "Selected flight was not offered")
        require(flight.origin == self.current_airport, "Flight origin differs from current airport")
        require(flight.scheduled_departure_at >= self.ready_to_board_at, "scheduled departure precedes boarding readiness")
        if outcome.actual_departure_at is not None:
            require(outcome.actual_departure_at >= self.ready_to_board_at, "actual departure precedes boarding readiness")
        require(all(timestamp >= self.decision_at for timestamp in outcome.known_times()), "Outcome precedes the decision")
        if outcome.reached_scheduled_destination is True:
            require(outcome.resulting_airport == flight.destination, "Outcome contradicts the scheduled destination")
        elif outcome.reached_scheduled_destination is False:
            require(outcome.resulting_airport != flight.destination, "Outcome contradicts failure to reach scheduled destination")
        if outcome.status is OutcomeStatus.CANCELLED:
            require(outcome.resulting_airport == flight.origin, "Cancelled service must leave the traveler at its origin")


@dataclass(frozen=True, kw_only=True)
class TerminalState(Record):
    current_airport: AirportRef | None
    next_decision_at: datetime | None
    last_known_at: datetime

    def __post_init__(self):
        super().__post_init__()
        if self.next_decision_at is not None:
            require(self.last_known_at <= self.next_decision_at, "Last known time exceeds terminal decision time")


@dataclass(frozen=True, kw_only=True)
class EpisodeRecord(Record):
    schema_version: int
    episode_id: str
    request: TripRequest
    settings: EpisodeSettings
    provenance: EpisodeProvenance
    steps: tuple[DecisionStep, ...]
    terminal: TerminalState
    termination_reason: TerminationReason

    def __post_init__(self):
        super().__post_init__()
        require(self.schema_version == 1, "Unsupported episode schema_version")
        require(len(self.steps) <= self.settings.max_decisions, "Episode exceeds decision limit")
        airport, decision_at = self.request.origin, self.request.start_at
        last_known_at = decision_at
        buffer = self.settings.initial_boarding_buffer_minutes
        attempted_flights = set()
        for index, step in enumerate(self.steps):
            require(step.selected_flight.flight_id not in attempted_flights, "Flight instance was already attempted")
            attempted_flights.add(step.selected_flight.flight_id)
            require(decision_at is not None, "Cannot continue after an unresolved outcome")
            require(airport != self.request.destination, "Cannot continue after reaching the trip destination")
            require(step.step_index == index, "Nonconsecutive step index")
            require(step.current_airport == airport and step.decision_at == decision_at, "Discontinuous decision state")
            require(step.ready_to_board_at == decision_at + timedelta(minutes=buffer), "Incorrect or double-applied boarding buffer")
            require(step.decision_at < self.episode_end_at and step.selected_flight.scheduled_departure_at < self.episode_end_at,
                    "Action selection is at or after the episode cutoff")
            require(step.selected_flight.schedule_ref == self.provenance.schedule_ref, "Flight references a different schedule snapshot")
            require(step.outcome.historical_sample.source.dataset_version == self.provenance.outcome_dataset_version,
                    "Outcome references a different historical dataset version")
            known = step.outcome.known_times()
            require(all(timestamp <= self.episode_end_at for timestamp in known), "Outcome extends beyond the hard episode cutoff")
            last_known_at = max([last_known_at, step.decision_at, *known])
            airport, decision_at = step.outcome.resulting_airport, step.outcome.next_decision_at
            buffer = (self.settings.cancellation_boarding_buffer_minutes
                      if step.outcome.status is OutcomeStatus.CANCELLED else self.settings.connection_buffer_minutes)
        reason = self.termination_reason
        require(self.terminal.current_airport == airport, "Terminal airport contradicts the trace")
        unresolved_reasons = {TerminationReason.UNRESOLVED_DIVERSION_TIMING, TerminationReason.UNRESOLVED_OUTCOME}
        if decision_at is None:
            require(reason in unresolved_reasons or reason is TerminationReason.TIME_LIMIT,
                    "Unresolved outcome requires an explicit unresolved termination reason")
            require(self.terminal.next_decision_at is None, "Unknown terminal timing must remain unknown")
            if reason is TerminationReason.UNRESOLVED_DIVERSION_TIMING:
                require(self.steps[-1].outcome.status is OutcomeStatus.DIVERTED
                        and self.steps[-1].outcome.actual_arrival_at is None,
                        "unresolved_diversion_timing requires a diversion with unknown arrival time")
            require((reason is TerminationReason.TIME_LIMIT)
                    == (self.steps[-1].outcome.unresolved_reason == "episode_cutoff"),
                    "Censored time-limit outcome and episode_cutoff reason must agree")
        else:
            require(reason not in unresolved_reasons, "Resolved trace cannot claim an unresolved termination")
            if reason is TerminationReason.TIME_LIMIT:
                decision_at = last_known_at = self.episode_end_at
            require(self.terminal.next_decision_at == decision_at, "Terminal decision time contradicts the trace")
        require(self.terminal.last_known_at == last_known_at, "Terminal last_known_at contradicts the trace")
        if self.arrival_at_destination is not None:
            require(reason is TerminationReason.DESTINATION_REACHED, "Destination reached but termination reason disagrees")
        else:
            require(reason is not TerminationReason.DESTINATION_REACHED, "Cannot claim destination reached without a resolved arrival")
        if reason is TerminationReason.DECISION_LIMIT:
            require(len(self.steps) == self.settings.max_decisions, "Decision-limit termination before the decision limit")

    @property
    def episode_end_at(self):
        return self.request.start_at + timedelta(minutes=self.settings.max_duration_minutes)

    @property
    def arrival_at_destination(self):
        if not self.steps:
            return None
        outcome = self.steps[-1].outcome
        if outcome.resulting_airport == self.request.destination and outcome.next_decision_at is not None:
            return outcome.actual_arrival_at
        return None

    @property
    def trip_lateness_minutes(self):
        arrival = self.arrival_at_destination
        if arrival is None:
            return None
        return max(0.0, (arrival - self.request.arrival_deadline).total_seconds() / 60)

    @property
    def cancellation_count(self):
        return sum(step.outcome.status is OutcomeStatus.CANCELLED for step in self.steps)

    @property
    def diversion_count(self):
        return sum(step.outcome.status is OutcomeStatus.DIVERTED for step in self.steps)
