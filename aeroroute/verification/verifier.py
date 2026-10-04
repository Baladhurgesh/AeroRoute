from dataclasses import dataclass
from datetime import datetime, timedelta

from ..domain.records import EpisodeRecord, EpisodeSettings, OutcomeStatus, Record, TerminalState, TerminationReason, TripRequest
from ..simulation.bindings import provenance
from ..simulation.rubric import RUBRIC_WEIGHTS, RubricComponent, elapsed_minutes, reliability_criteria, reliability_reward
from ..simulation.transitions import TransitionPolicy, transition


class VerificationError(ValueError):
    pass


@dataclass(frozen=True, kw_only=True)
class VerificationReport(Record):
    verifier_version: str
    valid: bool
    destination_reached: bool
    deadline_met: bool
    arrival_at_destination: datetime | None
    lateness_minutes: float | None
    elapsed_minutes: float | None
    last_known_elapsed_minutes: float
    decisions: int
    cancellations: int
    diversions: int
    missed_boardings: int
    criteria: tuple[RubricComponent, ...]
    terminal_reward: float

    def __post_init__(self):
        super().__post_init__()
        if tuple((item.name, item.weight) for item in self.criteria) != RUBRIC_WEIGHTS:
            raise ValueError("Rubric criteria must use the named 60/25/10/5 weights in order")
        earned = sum(item.weight * item.score for item in self.criteria) / 100
        if self.terminal_reward != earned:
            raise ValueError("terminal_reward must equal the weighted rubric")


class EpisodeVerifier:
    def __init__(self, schedule, provider, settings, policy=TransitionPolicy()):
        if type(settings) is not EpisodeSettings or type(policy) is not TransitionPolicy:
            raise ValueError("Expected validated settings and transition policy")
        self.schedule, self.provider, self.settings, self.policy = schedule, provider, settings, policy
        self.expected_provenance = provenance(schedule, provider, policy)
        self.mode = dict(self.expected_provenance.sampler_parameters)["mode"]

    def _offers(self, airport, ready, cutoff, attempted):
        offered, cursor = [], None
        while True:
            page = self.schedule.departures(airport.airport_id, ready, cutoff, attempted, cursor=cursor)
            offered.extend(flight.flight_id for flight in page.flights)
            if page.complete:
                return tuple(offered)
            if page.next_cursor is None or page.next_cursor == cursor:
                raise VerificationError("Incomplete schedule search cannot verify offered actions")
            cursor = page.next_cursor

    def verify(self, record, request):
        try:
            if type(record) is str:
                record = EpisodeRecord.from_json(record)
            elif type(record) is dict:
                record = EpisodeRecord.from_dict(record)
            else:
                record = EpisodeRecord.from_dict(record.to_dict())
            return self._verify(record, request)
        except VerificationError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise VerificationError(str(error)) from error

    def _verify(self, record, request):
        if type(request) is not TripRequest or record.request != request or record.settings != self.settings:
            raise VerificationError("Trace differs from trusted request/settings")
        if record.schema_version != 2 or record.provenance != self.expected_provenance:
            raise VerificationError("Unsupported or mismatched simulator/provider/transition provenance")
        cutoff = request.start_at + timedelta(minutes=self.settings.max_duration_minutes)
        airport, decision = request.origin, request.start_at
        buffer = self.settings.initial_boarding_buffer_minutes
        last_known, attempted = request.start_at, []
        for index, step in enumerate(record.steps):
            if decision is None or decision >= cutoff:
                raise VerificationError("Action after unresolved outcome or cutoff")
            ready = decision + timedelta(minutes=buffer)
            if ready >= cutoff:
                raise VerificationError("Action cannot meet readiness before cutoff")
            offers = self._offers(airport, ready, cutoff, tuple(attempted))
            if step.offered_flight_ids != offers:
                raise VerificationError("Offered services differ from complete schedule search")
            selected = self.schedule.get_flight(step.selected_flight.flight_id)
            if step.selected_flight != selected:
                raise VerificationError("Selected flight differs from immutable schedule")
            evidence = self.provider.sample(selected, episode_id=record.episode_id, step_index=index)
            expected = transition(evidence, selected, decision, ready, cutoff, self.policy, self.mode)
            if step.outcome != expected:
                raise VerificationError("Outcome differs from reproduced evidence/transition")
            if (step.current_airport, step.decision_at, step.ready_to_board_at) != (airport, decision, ready):
                raise VerificationError("Incorrect decision state/readiness")
            attempted.append(selected.flight_id)
            last_known = max([last_known, decision, *expected.known_times()])
            airport, decision = expected.resulting_airport, expected.next_decision_at
            buffer = (self.settings.cancellation_boarding_buffer_minutes if expected.status is OutcomeStatus.CANCELLED
                      else self.policy.missed_boarding_buffer_minutes if expected.status is OutcomeStatus.MISSED_BOARDING
                      else self.settings.connection_buffer_minutes)
        arrival = record.arrival_at_destination
        if decision is None:
            outcome = record.steps[-1].outcome
            reason = (TerminationReason.TIME_LIMIT if outcome.unresolved_reason == "episode_cutoff"
                      else TerminationReason.UNRESOLVED_DIVERSION_TIMING if outcome.status is OutcomeStatus.DIVERTED
                      else TerminationReason.UNRESOLVED_OUTCOME)
        elif arrival is not None:
            reason = TerminationReason.DESTINATION_REACHED
        elif decision >= cutoff:
            reason = TerminationReason.TIME_LIMIT
        elif len(record.steps) == self.settings.max_decisions:
            reason = TerminationReason.DECISION_LIMIT
        elif decision + timedelta(minutes=buffer) >= cutoff:
            reason = TerminationReason.TIME_LIMIT
        elif not self._offers(airport, decision + timedelta(minutes=buffer), cutoff, tuple(attempted)):
            reason = TerminationReason.NO_FEASIBLE_FLIGHTS
        else:
            raise VerificationError("Premature termination while feasible actions remain")
        if reason is TerminationReason.TIME_LIMIT and decision is not None:
            last_known = decision = cutoff
        if record.termination_reason != reason or record.terminal != TerminalState(current_airport=airport, next_decision_at=decision, last_known_at=last_known):
            raise VerificationError("Terminal state or reason differs from verified trace")
        criteria = reliability_criteria(record)
        return VerificationReport(verifier_version="reliability-rubric-v1", valid=True,
            destination_reached=arrival is not None, deadline_met=arrival is not None and arrival <= request.arrival_deadline,
            arrival_at_destination=arrival, lateness_minutes=record.trip_lateness_minutes,
            elapsed_minutes=elapsed_minutes(record),
            last_known_elapsed_minutes=(last_known - request.start_at).total_seconds() / 60,
            decisions=len(record.steps), cancellations=record.cancellation_count, diversions=record.diversion_count,
            missed_boardings=record.missed_boarding_count, criteria=criteria,
            terminal_reward=reliability_reward(record))
