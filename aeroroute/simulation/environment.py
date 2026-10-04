from dataclasses import dataclass
from datetime import datetime, timedelta

from ..domain.records import (
    AirportRef, DecisionStep, EpisodeRecord, EpisodeSettings, FlightOption,
    FlightOutcome, OutcomeStatus, TerminalState, TerminationReason, TripRequest,
)
from .bindings import provenance
from .rubric import reliability_reward
from .transitions import TransitionPolicy, transition


@dataclass(frozen=True)
class Observation:
    request: TripRequest
    current_airport: AirportRef | None
    decision_at: datetime | None
    ready_to_board_at: datetime | None
    flights: tuple[FlightOption, ...]
    done: bool = False
    termination_reason: TerminationReason | None = None


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    outcome: FlightOutcome
    reward: float
    terminated: bool
    truncated: bool


class FlightEnvironment:
    def __init__(self, schedule, provider, settings, policy=TransitionPolicy()):
        if type(settings) is not EpisodeSettings or type(policy) is not TransitionPolicy:
            raise ValueError("Expected validated episode settings and transition policy")
        self.schedule, self.provider, self.settings, self.policy = schedule, provider, settings, policy
        self.provenance = provenance(schedule, provider, policy)
        self.mode = dict(self.provenance.sampler_parameters)["mode"]
        self.steps = ()
        self.observation = None
        self.record = None

    def reset(self, request, *, episode_id="episode"):
        if type(request) is not TripRequest or type(episode_id) is not str or not episode_id.strip():
            raise ValueError("Expected a validated request and nonempty episode ID")
        observation, record = self._state(request, episode_id, ())
        self.request, self.episode_id = request, episode_id
        self.steps, self.observation, self.record = (), observation, record
        return observation

    def _state(self, request, episode_id, steps):
        cutoff = request.start_at + timedelta(minutes=self.settings.max_duration_minutes)
        airport, decision = request.origin, request.start_at
        buffer, reason = self.settings.initial_boarding_buffer_minutes, None
        last_known = request.start_at
        attempted = tuple(step.selected_flight.flight_id for step in steps)
        for step in steps:
            outcome = step.outcome
            last_known = max([last_known, step.decision_at, *outcome.known_times()])
            airport, decision = outcome.resulting_airport, outcome.next_decision_at
            buffer = (self.settings.cancellation_boarding_buffer_minutes if outcome.status is OutcomeStatus.CANCELLED
                      else self.policy.missed_boarding_buffer_minutes if outcome.status is OutcomeStatus.MISSED_BOARDING
                      else self.settings.connection_buffer_minutes)
        if steps and decision is None:
            outcome = steps[-1].outcome
            reason = (TerminationReason.TIME_LIMIT if outcome.unresolved_reason == "episode_cutoff"
                      else TerminationReason.UNRESOLVED_DIVERSION_TIMING if outcome.status is OutcomeStatus.DIVERTED
                      else TerminationReason.UNRESOLVED_OUTCOME)
        elif steps and airport == request.destination and steps[-1].outcome.actual_arrival_at is not None:
            reason = TerminationReason.DESTINATION_REACHED
        elif decision >= cutoff:
            reason = TerminationReason.TIME_LIMIT
        elif len(steps) >= self.settings.max_decisions:
            reason = TerminationReason.DECISION_LIMIT
        ready = decision + timedelta(minutes=buffer) if decision is not None else None
        if reason is None and ready >= cutoff:
            reason = TerminationReason.TIME_LIMIT
        flights = []
        if reason is None:
            cursor = None
            while True:
                page = self.schedule.departures(airport.airport_id, ready, cutoff, attempted, cursor=cursor)
                flights.extend(page.flights)
                if page.complete:
                    break
                if page.next_cursor is None or page.next_cursor == cursor:
                    raise ValueError("Incomplete schedule pagination")
                cursor = page.next_cursor
            if not flights:
                reason = TerminationReason.NO_FEASIBLE_FLIGHTS
        if reason is None:
            return Observation(request, airport, decision, ready, tuple(flights)), None
        if reason is TerminationReason.TIME_LIMIT and decision is not None:
            decision = last_known = cutoff
        terminal = TerminalState(current_airport=airport, next_decision_at=decision, last_known_at=last_known)
        record = EpisodeRecord(schema_version=2, episode_id=episode_id, request=request, settings=self.settings,
                               provenance=self.provenance, steps=steps, terminal=terminal, termination_reason=reason)
        return Observation(request, airport, decision, None, (), True, reason), record

    def step(self, service_instance_id):
        if self.observation is None or self.observation.done:
            raise ValueError("Reset an active episode before taking an action")
        selected = next((flight for flight in self.observation.flights if flight.flight_id == service_instance_id), None)
        if selected is None:
            raise ValueError("Action is not an offered service")
        observation = self.observation
        sample = self.provider.sample(selected, episode_id=self.episode_id, step_index=len(self.steps))
        cutoff = self.request.start_at + timedelta(minutes=self.settings.max_duration_minutes)
        outcome = transition(sample, selected, observation.decision_at, observation.ready_to_board_at, cutoff, self.policy, self.mode)
        step = DecisionStep(step_index=len(self.steps), current_airport=observation.current_airport,
            decision_at=observation.decision_at, ready_to_board_at=observation.ready_to_board_at,
            offered_flight_ids=tuple(flight.flight_id for flight in observation.flights), selected_flight=selected, outcome=outcome)
        steps = self.steps + (step,)
        following, record = self._state(self.request, self.episode_id, steps)
        self.steps, self.observation, self.record = steps, following, record
        truncated = following.termination_reason in (TerminationReason.TIME_LIMIT, TerminationReason.DECISION_LIMIT)
        reward = 0.0 if record is None else reliability_reward(record)
        return StepResult(following, outcome, reward, following.done and not truncated, truncated)
