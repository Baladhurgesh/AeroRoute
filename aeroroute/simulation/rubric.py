from dataclasses import dataclass

from ..domain.records import Record, TerminationReason, require


RUBRIC_WEIGHTS = (
    ("destination_reached", 60),
    ("deadline_met", 25),
    ("disruption_free", 10),
    ("completed_within_limit", 5),
)


@dataclass(frozen=True, kw_only=True)
class RubricComponent(Record):
    name: str
    weight: int
    score: float

    def __post_init__(self):
        super().__post_init__()
        expected = dict(RUBRIC_WEIGHTS).get(self.name)
        require(expected == self.weight, "Unknown rubric criterion or weight")
        require(self.score in (0.0, 1.0), "Rubric component score must be 0 or 1")


def trace_end(record):
    arrival = record.arrival_at_destination
    if arrival is not None:
        return arrival
    if record.termination_reason is TerminationReason.TIME_LIMIT:
        return record.episode_end_at
    return record.terminal.next_decision_at


def elapsed_minutes(record):
    end = trace_end(record)
    if end is None:
        return None
    return (end - record.request.start_at).total_seconds() / 60


def reliability_criteria(record):
    arrival = record.arrival_at_destination
    end = trace_end(record)
    scores = {
        "destination_reached": arrival is not None,
        "deadline_met": arrival is not None and arrival <= record.request.arrival_deadline,
        "disruption_free": not (record.cancellation_count or record.diversion_count or record.missed_boarding_count),
        "completed_within_limit": end is not None and end <= record.episode_end_at,
    }
    return tuple(RubricComponent(name=name, weight=weight, score=float(scores[name])) for name, weight in RUBRIC_WEIGHTS)


def reliability_reward(record):
    return sum(item.weight * item.score for item in reliability_criteria(record)) / 100
