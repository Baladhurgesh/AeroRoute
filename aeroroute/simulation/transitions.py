from dataclasses import dataclass
from datetime import timedelta

from ..domain.records import AirportRef, DiversionStop, FlightOutcome, OutcomeStatus


@dataclass(frozen=True)
class TransitionPolicy:
    cancellation_recovery_minutes: int = 60
    missed_boarding_buffer_minutes: int = 0
    version: str = "whole-timeline-recovery-v1"

    def __post_init__(self):
        if (self.version != "whole-timeline-recovery-v1"
                or any(type(value) is not int or value < 0 for value in
                       (self.cancellation_recovery_minutes, self.missed_boarding_buffer_minutes))):
            raise ValueError("Unsupported transition policy or negative recovery/buffer")

    def parameters(self):
        return (("cancellation_recovery_minutes", str(self.cancellation_recovery_minutes)),
                ("missed_boarding_buffer_minutes", str(self.missed_boarding_buffer_minutes)),
                ("policy_version", self.version))


def transition(sample, flight, decision_at, ready_at, cutoff, policy, mode):
    if mode not in ("empirical", "replay"):
        raise ValueError("Unknown evidence mode")
    row = sample.evidence.flight
    if (row["origin_airport_id"], row["destination_airport_id"]) != (flight.origin.airport_id, flight.destination.airport_id):
        raise ValueError("Historical evidence has a different route")
    base = {"historical_sample": sample.historical_sample, "timing_method": f"{mode}:{policy.version}"}
    def unresolved(reason, departure=None):
        return FlightOutcome(status=OutcomeStatus.UNRESOLVED, resulting_airport=None,
                             reached_scheduled_destination=None, actual_departure_at=departure,
                             unresolved_reason=reason, **base)
    scheduled = row["scheduled_departure"]
    if scheduled.status != "resolved" or scheduled.value is None:
        return unresolved("missing_transfer_anchor")
    offset = flight.scheduled_departure_at - scheduled.value if mode == "empirical" else timedelta(0)
    if mode == "replay" and scheduled.value != flight.scheduled_departure_at:
        raise ValueError("Replay scheduled departure differs from selected service")
    def event(fact):
        return fact.value + offset if fact.status == "resolved" and fact.value is not None else None
    departure, arrival = event(row["actual_departure"]), event(row["actual_arrival"])
    observed_departure = departure if departure is not None and ready_at <= departure <= cutoff else None
    category = row["outcome_category"]
    if category == "cancelled":
        if departure is not None:
            return unresolved("cancelled_with_departure_evidence", observed_departure)
        recovery = flight.scheduled_departure_at + timedelta(minutes=policy.cancellation_recovery_minutes)
        if recovery > cutoff:
            return unresolved("episode_cutoff")
        return FlightOutcome(status=OutcomeStatus.CANCELLED, resulting_airport=flight.origin,
                             reached_scheduled_destination=False, cancellation_code=row["cancellation_code"],
                             next_decision_at=recovery, **base)
    if category not in ("ordinary", "diverted_reached", "diverted_stranded", "diverted_unknown"):
        return unresolved("unclassified_historical_outcome", observed_departure)
    if departure is not None and departure < ready_at:
        return FlightOutcome(status=OutcomeStatus.MISSED_BOARDING, resulting_airport=flight.origin,
                             reached_scheduled_destination=False, next_decision_at=flight.scheduled_departure_at, **base)
    stops, invalid_stops = [], False
    for value in sample.evidence.stops:
        try:
            airport = AirportRef(airport_id=value["airport_id"], code=value["airport"])
        except (TypeError, ValueError):
            invalid_stops = True
            continue
        stops.append(DiversionStop(airport=airport, landed_at=event(value["landing"]), departed_at=event(value["departure"])))
    times = [departure, *[time for stop in stops for time in (stop.landed_at, stop.departed_at)], arrival]
    known = [time for time in times if time is not None]
    if known != sorted(known) or any(time < decision_at for time in known) or (departure is not None and arrival is not None and arrival <= departure):
        return unresolved("contradictory_historical_timeline", observed_departure)
    if any(time > cutoff for time in known):
        observed_stops = tuple(DiversionStop(airport=stop.airport,
                                           landed_at=stop.landed_at if stop.landed_at is not None and stop.landed_at <= cutoff else None,
                                           departed_at=stop.departed_at if stop.departed_at is not None and stop.departed_at <= cutoff else None)
                               for stop in stops if any(time is not None and time <= cutoff for time in (stop.landed_at, stop.departed_at)))
        if category.startswith("diverted") and observed_stops:
            return FlightOutcome(status=OutcomeStatus.DIVERTED, resulting_airport=None, reached_scheduled_destination=None,
                                 actual_departure_at=observed_departure, diversion_stops=observed_stops,
                                 unresolved_reason="episode_cutoff", **base)
        return unresolved("episode_cutoff", observed_departure)
    if category == "ordinary":
        if departure is None or arrival is None:
            return unresolved("missing_flight_timing", observed_departure)
        return FlightOutcome(status=OutcomeStatus.ARRIVED, resulting_airport=flight.destination,
                             reached_scheduled_destination=True, actual_departure_at=departure,
                             actual_arrival_at=arrival, next_decision_at=arrival, **base)
    complete = (not invalid_stops and row["diversion_landings"] == len(stops) and bool(stops)
                and [value["stop_index"] for value in sample.evidence.stops] == list(range(1, len(stops) + 1)))
    if category == "diverted_reached" and complete and departure is not None and arrival is not None:
        return FlightOutcome(status=OutcomeStatus.DIVERTED, resulting_airport=flight.destination,
                             reached_scheduled_destination=True, actual_departure_at=departure,
                             actual_arrival_at=arrival, next_decision_at=arrival, diversion_stops=tuple(stops), **base)
    endpoint = stops[-1].airport if category == "diverted_stranded" and complete else None
    return FlightOutcome(status=OutcomeStatus.DIVERTED, resulting_airport=endpoint,
                         reached_scheduled_destination=False if endpoint is not None else None,
                         actual_departure_at=observed_departure, diversion_stops=tuple(stops),
                         unresolved_reason="missing_diversion_gate_timing" if complete else "incomplete_diversion_evidence", **base)
