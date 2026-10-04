from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from functools import lru_cache

from ..domain.evidence import TimeFact
from ..domain.scalars import clock_minutes, parse_integer


UTC = timezone.utc


def unknown(issue, method="none", evidence=()):
    return TimeFact(method=method, issues=(issue,), evidence=tuple(evidence))


def resolved(value, method, checks=(), evidence=None):
    return TimeFact(value, "resolved", method, (), tuple(checks), tuple(evidence) if evidence is not None else (value,))


@lru_cache(maxsize=32768)
def local_candidates(day, minute, zone):
    local = datetime.combine(day, time(minute // 60, minute % 60))
    candidates = set()
    for fold in (0, 1):
        utc = local.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if utc.astimezone(zone).replace(tzinfo=None) == local:
            candidates.add(utc)
    return tuple(sorted(candidates))


def add_minutes(value, minutes):
    if value is None or minutes is None:
        return None
    try:
        return value + timedelta(minutes=minutes)
    except (OverflowError, ValueError):
        return None


def clock_agrees(value, minute, zone, tolerance):
    local = value.astimezone(zone)
    delta = abs(local.hour * 60 + local.minute - minute)
    return min(delta, 1440 - delta) <= tolerance


def check_clock(fact, raw_clock, zone, tolerance=0):
    if fact.value is None:
        return fact
    if raw_clock is None or not str(raw_clock).strip():
        return replace(fact, checks=fact.checks + ("reported_clock_missing",))
    minute = clock_minutes(raw_clock)
    if minute is None:
        return replace(fact, checks=fact.checks + ("reported_clock_invalid",), issues=fact.issues + ("invalid_reported_clock",))
    if zone is None:
        return replace(fact, checks=fact.checks + ("reported_clock_timezone_missing",))
    if not clock_agrees(fact.value, minute, zone, tolerance):
        return replace(fact, value=None, status="contradiction", checks=fact.checks + ("reported_clock_disagrees",),
                       issues=fact.issues + ("reported_clock_disagrees",))
    return replace(fact, checks=fact.checks + ("reported_clock_agrees",))


def schedule_times(day, departure_clock, arrival_clock, elapsed, origin_zone, destination_zone, tolerance_minutes=0):
    minute = clock_minutes(departure_clock)
    if day is None:
        departure = unknown("missing_flight_date")
    elif origin_zone is None:
        departure = unknown("origin_timezone_unresolved")
    elif minute is None:
        departure = unknown("missing_departure_clock" if not departure_clock else "invalid_departure_clock")
    elif parse_integer(departure_clock, "clock", []) == 2400:
        departure = unknown("midnight_date_ambiguous")
    else:
        candidates = local_candidates(day, minute, origin_zone)
        selected = candidates
        if len(candidates) > 1 and elapsed is not None and destination_zone and clock_minutes(arrival_clock) is not None:
            selected = tuple(value for value in candidates if add_minutes(value, elapsed) is not None
                             and clock_agrees(add_minutes(value, elapsed), clock_minutes(arrival_clock), destination_zone, tolerance_minutes))
        if len(selected) == 1:
            departure = resolved(selected[0], "scheduled_local" if len(candidates) == 1 else "scheduled_fold_disambiguated")
        else:
            departure = unknown("nonexistent_local_time" if not candidates else "ambiguous_local_time", evidence=candidates)
    arrival = add_minutes(departure.value, elapsed)
    if arrival is None:
        issue = "schedule_departure_unresolved" if departure.value is None else "missing_or_invalid_scheduled_elapsed"
        return departure, unknown(issue, "scheduled_elapsed")
    return departure, check_clock(resolved(arrival, "scheduled_elapsed"), arrival_clock, destination_zone, tolerance_minutes)


def departure_time(scheduled, delay, raw_clock, zone, cancelled=False, tolerance_minutes=0):
    value = add_minutes(scheduled.value, delay)
    if value is None:
        if cancelled and not raw_clock and delay is None:
            return TimeFact(status="not_applicable", method="cancelled_no_departure_reported")
        return unknown("schedule_departure_unresolved" if scheduled.value is None else "departure_delay_missing", "departure_delay")
    return check_clock(resolved(value, "departure_delay"), raw_clock, zone, tolerance_minutes)


def combine_arrivals(elapsed_candidate, delay_candidate, tolerance_minutes=0):
    evidence = tuple(value for value in (elapsed_candidate, delay_candidate) if value is not None)
    if not evidence:
        return unknown("no_supported_arrival_path")
    if elapsed_candidate is not None and delay_candidate is not None:
        if abs((elapsed_candidate - delay_candidate).total_seconds()) > tolerance_minutes * 60:
            return TimeFact(status="contradiction", method="two_paths", issues=("arrival_paths_disagree",),
                            checks=("arrival_paths_disagree",), evidence=evidence)
        return resolved(elapsed_candidate, "both_agree", ("arrival_paths_agree",), evidence)
    return resolved(evidence[0], "elapsed_only" if elapsed_candidate is not None else "arrival_delay_only",
                    ("arrival_delay_path_missing" if delay_candidate is None else "elapsed_path_missing",), evidence)


def arrival_time(departure, scheduled_arrival, elapsed, delay, raw_clock, zone, tolerance_minutes=0):
    fact = combine_arrivals(add_minutes(departure.value, elapsed), add_minutes(scheduled_arrival.value, delay), tolerance_minutes)
    fact = check_clock(fact, raw_clock, zone, tolerance_minutes)
    if fact.value is not None and departure.value is not None and fact.value <= departure.value:
        return replace(fact, value=None, status="contradiction", issues=fact.issues + ("arrival_not_after_departure",))
    return fact


def candidates_in_window(raw_clock, zone, lower, upper):
    if not raw_clock:
        return (), "stop_clock_missing"
    if clock_minutes(raw_clock) is None:
        return (), "invalid_stop_clock"
    if zone is None:
        return (), "stop_timezone_unresolved"
    if lower is None or upper is None:
        return (), "missing_time_anchor"
    if upper < lower:
        return (), "inconsistent_time_anchors"
    first, last = lower.astimezone(zone).date(), upper.astimezone(zone).date()
    if (last - first).days > 366:
        return (), "reconstruction_window_exceeds_resource_limit"
    candidates = []
    for offset in range((last - first).days + 1):
        candidates.extend(value for value in local_candidates(first + timedelta(days=offset), clock_minutes(raw_clock), zone)
                          if lower <= value <= upper)
    return tuple(candidates), None if candidates else "stop_clock_outside_anchors"


def diversion_sequence(stops, lower, upper):
    domains, errors = [], []
    for landing_clock, departure_clock, zone in stops:
        for clock in (landing_clock, departure_clock):
            values, issue = candidates_in_window(clock, zone, lower, upper)
            domains.append(values)
            errors.append(issue)
    previous = lower
    contradiction = False
    for index, values in enumerate(domains):
        if not values:
            continue
        domains[index] = tuple(value for value in values if previous is None or value >= previous)
        if not domains[index]:
            contradiction = True
        else:
            previous = domains[index][0]
    following = upper
    for index in reversed(range(len(domains))):
        values = domains[index]
        if not values:
            continue
        domains[index] = tuple(value for value in values if following is None or value <= following)
        if not domains[index]:
            contradiction = True
        else:
            following = domains[index][-1]
    facts = []
    for values, issue in zip(domains, errors):
        if contradiction and values:
            facts.append(TimeFact(status="contradiction", method="bounded_stop_clock", issues=("stop_chronology_conflict",), evidence=values))
        elif issue:
            facts.append(unknown(issue, "bounded_stop_clock"))
        elif len(values) == 1:
            facts.append(resolved(values[0], "unique_bounded_stop_clock"))
        else:
            facts.append(unknown("stop_chronology_conflict" if not values else "ambiguous_stop_date", "bounded_stop_clock", values))
    return [(facts[index], facts[index + 1]) for index in range(0, len(facts), 2)]
