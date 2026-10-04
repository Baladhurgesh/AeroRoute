from ..domain.evidence import TimeFact
from ..domain.scalars import parse_integer
from ..reference.timezones import pinned_zone
from ..storage.fields import TIME_FIELDS
from ..storage.identity import digest
from .parsing import parse_cells
from .times import arrival_time, departure_time, diversion_sequence, schedule_times, unknown


def outcome_category(row):
    cancelled, diverted = row["cancelled"], row["diverted"]
    if cancelled is None or diverted is None or (cancelled and diverted):
        return "unknown"
    if cancelled:
        return "cancelled"
    if not diverted:
        return "ordinary"
    reached = row["diversion_reached_destination"]
    return "diverted_reached" if reached is True else "diverted_stranded" if reached is False else "diverted_unknown"


def mapping_zone(mapping):
    return pinned_zone(mapping["time_zone"]) if mapping["time_zone"] else None


def normalize_row(raw, ordinal, source, resolver, tolerance_minutes):
    row = parse_cells(raw)
    row.update(source_row_id=ordinal, source_sha256=source["sha256"], source_file=source["path"],
               source_member=source["csv_member"],
               source_record_id=digest([source["sha256"], source["csv_member"], ordinal]))
    category = outcome_category(row)
    row["outcome_category"] = category
    flags = []
    origin = resolver.resolve(row["origin_airport_id"], row["origin"], row["flight_date"], row["origin_state_name"])
    destination = resolver.resolve(row["destination_airport_id"], row["destination"], row["flight_date"], row["destination_state_name"])
    for role, mapping in (("origin", origin), ("destination", destination)):
        row[f"{role}_time_zone"] = mapping["time_zone"]
        if mapping["status"] != "resolved":
            flags.append(f"{role}_mapping_{mapping['status']}")
    origin_zone, destination_zone = mapping_zone(origin), mapping_zone(destination)
    scheduled_departure, scheduled_arrival = schedule_times(
        row["flight_date"], row["scheduled_departure_clock"], row["scheduled_arrival_clock"],
        row["scheduled_elapsed_minutes"], origin_zone, destination_zone, tolerance_minutes)
    actual_departure = departure_time(scheduled_departure, row["departure_delay_minutes"], row["actual_departure_clock"],
                                     origin_zone, category == "cancelled", tolerance_minutes)
    if category == "cancelled":
        actual_arrival = TimeFact(status="not_applicable", method="cancelled_no_arrival")
    elif category in ("ordinary", "diverted_reached"):
        elapsed = row["actual_elapsed_minutes"] if category == "ordinary" else row["diversion_elapsed_minutes"]
        delay = row["arrival_delay_minutes"] if category == "ordinary" else row["diversion_arrival_delay_minutes"]
        actual_arrival = arrival_time(actual_departure, scheduled_arrival, elapsed, delay,
                                      row["actual_arrival_clock"], destination_zone, tolerance_minutes)
    else:
        actual_arrival = unknown("stranded_gate_arrival_not_reported" if category == "diverted_stranded" else "outcome_unresolved")
    facts = (scheduled_departure, scheduled_arrival, actual_departure, actual_arrival)
    row.update({name: fact.as_dict() for name, fact in zip(TIME_FIELDS, facts)})
    stops, clocks = [], []
    for index in range(1, 6):
        prefix = f"Div{index}"
        fields = ("Airport", "AirportID", "AirportSeqID", "WheelsOn", "WheelsOff", "TotalGTime", "LongestGTime")
        values = {field: (raw.get(prefix + field) or "").strip() or None for field in fields}
        if not any(value is not None for value in values.values()):
            continue
        airport_id = parse_integer(values["AirportID"], prefix + "AirportID", row["cell_issues"])
        mapping = resolver.resolve(airport_id, values["Airport"], row["flight_date"])
        if mapping["status"] != "resolved":
            flags.append("diversion_mapping_unresolved")
        stops.append({
            "source_record_id": row["source_record_id"], "source_row_id": ordinal, "stop_index": index,
            "airport_id": airport_id, "airport": values["Airport"],
            "airport_seq_id": parse_integer(values["AirportSeqID"], prefix + "AirportSeqID", row["cell_issues"]),
            "landing_clock": values["WheelsOn"], "departure_clock": values["WheelsOff"],
            "total_ground_minutes": parse_integer(values["TotalGTime"], prefix + "TotalGTime", row["cell_issues"]),
            "longest_ground_minutes": parse_integer(values["LongestGTime"], prefix + "LongestGTime", row["cell_issues"]),
            "time_zone": mapping["time_zone"], "mapping_status": mapping["status"],
        })
        clocks.append((values["WheelsOn"], values["WheelsOff"], mapping_zone(mapping)))
    for stop, (landing, departure) in zip(stops, diversion_sequence(clocks, actual_departure.value, actual_arrival.value)):
        stop.update(landing=landing.as_dict(), departure=departure.as_dict())
    if stops and not row["diverted"]:
        flags.append("diversion_stops_without_diversion_flag")
    count = row["diversion_landings"]
    complete_stops = (count is not None and 1 <= count <= 5 and [stop["stop_index"] for stop in stops] == list(range(1, count + 1))
                      and all(stop["airport_id"] is not None and stop["airport"] for stop in stops))
    if row["diverted"] and not complete_stops:
        flags.append("incomplete_diversion_stop_sequence")
    if category in ("ordinary", "diverted_reached"):
        endpoint_id, endpoint = row["destination_airport_id"], row["destination"]
        endpoint_method = "reported_scheduled_destination"
    elif category == "cancelled":
        endpoint_id, endpoint = row["origin_airport_id"], row["origin"]
        endpoint_method = "cancelled_at_origin"
    elif category == "diverted_stranded" and complete_stops:
        endpoint_id, endpoint = stops[-1]["airport_id"], stops[-1]["airport"]
        endpoint_method = "reported_final_diversion_stop"
    else:
        endpoint_id, endpoint, endpoint_method = None, None, "unresolved"
    row.update(endpoint_airport_id=endpoint_id, endpoint=endpoint,
               endpoint_status=endpoint_method if endpoint_id is not None and endpoint else "unresolved",
               endpoint_issues=[] if endpoint_id is not None and endpoint else ["endpoint_not_supported"],
               quality_flags=sorted(set(flags)))
    return row, stops
