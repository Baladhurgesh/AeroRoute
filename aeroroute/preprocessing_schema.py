from datetime import date
from decimal import Decimal, InvalidOperation

import pyarrow as pa


INTEGER_FIELDS = {
    "origin_airport_id": "OriginAirportID", "origin_airport_seq_id": "OriginAirportSeqID",
    "destination_airport_id": "DestAirportID", "destination_airport_seq_id": "DestAirportSeqID",
    "scheduled_elapsed_minutes": "CRSElapsedTime", "departure_delay_minutes": "DepDelay",
    "arrival_delay_minutes": "ArrDelay", "actual_elapsed_minutes": "ActualElapsedTime",
    "diversion_elapsed_minutes": "DivActualElapsedTime", "diversion_arrival_delay_minutes": "DivArrDelay",
    "diversion_landings": "DivAirportLandings", "taxi_in_minutes": "TaxiIn", "taxi_out_minutes": "TaxiOut",
    "air_time_minutes": "AirTime", "marketing_carrier_dot_id": "DOT_ID_Marketing_Airline",
    "operating_carrier_dot_id": "DOT_ID_Operating_Airline",
}
STRING_FIELDS = {
    "origin": "Origin", "destination": "Dest", "origin_state_name": "OriginStateName",
    "destination_state_name": "DestStateName", "marketing_carrier": "Marketing_Airline_Network",
    "operating_carrier": "Operating_Airline", "marketing_flight_number": "Flight_Number_Marketing_Airline",
    "operating_flight_number": "Flight_Number_Operating_Airline", "duplicate_flag": "Duplicate",
    "originally_scheduled_carrier": "Originally_Scheduled_Code_Share_Airline",
    "originally_scheduled_flight_number": "Flight_Num_Originally_Scheduled_Code_Share_Airline",
    "codeshare_partner": "Operated_or_Branded_Code_Share_Partners", "cancellation_code": "CancellationCode",
    "scheduled_departure_clock": "CRSDepTime", "scheduled_arrival_clock": "CRSArrTime",
    "actual_departure_clock": "DepTime", "actual_arrival_clock": "ArrTime",
}
BOOLEAN_FIELDS = {"cancelled": "Cancelled", "diverted": "Diverted", "diversion_reached_destination": "DivReachedDest"}
TIME_FIELDS = ("scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival")
ISSUE = pa.struct([("field", pa.string()), ("code", pa.string()), ("raw_value", pa.string())])
FACT = pa.struct([
    ("value", pa.timestamp("us", tz="UTC")), ("status", pa.string()), ("method", pa.string()),
    ("issues", pa.list_(pa.string())), ("checks", pa.list_(pa.string())),
    ("evidence", pa.list_(pa.timestamp("us", tz="UTC"))),
])
FLIGHTS_SCHEMA = pa.schema([
    ("source_record_id", pa.string()), ("source_sha256", pa.string()), ("source_file", pa.string()),
    ("source_member", pa.string()), ("source_row_id", pa.int64()), ("flight_date", pa.date32()),
    *[(name, pa.int64()) for name in INTEGER_FIELDS], *[(name, pa.string()) for name in STRING_FIELDS],
    *[(name, pa.bool_()) for name in BOOLEAN_FIELDS], *[(name, FACT) for name in TIME_FIELDS],
    ("outcome_category", pa.string()), ("endpoint_airport_id", pa.int64()), ("endpoint", pa.string()),
    ("endpoint_status", pa.string()), ("endpoint_issues", pa.list_(pa.string())),
    ("origin_time_zone", pa.string()), ("destination_time_zone", pa.string()),
    ("cell_issues", pa.list_(ISSUE)), ("quality_flags", pa.list_(pa.string())),
])
STOPS_SCHEMA = pa.schema([
    ("source_record_id", pa.string()), ("source_row_id", pa.int64()), ("stop_index", pa.int8()),
    ("airport_id", pa.int64()), ("airport_seq_id", pa.int64()), ("airport", pa.string()),
    ("landing_clock", pa.string()), ("departure_clock", pa.string()),
    ("total_ground_minutes", pa.int64()), ("longest_ground_minutes", pa.int64()),
    ("time_zone", pa.string()), ("mapping_status", pa.string()), ("landing", FACT), ("departure", FACT),
])
AIRPORTS_SCHEMA = pa.schema([
    ("airport_id", pa.int64()), ("code", pa.string()), ("time_zone", pa.string()), ("status", pa.string()),
    ("method", pa.string()), ("source", pa.string()), ("reference_icao", pa.string()),
    ("reference_name", pa.string()), ("expected_state", pa.string()), ("issues", pa.list_(pa.string())),
    ("first_observed_date", pa.date32()), ("last_observed_date", pa.date32()), ("occurrences", pa.int64()),
])


def parse_integer(raw, field, issues):
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip()
    try:
        value = Decimal(text)
        if not value.is_finite() or value != value.to_integral_value() or not -(2 ** 63) <= value < 2 ** 63:
            raise ValueError
        return int(value)
    except (InvalidOperation, ValueError, OverflowError):
        issues.append({"field": field, "code": "invalid_integer", "raw_value": str(raw)})
        return None


def parse_boolean(raw, field, issues):
    value = parse_integer(raw, field, issues)
    if value is None:
        return None
    if value not in (0, 1):
        issues.append({"field": field, "code": "invalid_boolean", "raw_value": str(raw)})
        return None
    return bool(value)


def parse_date(raw, issues):
    try:
        return date.fromisoformat(raw)
    except (ValueError, TypeError):
        issues.append({"field": "FlightDate", "code": "invalid_date", "raw_value": raw})
        return None


def parse_cells(raw):
    issues = []
    result = {name: parse_integer(raw.get(source), source, issues) for name, source in INTEGER_FIELDS.items()}
    result.update({name: (raw.get(source) or "").strip() or None for name, source in STRING_FIELDS.items()})
    result.update({name: parse_boolean(raw.get(source), source, issues) for name, source in BOOLEAN_FIELDS.items()})
    result["flight_date"] = parse_date(raw.get("FlightDate"), issues)
    for name in ("scheduled_elapsed_minutes", "actual_elapsed_minutes", "diversion_elapsed_minutes"):
        if result[name] is not None and result[name] <= 0:
            issues.append({"field": INTEGER_FIELDS[name], "code": "nonpositive_duration", "raw_value": raw.get(INTEGER_FIELDS[name])})
            result[name] = None
    for name in ("origin_airport_id", "destination_airport_id"):
        if result[name] is not None and result[name] <= 0:
            issues.append({"field": INTEGER_FIELDS[name], "code": "invalid_airport_id", "raw_value": raw.get(INTEGER_FIELDS[name])})
            result[name] = None
    result["cell_issues"] = issues
    return result
