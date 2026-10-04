import pyarrow as pa

from .fields import BOOLEAN_FIELDS, INTEGER_FIELDS, STRING_FIELDS, TIME_FIELDS


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
