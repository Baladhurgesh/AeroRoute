from datetime import date

from ..domain.scalars import parse_integer
from ..storage.fields import BOOLEAN_FIELDS, INTEGER_FIELDS, STRING_FIELDS


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
