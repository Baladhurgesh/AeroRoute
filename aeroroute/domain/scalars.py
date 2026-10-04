from decimal import Decimal, InvalidOperation
from functools import lru_cache


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


@lru_cache(maxsize=8192)
def clock_minutes(raw):
    value = parse_integer(raw, "clock", [])
    if value == 2400:
        return 0
    if value is None or value < 0 or value // 100 > 23 or value % 100 > 59:
        return None
    return value // 100 * 60 + value % 100
