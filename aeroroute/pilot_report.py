from .reporting.pilot import audit_flights, build_report, field_rates
from .storage.fields import BOOLEAN_FIELDS, INTEGER_FIELDS, STRING_FIELDS
from .cli.pilot_report import main


if __name__ == "__main__":
    raise SystemExit(main())
