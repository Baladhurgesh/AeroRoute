from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from .records import SourceRecordRef


@dataclass(frozen=True)
class TimeFact:
    value: datetime | None = None
    status: str = "unresolved"
    method: str = "none"
    issues: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    evidence: tuple[datetime, ...] = ()

    def as_dict(self):
        return {"value": self.value, "status": self.status, "method": self.method,
                "issues": self.issues, "checks": self.checks, "evidence": self.evidence}


@dataclass(frozen=True)
class FlightEvidence:
    source: SourceRecordRef
    flight: Mapping[str, object]
    stops: tuple[Mapping[str, object], ...]
