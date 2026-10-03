import hashlib
import io
from datetime import date
from functools import cache
from importlib import metadata, resources
from zoneinfo import ZoneInfo

import airportsdata

from .data_snapshots import canonical_bytes, digest, load_json


@cache
def zone_bytes(key):
    if not key or any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("Invalid IANA timezone key")
    return resources.files("tzdata.zoneinfo").joinpath(*key.split("/")).read_bytes()


@cache
def pinned_zone(key):
    return ZoneInfo.from_file(io.BytesIO(zone_bytes(key)), key=key)


@cache
def reference_data():
    return airportsdata.load("IATA"), airportsdata.load("LID")


class AirportResolver:
    def __init__(self, overrides=None):
        self.iata, self.lid = reference_data()
        self.source_version = metadata.version("airportsdata")
        self.overrides = load_json(overrides) if overrides else []
        self.requests = set()
        self.entries = {}
        self.observations = {}
        self._cache = {}
        for entry in self.overrides:
            if not all(entry.get(key) for key in ("airport_id", "code", "time_zone", "source", "valid_from", "valid_to")):
                raise ValueError("Airport overrides require identity, zone, cited source, and validity dates")
            if date.fromisoformat(entry["valid_from"]) > date.fromisoformat(entry["valid_to"]):
                raise ValueError("Reversed airport override date range")
            pinned_zone(entry["time_zone"])

    def resolve(self, airport_id, code, flight_date, expected_state=None):
        code = (code or "").strip().upper()
        day = flight_date.isoformat() if flight_date else None
        key = (airport_id, code, day, expected_state)
        self.requests.add(key)
        if key not in self._cache:
            result = self._resolve(airport_id, code, day, expected_state)
            identity = digest(result)
            self._cache[key] = (identity, result)
            self.entries[identity] = result
        identity, result = self._cache[key]
        observed = self.observations.setdefault(identity, {"dates": set(), "occurrences": 0})
        if flight_date:
            observed["dates"].add(flight_date)
        observed["occurrences"] += 1
        return result

    def _resolve(self, airport_id, code, day, expected_state):
        result = {"airport_id": airport_id, "code": code or None, "time_zone": None, "status": "unmapped",
                  "method": "none", "source": f"airportsdata:{self.source_version}",
                  "reference_icao": None, "reference_name": None, "expected_state": expected_state, "issues": []}
        overrides = [entry for entry in self.overrides if entry["airport_id"] == airport_id and entry["code"] == code
                     and day and entry["valid_from"] <= day <= entry["valid_to"]]
        if len(overrides) > 1:
            raise ValueError(f"Conflicting airport overrides: {airport_id}/{code}/{day}")
        if overrides:
            result.update(time_zone=overrides[0]["time_zone"], status="resolved", method="reviewed_override",
                          source=overrides[0]["source"])
            return result
        candidate = self.iata.get(code)
        method = "exact_iata"
        if candidate is None:
            candidate = self.lid.get(code)
            if candidate and candidate["country"] != "US":
                candidate = None
            method = "exact_us_faa_lid"
        if candidate is None:
            result["issues"] = ["airport_not_in_reference"]
            return result
        result.update(reference_icao=candidate["icao"], reference_name=candidate["name"], method=method)
        if expected_state and expected_state.casefold() != candidate["subd"].casefold():
            result.update(status="conflict", issues=["state_name_mismatch"])
            return result
        try:
            pinned_zone(candidate["tz"])
        except (FileNotFoundError, ValueError):
            result["issues"] = ["timezone_not_in_pinned_tzdata"]
            return result
        result.update(time_zone=candidate["tz"], status="resolved")
        return result

    def replay_requests(self, requests):
        for airport_id, code, day, expected_state in requests:
            self.resolve(airport_id, code, date.fromisoformat(day) if day else None, expected_state)

    def reference_digest(self):
        entries = sorted(self.entries.values(), key=canonical_bytes)
        zones = {entry["time_zone"] for entry in entries if entry["time_zone"]}
        return digest({"airports": entries, "tzdata_version": metadata.version("tzdata"),
                       "tzif_sha256": {key: hashlib.sha256(zone_bytes(key)).hexdigest() for key in sorted(zones)}})

    def serialized_requests(self):
        return sorted(self.requests, key=canonical_bytes)

    def airport_rows(self):
        result = []
        for identity, entry in sorted(self.entries.items()):
            observed = self.observations[identity]
            dates = observed["dates"]
            result.append({**entry, "first_observed_date": min(dates) if dates else None,
                           "last_observed_date": max(dates) if dates else None,
                           "occurrences": observed["occurrences"]})
        return result
