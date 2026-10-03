import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq

from .data_access import publish, readonly_database, staging
from .data_snapshots import canonical_bytes, digest, load_json, verify_derived, write_json
from .records import AirportRef
from .flight_times import clock_minutes


SCHEDULE_FIELDS = (
    "flight_date", "origin_airport_id", "origin", "destination_airport_id", "destination",
    "operating_carrier", "operating_flight_number", "operating_carrier_dot_id",
    "marketing_carrier", "marketing_flight_number", "marketing_carrier_dot_id",
    "originally_scheduled_carrier", "originally_scheduled_flight_number", "codeshare_partner",
    "scheduled_departure_clock", "scheduled_arrival_clock", "scheduled_elapsed_minutes",
    "scheduled_departure", "scheduled_arrival", "duplicate_flag",
)
OUTCOME_FIELDS = (
    "cancelled", "diverted", "diversion_reached_destination", "cancellation_code", "outcome_category",
    "actual_departure", "actual_arrival", "actual_departure_clock", "actual_arrival_clock",
    "departure_delay_minutes", "arrival_delay_minutes", "actual_elapsed_minutes", "diversion_elapsed_minutes",
    "diversion_arrival_delay_minutes", "diversion_landings", "endpoint_airport_id", "endpoint",
    "endpoint_status", "endpoint_issues", "taxi_in_minutes", "taxi_out_minutes", "air_time_minutes",
)


def json_value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ResolutionPolicy:
    version: str = "exact-operating-service-v1"
    review_reference: str | None = None
    max_group_size: int = 1000

    def __post_init__(self):
        if self.version != "exact-operating-service-v1" or type(self.max_group_size) is not int or self.max_group_size < 2:
            raise ValueError("Unsupported resolution policy")
        if self.review_reference is not None and not self.review_reference.strip():
            raise ValueError("Review reference cannot be empty")


def scheduled_attributes(row):
    try:
        AirportRef(airport_id=row["origin_airport_id"], code=row["origin"])
        AirportRef(airport_id=row["destination_airport_id"], code=row["destination"])
    except (TypeError, ValueError):
        return None, "invalid_airport_identity"
    for name in ("operating_carrier", "operating_flight_number", "marketing_carrier", "marketing_flight_number", "flight_date"):
        if not row[name]:
            return None, "missing_service_identity"
    departure, arrival = row["scheduled_departure"], row["scheduled_arrival"]
    if departure["status"] != "resolved" or arrival["status"] != "resolved" or not departure["value"] or not arrival["value"]:
        return None, "unresolved_schedule_time"
    if arrival["value"] <= departure["value"] or row["origin_airport_id"] == row["destination_airport_id"]:
        return None, "invalid_schedule"
    return json_value({**{key: row[key] for key in SCHEDULE_FIELDS},
                       "scheduled_departure": departure["value"], "scheduled_arrival": arrival["value"]}), None


def evidence_digest(store, record_id):
    row, stops = store.raw(record_id)
    value = {key: row[key] for key in OUTCOME_FIELDS}
    value["stops"] = [{key: item for key, item in stop.items() if key not in ("source_record_id", "source_row_id")} for stop in stops]
    return digest(json_value(value))


def build_services(store, output, policy=ResolutionPolicy()):
    identity = {"kind": "services", "schema_version": 1, "builder_version": 1,
                "snapshot": store.snapshot["manifest"]["dataset_version"], "lookup": store.lookup_path.name,
                "policy": asdict(policy)}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    report = {"source_rows": 0, "support_rows": 0, "services": 0, "collision_groups": 0,
              "schedule_conflicts": 0, "evidence_conflicts": 0, "alias_conflicts": 0,
              "exclusions": Counter(), "duplicate_flags": Counter(), "review_reference": policy.review_reference}
    with sqlite3.connect(stage / "services.sqlite") as database:
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA cache_size=-8192")
        database.executescript("""
            CREATE TABLE candidates (record_id TEXT PRIMARY KEY, service_id TEXT, split TEXT, attributes TEXT);
            CREATE INDEX candidates_service ON candidates(service_id, record_id);
            CREATE TABLE aliases (alias TEXT, service_id TEXT, PRIMARY KEY(alias, service_id));
            CREATE TABLE services (service_id TEXT PRIMARY KEY, split TEXT, year INTEGER, month INTEGER,
                origin INTEGER, destination INTEGER, carrier TEXT, quarter INTEGER, weekday INTEGER,
                minute INTEGER, attributes TEXT, primary_source TEXT, evidence_source TEXT,
                schedule_issue TEXT, evidence_issue TEXT);
            CREATE INDEX services_route ON services(split, origin, destination, year);
            CREATE TABLE diagnostics (kind TEXT, split TEXT, record_id TEXT, service_id TEXT, detail TEXT);
            CREATE INDEX diagnostics_split ON diagnostics(split, kind);
        """)
        for partition in store.snapshot["partitions"]:
            with pq.ParquetFile(partition["path"] / "flights.parquet") as parquet:
                for batch in parquet.iter_batches(batch_size=4096, columns=["source_record_id", *SCHEDULE_FIELDS]):
                    for row in batch.to_pylist():
                        report["source_rows"] += 1
                        report["duplicate_flags"][row["duplicate_flag"] or "missing"] += 1
                        attributes, issue = scheduled_attributes(row)
                        split, record_id = partition["group"], row["source_record_id"]
                        if issue:
                            report["exclusions"][issue] += 1
                            database.execute("INSERT INTO diagnostics VALUES (?, ?, ?, NULL, ?)", ("exclusion", split, record_id, issue))
                            continue
                        key = [attributes[field] for field in ("flight_date", "operating_carrier", "operating_flight_number",
                               "origin_airport_id", "destination_airport_id", "scheduled_departure")]
                        service_id = digest([policy.version, key])
                        database.execute("INSERT INTO candidates VALUES (?, ?, ?, ?)",
                                         (record_id, service_id, split, canonical_bytes(attributes).decode()))
                        for carrier, number in (("operating_carrier", "operating_flight_number"),
                                                ("marketing_carrier", "marketing_flight_number"),
                                                ("originally_scheduled_carrier", "originally_scheduled_flight_number")):
                            if attributes[carrier] and attributes[number]:
                                alias = digest([attributes["flight_date"], attributes["origin_airport_id"],
                                                attributes["scheduled_departure"], attributes[carrier], attributes[number]])
                                database.execute("INSERT OR IGNORE INTO aliases VALUES (?, ?)", (alias, service_id))
            database.commit()
        for group in database.execute("SELECT service_id, count(*) AS n FROM candidates GROUP BY service_id"):
            service_id, count = group["service_id"], group["n"]
            report["collision_groups"] += count > 1
            if count > policy.max_group_size:
                database.execute("INSERT INTO diagnostics SELECT 'schedule_conflict', split, record_id, service_id, 'oversized_group' FROM candidates WHERE service_id=?", (service_id,))
                report["schedule_conflicts"] += 1
                continue
            members = database.execute("SELECT * FROM candidates WHERE service_id=? ORDER BY record_id", (service_id,)).fetchall()
            values = [json.loads(member["attributes"]) for member in members]
            canonical_index = min(range(count), key=lambda i: (values[i]["marketing_carrier"], values[i]["marketing_flight_number"], members[i]["record_id"]))
            primary, attributes = members[canonical_index]["record_id"], values[canonical_index]
            core_fields = ("scheduled_arrival", "origin", "destination", "operating_carrier_dot_id",
                           "scheduled_departure_clock", "scheduled_arrival_clock", "scheduled_elapsed_minutes")
            schedule_issue = "conflicting_schedule_fields" if len({digest([value[key] for key in core_fields]) for value in values}) > 1 else None
            evidence_issue = None
            if count > 1 and len({evidence_digest(store, member["record_id"]) for member in members}) > 1:
                evidence_issue = "conflicting_observations"
            day = date.fromisoformat(attributes["flight_date"])
            clock = attributes["scheduled_departure_clock"]
            minute = clock_minutes(clock)
            database.execute("INSERT INTO services VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (service_id, members[0]["split"], day.year, day.month, attributes["origin_airport_id"],
                 attributes["destination_airport_id"], attributes["operating_carrier"], (day.month - 1) // 3 + 1,
                 day.weekday(), minute, canonical_bytes(attributes).decode(), primary,
                 None if evidence_issue else primary, schedule_issue, evidence_issue))
            report["schedule_conflicts"] += schedule_issue is not None
            report["evidence_conflicts"] += evidence_issue is not None
            report["support_rows"] += count
            report["services"] += 1
        for conflict in database.execute("SELECT alias FROM aliases GROUP BY alias HAVING count(*)>1"):
            report["alias_conflicts"] += 1
            database.execute("UPDATE services SET schedule_issue='ambiguous_service_alias' WHERE service_id IN (SELECT service_id FROM aliases WHERE alias=?)", (conflict["alias"],))
        database.commit()
    database.close()
    write_json(stage / "quality.json", report)
    return publish(stage, final, identity)


class ServiceCatalog:
    def __init__(self, store, path):
        self.store, self.path = store, Path(path)
        self.manifest = verify_derived(path, "services")
        if (self.manifest["identity"]["snapshot"] != store.snapshot["manifest"]["dataset_version"]
                or self.manifest["identity"]["lookup"] != store.lookup_path.name):
            raise ValueError("Service catalog references a different snapshot/lookup")
        self.database = readonly_database(self.path / "services.sqlite")
        self.report = load_json(self.path / "quality.json")

    def ready(self, split, evidence=True):
        self.store.inventory(split)
        if split == "all":
            return all(self.ready(group, evidence) for group in ("historical", "evaluation", "unassigned")
                       if self.store.snapshot["inventories"][group]["sources"])
        if not self.manifest["identity"]["policy"]["review_reference"]:
            return False
        condition = "schedule_issue IS NOT NULL" + (" OR evidence_issue IS NOT NULL" if evidence else "")
        if self.database.execute(f"SELECT 1 FROM services WHERE split=? AND ({condition}) LIMIT 1", (split,)).fetchone():
            return False
        return self.database.execute("SELECT 1 FROM diagnostics WHERE split=? AND (kind='schedule_conflict' OR detail='missing_service_identity') LIMIT 1", (split,)).fetchone() is None

    def require_ready(self, split, evidence=True):
        if not self.ready(split, evidence):
            raise ValueError(f"Catalog is not {'provider' if evidence else 'schedule'} ready for {split}; review diagnostics")

    def services(self, split=None, schedule_order=False):
        order = "origin, json_extract(attributes, '$.scheduled_departure'), service_id" if schedule_order else "service_id"
        query = "SELECT * FROM services" + (" WHERE split=?" if split else "") + " ORDER BY " + order
        for row in self.database.execute(query, (split,) if split else ()):
            yield {**dict(row), "service_instance_id": row["service_id"], "attributes": json.loads(row["attributes"])}

    def get(self, service_id):
        row = self.database.execute("SELECT * FROM services WHERE service_id=?", (service_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown service instance")
        return {**dict(row), "service_instance_id": service_id, "attributes": json.loads(row["attributes"])}

    def supports(self, service_id):
        self.get(service_id)
        return tuple(row[0] for row in self.database.execute("SELECT record_id FROM candidates WHERE service_id=? ORDER BY record_id", (service_id,)))

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
