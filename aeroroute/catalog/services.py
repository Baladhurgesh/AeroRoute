import json
import os
import shutil
import sqlite3
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..storage.artifacts import publish, staging, verify_derived
from ..storage.lookup import readonly_database
from ..storage.identity import digest, file_hash, load_json, write_json
from ..storage.parquet import RowGroupCache
from ..domain.records import AirportRef
from ..domain.scalars import clock_minutes


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


SERVICE_BATCH_ROWS = 32768
PROJECTION_SCHEMA = pa.schema([
    ("service_pk", pa.int64()), ("primary_archive_id", pa.int64()),
    ("primary_source_ordinal", pa.int64()), ("origin_airport_id", pa.int64()), ("destination_airport_id", pa.int64()),
    ("origin", pa.string()), ("destination", pa.string()), ("marketing_carrier", pa.string()),
    ("marketing_flight_number", pa.string()), ("marketing_carrier_dot_id", pa.int64()),
    ("operating_carrier", pa.string()), ("operating_flight_number", pa.string()), ("operating_carrier_dot_id", pa.int64()),
    ("scheduled_departure_utc", pa.timestamp("us", tz="UTC")), ("scheduled_arrival_utc", pa.timestamp("us", tz="UTC")),
    ("flight_date", pa.date32()), ("weekday", pa.int16()), ("quarter", pa.int16()), ("minute", pa.int16()),
    ("local_departure_bucket", pa.int16()), ("scheduled_departure_clock", pa.string()),
    ("scheduled_arrival_clock", pa.string()), ("scheduled_elapsed_minutes", pa.int64()),
    ("originally_scheduled_carrier", pa.string()), ("originally_scheduled_flight_number", pa.string()),
    ("codeshare_partner", pa.string()), ("duplicate_flag", pa.string()),
])
SUPPORT_SCHEMA = pa.schema([("service_pk", pa.int64()), ("ordinal", pa.int64())])
CANDIDATE_COLUMNS = (
    "ordinal", "record_id", "service_id", "flight_date", "origin_airport_id", "origin", "destination_airport_id",
    "destination", "operating_carrier", "operating_flight_number", "operating_carrier_dot_id", "marketing_carrier",
    "marketing_flight_number", "marketing_carrier_dot_id", "originally_scheduled_carrier",
    "originally_scheduled_flight_number", "codeshare_partner", "scheduled_departure_clock", "scheduled_arrival_clock",
    "scheduled_elapsed_minutes", "scheduled_departure", "scheduled_arrival", "duplicate_flag",
)
EVIDENCE_COLUMNS = ("source_record_id", "source_sha256", "source_member", "source_row_id", *OUTCOME_FIELDS)


def evidence_digest(store, source_sha256, member, ordinal):
    row, stops = store.raw_at(source_sha256, member, ordinal, columns=list(EVIDENCE_COLUMNS))
    value = {key: row[key] for key in OUTCOME_FIELDS}
    value["stops"] = [{key: item for key, item in stop.items() if key not in ("source_record_id", "source_row_id")} for stop in stops]
    return digest(json_value(value))


def _text(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def attributes_from_projection(row):
    values = {key: row[key] for key in SCHEDULE_FIELDS if key not in ("scheduled_departure", "scheduled_arrival", "flight_date")}
    values["flight_date"] = _text(row["flight_date"])
    values["scheduled_departure"] = _text(row["scheduled_departure_utc"])
    values["scheduled_arrival"] = _text(row["scheduled_arrival_utc"])
    return values


def stored_service_id(service_id):
    return bytes.fromhex(service_id) if isinstance(service_id, str) else service_id


def public_service_id(service_id):
    return service_id.hex() if isinstance(service_id, bytes) else service_id


def _projection_row(service_pk, archive_id, primary_ordinal, attributes, minute):
    day = date.fromisoformat(attributes["flight_date"])
    return {
        "service_pk": service_pk, "primary_archive_id": archive_id,
        "primary_source_ordinal": primary_ordinal, "origin_airport_id": attributes["origin_airport_id"],
        "destination_airport_id": attributes["destination_airport_id"], "origin": attributes["origin"],
        "destination": attributes["destination"], "marketing_carrier": attributes["marketing_carrier"],
        "marketing_flight_number": attributes["marketing_flight_number"],
        "marketing_carrier_dot_id": attributes["marketing_carrier_dot_id"],
        "operating_carrier": attributes["operating_carrier"],
        "operating_flight_number": attributes["operating_flight_number"],
        "operating_carrier_dot_id": attributes["operating_carrier_dot_id"],
        "scheduled_departure_utc": datetime.fromisoformat(attributes["scheduled_departure"]),
        "scheduled_arrival_utc": datetime.fromisoformat(attributes["scheduled_arrival"]),
        "flight_date": day, "weekday": day.weekday(), "quarter": (day.month - 1) // 3 + 1, "minute": minute,
        "local_departure_bucket": None if minute is None else minute // 360,
        "scheduled_departure_clock": attributes["scheduled_departure_clock"],
        "scheduled_arrival_clock": attributes["scheduled_arrival_clock"],
        "scheduled_elapsed_minutes": attributes["scheduled_elapsed_minutes"],
        "originally_scheduled_carrier": attributes["originally_scheduled_carrier"],
        "originally_scheduled_flight_number": attributes["originally_scheduled_flight_number"],
        "codeshare_partner": attributes["codeshare_partner"], "duplicate_flag": attributes["duplicate_flag"],
    }


_WORKER = {}
_SERVICE_COLUMNS = (
    "service_id", "split", "year", "month", "origin", "destination", "carrier", "quarter", "weekday", "minute",
    "departure", "schedule_issue", "evidence_issue", "archive_id", "primary_ordinal", "evidence_ordinal",
)


def _init_worker(snapshot, lookup_path):
    from ..storage.dataset import DatasetStore
    _WORKER["store"] = DatasetStore(snapshot["path"], lookup_path, verified_snapshot=snapshot)


def _blank_counts():
    return {"source_rows": 0, "support_rows": 0, "services": 0, "collision_groups": 0,
            "schedule_conflicts": 0, "evidence_conflicts": 0, "alias_conflicts": 0,
            "exclusions": Counter(), "duplicate_flags": Counter()}


def _add_counts(total, part):
    for key in ("source_rows", "support_rows", "services", "collision_groups", "schedule_conflicts", "evidence_conflicts", "alias_conflicts"):
        total[key] += part[key]
    total["exclusions"].update(part["exclusions"])
    total["duplicate_flags"].update(part["duplicate_flags"])


def _resolve_partition(store, year, month, policy, output_path):
    partition = next(item for item in store.snapshot["partitions"] if (item["source"]["year"], item["source"]["month"]) == (year, month))
    source = partition["source"]
    archive = store.database.execute("SELECT id FROM archives WHERE source_sha256=? AND member=?",
                                     (source["sha256"], source["csv_member"])).fetchone()
    if archive is None:
        raise ValueError("Normalized partition is absent from the source lookup")
    archive_id = archive["id"]
    period = f"{year:04d}-{month:02d}"
    print(f"Resolving services for {period}", flush=True)
    report = _blank_counts()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_path) as database:
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA cache_size=-8192")
        database.execute("PRAGMA temp_store=FILE")
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.executescript("""
            CREATE TABLE resolved (service_id BLOB PRIMARY KEY, split TEXT NOT NULL, year INTEGER NOT NULL,
                month INTEGER NOT NULL, origin INTEGER NOT NULL, destination INTEGER NOT NULL, carrier TEXT NOT NULL,
                quarter INTEGER NOT NULL, weekday INTEGER NOT NULL, minute INTEGER, departure INTEGER NOT NULL,
                schedule_issue TEXT, evidence_issue TEXT, archive_id INTEGER NOT NULL, primary_ordinal INTEGER NOT NULL,
                evidence_ordinal INTEGER) WITHOUT ROWID;
            CREATE TABLE support_rows (service_id BLOB NOT NULL, ordinal INTEGER NOT NULL);
            CREATE TABLE diagnostics (kind TEXT, split TEXT, record_id TEXT, service_id TEXT, detail TEXT);
            CREATE TEMP TABLE candidates (ordinal INTEGER PRIMARY KEY, record_id TEXT NOT NULL, service_id TEXT NOT NULL,
                split TEXT NOT NULL, flight_date TEXT, origin_airport_id INTEGER, origin TEXT, destination_airport_id INTEGER,
                destination TEXT, operating_carrier TEXT, operating_flight_number TEXT, operating_carrier_dot_id INTEGER,
                marketing_carrier TEXT, marketing_flight_number TEXT, marketing_carrier_dot_id INTEGER,
                originally_scheduled_carrier TEXT, originally_scheduled_flight_number TEXT, codeshare_partner TEXT,
                scheduled_departure_clock TEXT, scheduled_arrival_clock TEXT, scheduled_elapsed_minutes INTEGER,
                scheduled_departure TEXT, scheduled_arrival TEXT, duplicate_flag TEXT);
            CREATE INDEX candidates_service ON candidates(service_id, record_id);
            CREATE TEMP TABLE aliases (alias TEXT, service_id BLOB, PRIMARY KEY(alias, service_id)) WITHOUT ROWID;
        """)
        with pq.ParquetFile(partition["path"] / "flights.parquet") as parquet:
            for batch in parquet.iter_batches(batch_size=SERVICE_BATCH_ROWS, columns=["source_record_id", "source_row_id", *SCHEDULE_FIELDS]):
                candidates, aliases = [], []
                for row in batch.to_pylist():
                    report["source_rows"] += 1
                    report["duplicate_flags"][row["duplicate_flag"] or "missing"] += 1
                    attributes, issue = scheduled_attributes(row)
                    split, record_id = partition["group"], row["source_record_id"]
                    if issue:
                        report["exclusions"][issue] += 1
                        database.execute("INSERT INTO diagnostics VALUES (?, ?, ?, NULL, ?)", ("exclusion", split, record_id, issue))
                        continue
                    day = date.fromisoformat(attributes["flight_date"])
                    if (day.year, day.month) != (source["year"], source["month"]):
                        raise ValueError("Flight date is outside its source partition")
                    key = [attributes[field] for field in ("flight_date", "operating_carrier", "operating_flight_number",
                           "origin_airport_id", "destination_airport_id", "scheduled_departure")]
                    service_id = digest([policy.version, key])
                    candidates.append((row["source_row_id"], record_id, service_id, split, *[attributes[name] for name in CANDIDATE_COLUMNS[3:]]))
                    for carrier, number in (("operating_carrier", "operating_flight_number"),
                                            ("marketing_carrier", "marketing_flight_number"),
                                            ("originally_scheduled_carrier", "originally_scheduled_flight_number")):
                        if attributes[carrier] and attributes[number]:
                            alias = digest([attributes["flight_date"], attributes["origin_airport_id"],
                                            attributes["scheduled_departure"], attributes[carrier], attributes[number]])
                            aliases.append((alias, stored_service_id(service_id)))
                if candidates:
                    database.executemany("INSERT INTO candidates VALUES (" + ",".join("?" * 24) + ")", candidates)
                if aliases:
                    database.executemany("INSERT OR IGNORE INTO aliases VALUES (?, ?)", aliases)
        resolved, support = [], []
        pending = []
        def finish(members):
            if not members:
                return
            service_id = members[0]["service_id"]
            count = len(members)
            report["collision_groups"] += count > 1
            if count > policy.max_group_size:
                database.execute("INSERT INTO diagnostics SELECT 'schedule_conflict', split, record_id, service_id, 'oversized_group' FROM candidates WHERE service_id=?", (service_id,))
                report["schedule_conflicts"] += 1
                return
            values = [dict(member) for member in members]
            canonical_index = min(range(count), key=lambda i: (values[i]["marketing_carrier"], values[i]["marketing_flight_number"], members[i]["record_id"]))
            attributes = values[canonical_index]
            core_fields = ("scheduled_arrival", "origin", "destination", "operating_carrier_dot_id",
                           "scheduled_departure_clock", "scheduled_arrival_clock", "scheduled_elapsed_minutes")
            schedule_issue = "conflicting_schedule_fields" if len({digest([value[key] for key in core_fields]) for value in values}) > 1 else None
            evidence_issue = None
            if count > 1 and len({evidence_digest(store, source["sha256"], source["csv_member"], member["ordinal"]) for member in members}) > 1:
                evidence_issue = "conflicting_observations"
            minute = clock_minutes(attributes["scheduled_departure_clock"])
            day = date.fromisoformat(attributes["flight_date"])
            primary_ordinal = attributes["ordinal"]
            departure = int(datetime.fromisoformat(attributes["scheduled_departure"]).timestamp())
            resolved.append((stored_service_id(service_id), attributes["split"], day.year, day.month,
                             attributes["origin_airport_id"], attributes["destination_airport_id"], attributes["operating_carrier"],
                             (day.month - 1) // 3 + 1, day.weekday(), minute, departure, schedule_issue, evidence_issue,
                             archive_id, primary_ordinal, None if evidence_issue else primary_ordinal))
            if count > 1:
                service_key = resolved[-1][0]
                support.extend((service_key, member["ordinal"]) for member in members)
            report["schedule_conflicts"] += schedule_issue is not None
            report["evidence_conflicts"] += evidence_issue is not None
            report["support_rows"] += count
            report["services"] += 1
        for member in database.execute("SELECT * FROM candidates ORDER BY service_id, record_id").fetchall():
            if pending and pending[0]["service_id"] != member["service_id"]:
                finish(pending)
                pending = []
            pending.append(member)
        finish(pending)
        if resolved:
            database.executemany("INSERT INTO resolved VALUES (" + ",".join("?" * len(_SERVICE_COLUMNS)) + ")", resolved)
        if support:
            database.executemany("INSERT INTO support_rows VALUES (?, ?)", support)
        for conflict in database.execute("SELECT alias FROM aliases GROUP BY alias HAVING count(*)>1"):
            report["alias_conflicts"] += 1
            database.execute("UPDATE resolved SET schedule_issue='ambiguous_service_alias' WHERE service_id IN (SELECT service_id FROM aliases WHERE alias=?)", (conflict["alias"],))
        database.commit()
    return {"year": year, "month": month, "report": report, "database": str(output_path)}


def _resolve_month(year, month, policy, output_path):
    return _resolve_partition(_WORKER["store"], year, month, policy, output_path)


def _merge_resolved(database, stage, resolved, next_pk):
    month_db = resolved["database"]
    year, month = resolved["year"], resolved["month"]
    period = f"{year:04d}-{month:02d}"
    support_name = f"support-{period}.parquet"
    database.execute("ATTACH DATABASE ? AS month", (month_db,))
    try:
        count = database.execute("SELECT count(*) FROM month.resolved").fetchone()[0]
        if next_pk + count > 2 ** 31:
            raise ValueError("Service key exceeds the validated int32 range")
        database.execute(
            "INSERT INTO services (service_id, service_pk, split, year, month, origin, destination, carrier, quarter, "
            "weekday, minute, departure, schedule_issue, evidence_issue, archive_id, primary_ordinal, evidence_ordinal) "
            "SELECT service_id, ? + ROW_NUMBER() OVER (ORDER BY service_id), split, year, month, origin, destination, "
            "carrier, quarter, weekday, minute, departure, schedule_issue, evidence_issue, archive_id, primary_ordinal, "
            "evidence_ordinal FROM month.resolved",
            (next_pk - 1,))
        support = [{"service_pk": row[0], "ordinal": row[1]} for row in database.execute(
            "SELECT services.service_pk, month.support_rows.ordinal FROM month.support_rows "
            "JOIN services ON services.service_id = month.support_rows.service_id "
            "ORDER BY services.service_pk, month.support_rows.ordinal")]
        with pq.ParquetWriter(stage / support_name, SUPPORT_SCHEMA, compression="zstd", use_dictionary=True) as writer:
            if support:
                for start in range(0, len(support), SERVICE_BATCH_ROWS):
                    chunk = support[start:start + SERVICE_BATCH_ROWS]
                    writer.write_table(pa.Table.from_pylist(chunk, schema=SUPPORT_SCHEMA), row_group_size=len(chunk))
            else:
                writer.write_table(pa.Table.from_pylist([], schema=SUPPORT_SCHEMA))
        database.execute("INSERT INTO files(path, sha256) VALUES (?, ?)", (support_name, file_hash(stage / support_name)))
        database.execute("INSERT INTO diagnostics SELECT kind, split, record_id, service_id, detail FROM month.diagnostics")
        database.commit()
    finally:
        database.execute("DETACH DATABASE month")
    Path(month_db).unlink()
    print(f"Merged services for {period}", flush=True)
    return next_pk + count


def _require_free_space(path):
    if shutil.disk_usage(path).free < 1536 * 1024 ** 2:
        raise OSError("Stopping before this month because less than 1.5 GiB remains free")


def build_services(store, output, policy=ResolutionPolicy()):
    identity = {"kind": "services", "schema_version": 1, "builder_version": 3,
                "projection": "typed-monthly-v1",
                "snapshot": store.snapshot["manifest"]["dataset_version"], "lookup": store.lookup_path.name,
                "policy": asdict(policy)}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    report = _blank_counts()
    report["review_reference"] = policy.review_reference
    next_pk = 1
    try:
        with sqlite3.connect(stage / "services.sqlite") as database:
            database.row_factory = sqlite3.Row
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.executescript("""
                CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE, sha256 TEXT NOT NULL);
                CREATE TABLE services (service_id BLOB NOT NULL, service_pk INTEGER NOT NULL, split TEXT NOT NULL,
                    year INTEGER NOT NULL, month INTEGER NOT NULL, origin INTEGER NOT NULL, destination INTEGER NOT NULL,
                    carrier TEXT NOT NULL, quarter INTEGER NOT NULL, weekday INTEGER NOT NULL, minute INTEGER,
                    departure INTEGER NOT NULL, schedule_issue TEXT, evidence_issue TEXT,
                    archive_id INTEGER NOT NULL, primary_ordinal INTEGER NOT NULL, evidence_ordinal INTEGER,
                    PRIMARY KEY (service_id)) WITHOUT ROWID;
                CREATE TABLE diagnostics (kind TEXT, split TEXT, record_id TEXT, service_id TEXT, detail TEXT);
                CREATE INDEX diagnostics_split ON diagnostics(split, kind);
            """)
            order = [(item["source"]["year"], item["source"]["month"]) for item in sorted(
                store.snapshot["partitions"], key=lambda item: (item["source"]["year"], item["source"]["month"]))]
            workers = 1 if len(order) < 2 else min(4, os.cpu_count() or 1)
            work = stage / "work"
            work.mkdir()
            if workers == 1:
                for year, month in order:
                    _require_free_space(stage)
                    resolved = _resolve_partition(store, year, month, policy, work / f"{year:04d}-{month:02d}.sqlite")
                    _add_counts(report, resolved["report"])
                    next_pk = _merge_resolved(database, stage, resolved, next_pk)
            else:
                finished, futures, submitted, merged = {}, {}, 0, 0
                def fill(executor):
                    nonlocal submitted
                    while submitted < len(order) and submitted - merged < workers:
                        _require_free_space(stage)
                        year, month = order[submitted]
                        futures[executor.submit(_resolve_month, year, month, policy, str(work / f"{year:04d}-{month:02d}.sqlite"))] = (year, month)
                        submitted += 1
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(store.snapshot, str(store.lookup_path))) as executor:
                    fill(executor)
                    while merged < len(order):
                        while merged < len(order) and order[merged] in finished:
                            resolved = finished.pop(order[merged])
                            _add_counts(report, resolved["report"])
                            next_pk = _merge_resolved(database, stage, resolved, next_pk)
                            merged += 1
                        if merged >= len(order):
                            break
                        fill(executor)
                        future = next(as_completed(futures))
                        key = futures.pop(future)
                        finished[key] = future.result()
            database.execute("CREATE UNIQUE INDEX services_pk ON services(service_pk)")
            database.execute("CREATE INDEX services_route ON services(split, origin, destination, year, carrier)")
            database.commit()
        shutil.rmtree(work)
        report["service_key"] = "int32"
        write_json(stage / "quality.json", report)
        return publish(stage, final, identity)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


class ServiceCatalog:
    def __init__(self, store, path, *, verify_artifacts=True):
        self.store, self.path = store, Path(path)
        if verify_artifacts:
            self.manifest = verify_derived(path, "services")
        else:
            self.manifest = load_json(self.path / "manifest.json")
            identity = self.manifest["identity"]
            if (self.manifest.get("status") != "complete" or identity.get("kind") != "services"
                    or digest(identity) != self.path.name):
                raise ValueError("Service catalog identity does not match its directory")
        if (self.manifest["identity"]["snapshot"] != store.snapshot["manifest"]["dataset_version"]
                or self.manifest["identity"]["lookup"] != store.lookup_path.name):
            raise ValueError("Service catalog references a different snapshot/lookup")
        self.database = readonly_database(self.path / "services.sqlite")
        self.report = load_json(self.path / "quality.json")
        self.cache = None
        if self.manifest["identity"]["builder_version"] >= 3:
            files = {row["id"]: dict(row) for row in self.database.execute("SELECT * FROM files")}
            for item in files.values():
                if self.manifest["artifacts"].get(item["path"], {}).get("sha256") != item["sha256"]:
                    self.database.close()
                    raise ValueError("Service projection artifact mismatch")
            self.cache = RowGroupCache(self.path, files)

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

    def _legacy(self, row):
        encoded = row["attributes"]
        return {**dict(row), "service_instance_id": row["service_id"],
                "attributes": json.loads(zlib.decompress(encoded) if isinstance(encoded, bytes) else encoded)}

    def _projected(self, row):
        service_id = public_service_id(row["service_id"])
        archive = self.store.archive(row["archive_id"])
        flight, _ = self.store.raw_at(archive["source_sha256"], archive["member"], row["primary_ordinal"],
                                      columns=["source_record_id", "source_row_id", "source_sha256", "source_member", *SCHEDULE_FIELDS])
        attributes, issue = scheduled_attributes(flight)
        key = None if attributes is None else [attributes[field] for field in (
            "flight_date", "operating_carrier", "operating_flight_number", "origin_airport_id",
            "destination_airport_id", "scheduled_departure")]
        if issue or digest([self.manifest["identity"]["policy"]["version"], key]) != service_id:
            raise ValueError("Service projection locator mismatch")
        if attributes["origin_airport_id"] != row["origin"] or attributes["destination_airport_id"] != row["destination"]:
            raise ValueError("Service projection locator mismatch")
        primary = digest([archive["source_sha256"], archive["member"], row["primary_ordinal"]])
        evidence_ordinal = row["evidence_ordinal"]
        evidence = None if evidence_ordinal is None else digest([archive["source_sha256"], archive["member"], evidence_ordinal])
        return {**dict(row), "service_id": service_id, "service_instance_id": service_id,
                "attributes": attributes, "primary_source": primary, "evidence_source": evidence,
                "source_sha256": archive["source_sha256"], "source_member": archive["member"],
                "primary_ordinal": row["primary_ordinal"],
                "source_ordinal": row["primary_ordinal"] if evidence_ordinal is None else evidence_ordinal}

    def _materialize(self, row):
        if self.manifest["identity"]["builder_version"] >= 3:
            return self._projected(row)
        return self._legacy(row)

    def services(self, split=None, schedule_order=False):
        departure = "departure" if self.manifest["identity"]["builder_version"] >= 2 else "json_extract(attributes, '$.scheduled_departure')"
        order = f"origin, {departure}, service_id" if schedule_order else "service_id"
        query = "SELECT * FROM services" + (" WHERE split=?" if split else "") + " ORDER BY " + order
        for row in self.database.execute(query, (split,) if split else ()):
            yield self._materialize(row)

    def get(self, service_id):
        key = stored_service_id(service_id) if self.manifest["identity"]["builder_version"] >= 3 else service_id
        row = self.database.execute("SELECT * FROM services WHERE service_id=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Unknown service instance")
        return self._materialize(row)

    def supports(self, service_id):
        service = self.get(service_id)
        if self.manifest["identity"]["builder_version"] < 3:
            return tuple(row[0] for row in self.database.execute("SELECT record_id FROM candidates WHERE service_id=? ORDER BY record_id", (service_id,)))
        path = self.path / f"support-{service['year']:04d}-{service['month']:02d}.parquet"
        table = pq.read_table(path, columns=["service_pk", "ordinal"], filters=[("service_pk", "=", service["service_pk"])])
        ordinals = table.column("ordinal").to_pylist() or [service["primary_ordinal"]]
        return tuple(sorted(digest([service["source_sha256"], service["source_member"], ordinal]) for ordinal in ordinals))

    def close(self):
        if self.cache is not None:
            self.cache.close()
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
