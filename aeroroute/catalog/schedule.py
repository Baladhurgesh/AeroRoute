import json
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..catalog.services import public_service_id, scheduled_attributes
from ..reference.timezones import pinned_zone
from ..storage.parquet import RowGroupCache
from ..storage.lookup import readonly_database
from ..storage.artifacts import artifact_metadata, publish, staging, verify_derived
from ..storage.identity import digest
from ..domain.records import AirportRef, FlightOption, ScheduleRef, check_type


SCHEDULE_SCHEMA = pa.schema([("service_id", pa.string()), ("attributes", pa.string()), ("primary_source", pa.string())])


def _schedule_month(selected, flights_path, year, month, shard_rows, output_path):
    pending, seen = [], 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_path) as database, pq.ParquetFile(flights_path) as parquet:
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("""
            CREATE TABLE flights (service_id TEXT NOT NULL, origin INTEGER NOT NULL, destination INTEGER NOT NULL,
                departure TEXT NOT NULL, arrival TEXT NOT NULL, origin_code TEXT NOT NULL, destination_code TEXT NOT NULL,
                marketing_carrier TEXT NOT NULL, marketing_flight_number TEXT NOT NULL, operating_carrier TEXT NOT NULL,
                operating_flight_number TEXT NOT NULL, source_sha256 TEXT NOT NULL, source_member TEXT NOT NULL,
                source_ordinal INTEGER NOT NULL, has_evidence INTEGER NOT NULL)
        """)
        from ..catalog.services import SCHEDULE_FIELDS
        columns = ["source_row_id", "source_sha256", "source_member", *SCHEDULE_FIELDS]
        for batch in parquet.iter_batches(batch_size=shard_rows, columns=columns):
            for row in batch.to_pylist():
                known = selected.get(row["source_row_id"])
                if known is None:
                    continue
                attributes, issue = scheduled_attributes(row)
                service_id, origin, destination, has_evidence = known
                if issue or attributes["origin_airport_id"] != origin or attributes["destination_airport_id"] != destination:
                    raise ValueError(f"Schedule row does not match catalog service {service_id}")
                pending.append((service_id, origin, destination, attributes["scheduled_departure"], attributes["scheduled_arrival"],
                                attributes["origin"], attributes["destination"], attributes["marketing_carrier"],
                                attributes["marketing_flight_number"], attributes["operating_carrier"],
                                attributes["operating_flight_number"], row["source_sha256"], row["source_member"],
                                row["source_row_id"], int(has_evidence)))
                seen += 1
            if pending:
                database.executemany("INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", pending)
                pending.clear()
        database.commit()
    if seen != len(selected):
        raise ValueError(f"Schedule month {year:04d}-{month:02d} is missing catalog services")
    print(f"Indexed {year:04d}-{month:02d}: {seen}", flush=True)
    return str(output_path), seen


def build_schedule(catalog, output, periods, shard_rows=65536):
    periods = tuple(sorted(set(tuple(period) for period in periods)))
    available = {(p["source"]["year"], p["source"]["month"]) for p in catalog.store.snapshot["partitions"]}
    if not periods or not set(periods) <= available or type(shard_rows) is not int or shard_rows < 1:
        raise ValueError("Invalid or unavailable schedule periods")
    identity = {"kind": "schedule", "schema_version": 1, "builder_version": 2, "catalog": catalog.path.name,
                "periods": [list(period) for period in periods], "shard_rows": shard_rows}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    zones = {}
    for partition in catalog.store.snapshot["partitions"]:
        if (partition["source"]["year"], partition["source"]["month"]) not in periods:
            continue
        with pq.ParquetFile(partition["path"] / "airports.parquet") as parquet:
            for batch in parquet.iter_batches(columns=["airport_id", "time_zone", "status"]):
                for row in batch.to_pylist():
                    if row["airport_id"] is not None:
                        zones.setdefault(str(row["airport_id"]), set()).add(row["time_zone"] if row["status"] == "resolved" else None)
    with sqlite3.connect(stage / "schedule.sqlite") as database:
        database.execute("PRAGMA cache_size=-8192")
        database.execute("PRAGMA temp_store=FILE")
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.executescript("""
            CREATE TABLE flights (service_id TEXT NOT NULL, origin INTEGER NOT NULL, destination INTEGER NOT NULL,
                departure TEXT NOT NULL, arrival TEXT NOT NULL, origin_code TEXT NOT NULL, destination_code TEXT NOT NULL,
                marketing_carrier TEXT NOT NULL, marketing_flight_number TEXT NOT NULL, operating_carrier TEXT NOT NULL,
                operating_flight_number TEXT NOT NULL, source_sha256 TEXT NOT NULL, source_member TEXT NOT NULL,
                source_ordinal INTEGER NOT NULL, has_evidence INTEGER NOT NULL);
        """)
        work = stage / "work"
        work.mkdir()
        selected_by_period = {period: {} for period in periods}
        print("Reading catalog services once", flush=True)
        kept = scanned = 0
        for row in catalog.database.execute(
                "SELECT service_id, origin, destination, primary_ordinal, evidence_ordinal, year, month FROM services"):
            scanned += 1
            period = (row["year"], row["month"])
            bucket = selected_by_period.get(period)
            if bucket is not None:
                bucket[row["primary_ordinal"]] = (public_service_id(row["service_id"]), row["origin"], row["destination"],
                                                   row["evidence_ordinal"] is not None)
                kept += 1
            if scanned % 2000000 == 0:
                print(f"Catalog rows read: {scanned}; services kept: {kept}", flush=True)
        print(f"Catalog rows read: {scanned}; services kept: {kept}", flush=True)
        jobs = []
        for partition in catalog.store.snapshot["partitions"]:
            source = partition["source"]
            period = (source["year"], source["month"])
            if period not in periods:
                continue
            jobs.append((selected_by_period[period], str(partition["path"] / "flights.parquet"),
                         period[0], period[1], shard_rows, str(work / f"{period[0]:04d}-{period[1]:02d}.sqlite")))
        print(f"Indexing {len(jobs)} schedule periods", flush=True)
        workers = 1 if len(jobs) < 2 else min(4, os.cpu_count() or 1)
        if workers == 1:
            results = [_schedule_month(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_schedule_month, *job) for job in jobs]
                results = [future.result() for future in as_completed(futures)]
        indexed = 0
        for path, count in results:
            database.execute("ATTACH DATABASE ? AS month", (path,))
            database.execute("INSERT INTO flights SELECT service_id, origin, destination, departure, arrival, origin_code, "
                             "destination_code, marketing_carrier, marketing_flight_number, operating_carrier, "
                             "operating_flight_number, source_sha256, source_member, source_ordinal, has_evidence FROM month.flights")
            database.commit()
            database.execute("DETACH DATABASE month")
            Path(path).unlink()
            indexed += count
        database.execute("CREATE UNIQUE INDEX flights_service ON flights(service_id)")
        database.execute("CREATE INDEX departures ON flights(origin, departure, service_id)")
        print(f"Schedule flights indexed: {indexed}", flush=True)
        database.commit()
    database.close()
    work.rmdir()
    return publish(stage, final, identity, airport_zones={key: sorted(value, key=lambda item: item or "") for key, value in zones.items()})


@dataclass(frozen=True)
class DepartureCursor:
    query_sha256: str
    departure: str
    service_id: str


@dataclass(frozen=True)
class DepartureResult:
    flights: tuple[FlightOption, ...]
    complete: bool
    eligibility_policy: str
    next_cursor: DepartureCursor | None = None


class ScheduleStore:
    def __init__(self, catalog, path, cache_bytes=16 * 1024 ** 2, max_handles=8):
        self.catalog, self.path = catalog, Path(path)
        self.manifest = verify_derived(path, "schedule")
        if self.manifest["identity"]["catalog"] != catalog.path.name:
            raise ValueError("Schedule belongs to a different service catalog")
        self.database = readonly_database(self.path / "schedule.sqlite")
        if self.manifest["identity"]["builder_version"] >= 2:
            self.cache = RowGroupCache(self.path, {}, cache_bytes, max_handles)
        else:
            files = {row["id"]: dict(row) for row in self.database.execute("SELECT * FROM files")}
            for item in files.values():
                if self.manifest["artifacts"].get(item["path"], {}).get("sha256") != item["sha256"]:
                    raise ValueError("Schedule locator artifact mismatch")
            self.cache = RowGroupCache(self.path, files, cache_bytes, max_handles)
        self.reference = ScheduleRef(snapshot_id=self.path.name, sha256=digest(self.manifest))
        self.periods = tuple(tuple(period) for period in self.manifest["identity"]["periods"])

    def get_flight(self, service_id):
        location = self.database.execute("SELECT * FROM flights WHERE service_id=?", (service_id,)).fetchone()
        if location is None:
            raise ValueError("Flight not in schedule snapshot")
        if self.manifest["identity"]["builder_version"] >= 2:
            return FlightOption(flight_id=service_id,
                origin=AirportRef(airport_id=location["origin"], code=location["origin_code"]),
                destination=AirportRef(airport_id=location["destination"], code=location["destination_code"]),
                marketing_carrier=location["marketing_carrier"], marketing_flight_number=location["marketing_flight_number"],
                operating_carrier=location["operating_carrier"], operating_flight_number=location["operating_flight_number"],
                scheduled_departure_at=datetime.fromisoformat(location["departure"]),
                scheduled_arrival_at=datetime.fromisoformat(location["arrival"]), schedule_ref=self.reference,
                schedule_source=self.catalog.store.reference_at(location["source_sha256"], location["source_member"],
                                                                location["source_ordinal"], "all"))
        row = self.cache.row(location["file_id"], location["row_group"], location["row_offset"])
        if row["service_id"] != service_id:
            raise ValueError("Schedule physical locator mismatch")
        value = json.loads(row["attributes"])
        return FlightOption(flight_id=service_id,
            origin=AirportRef(airport_id=value["origin_airport_id"], code=value["origin"]),
            destination=AirportRef(airport_id=value["destination_airport_id"], code=value["destination"]),
            marketing_carrier=value["marketing_carrier"], marketing_flight_number=value["marketing_flight_number"],
            operating_carrier=value["operating_carrier"], operating_flight_number=value["operating_flight_number"],
            scheduled_departure_at=datetime.fromisoformat(value["scheduled_departure"]),
            scheduled_arrival_at=datetime.fromisoformat(value["scheduled_arrival"]), schedule_ref=self.reference,
            schedule_source=self.catalog.store.source_ref(row["primary_source"], "all"))

    def all_flights(self):
        for row in self.database.execute("SELECT service_id FROM flights ORDER BY departure, service_id"):
            yield self.get_flight(row[0])

    def check_coverage(self, origin, start, end):
        zones = self.manifest["airport_zones"].get(str(origin), [])
        if len(zones) != 1 or zones[0] is None:
            raise ValueError("Schedule coverage has no unambiguous airport timezone")
        zone = pinned_zone(zones[0])
        cursor = start
        for year, month in self.periods:
            lower = datetime(year, month, 1, tzinfo=zone).astimezone(timezone.utc)
            upper = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=zone).astimezone(timezone.utc)
            if lower <= cursor < upper:
                cursor = min(upper, end)
                if cursor == end:
                    return
        raise ValueError("Requested window is outside complete schedule coverage")

    def departures(self, origin_airport_id, ready_to_board_at, departure_before, attempted_service_ids=(), *, page_size=1000, cursor=None):
        check_type(ready_to_board_at, datetime, "ready_to_board_at")
        check_type(departure_before, datetime, "departure_before")
        if (type(origin_airport_id) is not int or departure_before <= ready_to_board_at
                or type(page_size) is not int or not 1 <= page_size <= 10000):
            raise ValueError("Invalid departure query")
        self.check_coverage(origin_airport_id, ready_to_board_at, departure_before)
        attempted = set(attempted_service_ids)
        query_id = digest([self.reference.sha256, origin_airport_id, ready_to_board_at.isoformat(),
                           departure_before.isoformat(), sorted(attempted)])
        query = "SELECT service_id, departure FROM flights WHERE origin=? AND departure>=? AND departure<?"
        args = [origin_airport_id, ready_to_board_at.isoformat(), departure_before.isoformat()]
        if cursor is not None:
            if not isinstance(cursor, DepartureCursor) or cursor.query_sha256 != query_id:
                raise ValueError("Departure cursor belongs to a different query")
            query += " AND (departure, service_id)>(?, ?)"
            args.extend((cursor.departure, cursor.service_id))
        candidates = []
        for row in self.database.execute(query + " ORDER BY departure, service_id", args):
            if row["service_id"] not in attempted:
                candidates.append(row)
                if len(candidates) > page_size:
                    break
        complete = len(candidates) <= page_size
        candidates = candidates[:page_size]
        flights = tuple(self.get_flight(row["service_id"]) for row in candidates)
        following = None if complete else DepartureCursor(query_id, candidates[-1]["departure"], candidates[-1]["service_id"])
        return DepartureResult(flights, complete, self.catalog.manifest["identity"]["policy"]["version"], following)

    def close(self):
        self.cache.close()
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
