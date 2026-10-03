import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .airport_timezones import pinned_zone
from .data_access import RowGroupCache, publish, readonly_database, staging
from .data_snapshots import artifact_metadata, digest, verify_derived
from .records import AirportRef, FlightOption, ScheduleRef, check_type


SCHEDULE_SCHEMA = pa.schema([("service_id", pa.string()), ("attributes", pa.string()), ("primary_source", pa.string())])


def build_schedule(catalog, output, periods, shard_rows=65536):
    periods = tuple(sorted(set(tuple(period) for period in periods)))
    available = {(p["source"]["year"], p["source"]["month"]) for p in catalog.store.snapshot["partitions"]}
    if not periods or not set(periods) <= available or type(shard_rows) is not int or shard_rows < 1:
        raise ValueError("Invalid or unavailable schedule periods")
    for partition in catalog.store.snapshot["partitions"]:
        if (partition["source"]["year"], partition["source"]["month"]) in periods:
            catalog.require_ready(partition["group"], evidence=False)
    identity = {"kind": "schedule", "schema_version": 1, "builder_version": 1, "catalog": catalog.path.name,
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
        database.executescript("""
            CREATE TABLE flights (service_id TEXT PRIMARY KEY, origin INTEGER, departure TEXT,
                file_id INTEGER, row_group INTEGER, row_offset INTEGER);
            CREATE INDEX departures ON flights(origin, departure, service_id);
            CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE, sha256 TEXT);
        """)
        rows, locations, shard = [], [], 0
        def flush():
            nonlocal shard
            if not rows:
                return
            name = f"schedule-{shard:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEDULE_SCHEMA), stage / name,
                           row_group_size=min(4096, shard_rows), compression="zstd")
            metadata = artifact_metadata(stage / name)
            file_id = database.execute("INSERT INTO files(path, sha256) VALUES (?, ?)", (name, metadata["sha256"])).lastrowid
            with pq.ParquetFile(stage / name) as parquet:
                offset = 0
                for group in range(parquet.num_row_groups):
                    count = parquet.metadata.row_group(group).num_rows
                    database.executemany("INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?)",
                        [(service, origin, departure, file_id, group, index) for index, (service, origin, departure)
                         in enumerate(locations[offset:offset + count])])
                    offset += count
            rows.clear()
            locations.clear()
            shard += 1
        for service in catalog.services(schedule_order=True):
            if (service["year"], service["month"]) not in periods:
                continue
            attributes = service["attributes"]
            rows.append({"service_id": service["service_instance_id"], "attributes": json.dumps(attributes, sort_keys=True),
                         "primary_source": service["primary_source"]})
            locations.append((service["service_instance_id"], service["origin"], attributes["scheduled_departure"]))
            if len(rows) >= shard_rows:
                flush()
        flush()
        database.commit()
    database.close()
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
