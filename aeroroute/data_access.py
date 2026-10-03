import sqlite3
import uuid
from collections import Counter, OrderedDict
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pyarrow.parquet as pq

from .data_snapshots import artifact_metadata, contained_path, digest, verify_dataset, verify_derived, write_json
from .flight_times import TimeFact
from .records import SourceRecordRef


def readonly_database(path):
    database = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA cache_size=-8192")
    return database


def staging(output, identity):
    output = Path(output)
    final = output / digest(identity)
    if final.exists():
        manifest = verify_derived(final, identity["kind"])
        if manifest["identity"] != identity:
            raise ValueError("Derived identity mismatch")
        return final, None
    output.mkdir(parents=True, exist_ok=True)
    stage = output / (".pending-" + uuid.uuid4().hex)
    stage.mkdir()
    return final, stage


def publish(stage, final, identity, **details):
    artifacts = {path.name: artifact_metadata(path) for path in stage.iterdir() if path.is_file()}
    write_json(stage / "manifest.json", {"status": "complete", "identity": identity,
                                         "artifacts": artifacts, **details})
    if final.exists():
        raise FileExistsError(f"Refusing to replace completed artifact: {final}")
    stage.rename(final)
    return final


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def fact(value):
    return TimeFact(value=value["value"], status=value["status"], method=value["method"],
                    issues=tuple(value["issues"]), checks=tuple(value["checks"]), evidence=tuple(value["evidence"]))


@dataclass(frozen=True)
class FlightEvidence:
    source: SourceRecordRef
    flight: Mapping[str, object]
    stops: tuple[Mapping[str, object], ...]


def build_lookup(snapshot_path, output):
    snapshot = verify_dataset(snapshot_path)
    identity = {"kind": "lookup", "schema_version": 1, "lookup_version": 1,
                "snapshot": snapshot["manifest"]["dataset_version"]}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    with closing(sqlite3.connect(stage / "lookup.sqlite")) as database, database:
        database.execute("PRAGMA cache_size=-8192")
        database.executescript("""
            CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE, sha256 TEXT NOT NULL);
            CREATE TABLE sources (record_id TEXT PRIMARY KEY, source_sha256 TEXT, member TEXT,
                ordinal INTEGER, source_file TEXT, split TEXT, year INTEGER, month INTEGER,
                file_id INTEGER, row_group INTEGER, row_offset INTEGER, stop_count INTEGER DEFAULT 0,
                UNIQUE(source_sha256, member, ordinal));
            CREATE INDEX source_split ON sources(split, year, month, ordinal);
            CREATE TABLE stops (record_id TEXT, stop_index INTEGER, file_id INTEGER,
                row_group INTEGER, row_offset INTEGER, PRIMARY KEY(record_id, stop_index));
        """)
        for partition in snapshot["partitions"]:
            source, manifest, path = partition["source"], partition["manifest"], partition["path"]
            for name in ("flights.parquet", "diversion_stops.parquet"):
                parquet_path = path / name
                file_id = database.execute("INSERT INTO files(path, sha256) VALUES (?, ?)",
                    (parquet_path.relative_to(snapshot["root"]).as_posix(), manifest["artifacts"][name]["sha256"])).lastrowid
                with pq.ParquetFile(parquet_path) as parquet:
                    columns = (["source_record_id", "source_sha256", "source_member", "source_file", "source_row_id"]
                               if name == "flights.parquet" else ["source_record_id", "source_row_id", "stop_index"])
                    for group in range(parquet.num_row_groups):
                        for offset, row in enumerate(parquet.read_row_group(group, columns=columns).to_pylist()):
                            ordinal = row["source_row_id"]
                            record_id = digest([source["sha256"], source["csv_member"], ordinal])
                            if (type(ordinal) is not int or not 1 <= ordinal <= source["row_count"]
                                    or row["source_record_id"] != record_id):
                                raise ValueError("Invalid source ordinal or record identity")
                            if name == "flights.parquet":
                                if (row["source_sha256"], row["source_member"], row["source_file"]) != (
                                        source["sha256"], source["csv_member"], source["path"]):
                                    raise ValueError("Source row contradicts inventory")
                                database.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                                    (record_id, source["sha256"], source["csv_member"], ordinal, source["path"],
                                     partition["group"], source["year"], source["month"], file_id, group, offset))
                            else:
                                if not 1 <= row["stop_index"] <= 5:
                                    raise ValueError("Invalid diversion stop index")
                                changed = database.execute("UPDATE sources SET stop_count=stop_count+1 WHERE record_id=?", (record_id,))
                                if changed.rowcount != 1:
                                    raise ValueError("Orphan diversion stop")
                                database.execute("INSERT INTO stops VALUES (?, ?, ?, ?, ?)",
                                                 (record_id, row["stop_index"], file_id, group, offset))
        database.commit()
    return publish(stage, final, identity)


class RowGroupCache:
    def __init__(self, root, files, cache_bytes=64 * 1024 ** 2, max_handles=16):
        if type(cache_bytes) is not int or cache_bytes < 0 or type(max_handles) is not int or max_handles < 1:
            raise ValueError("Invalid cache limits")
        self.root, self.files = Path(root), files
        self.cache_bytes, self.max_handles = cache_bytes, max_handles
        self.groups, self.handles = OrderedDict(), OrderedDict()
        self.cache_size_bytes = 0
        self.metrics = Counter()

    def row(self, file_id, group, offset, columns=None):
        item = self.files[file_id]
        key = (item["sha256"], group, tuple(columns) if columns is not None else None)
        if key in self.groups:
            table = self.groups.pop(key)
            self.groups[key] = table
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1
            path = contained_path(self.root, item["path"])
            handle = self.handles.pop(file_id, None)
            if handle is None:
                if len(self.handles) >= self.max_handles:
                    self.handles.popitem(last=False)[1].close()
                handle = pq.ParquetFile(path)
            self.handles[file_id] = handle
            if not 0 <= group < handle.num_row_groups:
                raise ValueError("Row group locator out of bounds")
            table = handle.read_row_group(group, columns=columns)
            self.metrics["row_groups_read"] += 1
            self.metrics["decoded_bytes"] += table.nbytes
            self.metrics["decoded_rows"] += table.num_rows
            if table.nbytes <= self.cache_bytes:
                while self.groups and self.cache_size_bytes + table.nbytes > self.cache_bytes:
                    self.cache_size_bytes -= self.groups.popitem(last=False)[1].nbytes
                self.groups[key] = table
                self.cache_size_bytes += table.nbytes
        if not 0 <= offset < table.num_rows:
            raise ValueError("Row offset out of bounds")
        return table.slice(offset, 1).to_pylist()[0]

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.groups.clear()
        self.cache_size_bytes = 0


class DatasetStore:
    def __init__(self, snapshot_path, lookup_path, cache_bytes=64 * 1024 ** 2, max_handles=16):
        self.snapshot = verify_dataset(snapshot_path)
        self.lookup_path = Path(lookup_path)
        self.lookup_manifest = verify_derived(lookup_path, "lookup")
        if self.lookup_manifest["identity"]["snapshot"] != self.snapshot["manifest"]["dataset_version"]:
            raise ValueError("Lookup references a different normalized snapshot")
        self.database = readonly_database(self.lookup_path / "lookup.sqlite")
        files = {row["id"]: dict(row) for row in self.database.execute("SELECT * FROM files")}
        expected = {str((p["path"] / name).relative_to(self.snapshot["root"])): p["manifest"]["artifacts"][name]["sha256"]
                    for p in self.snapshot["partitions"] for name in ("flights.parquet", "diversion_stops.parquet")}
        if {row["path"]: row["sha256"] for row in files.values()} != expected:
            self.database.close()
            raise ValueError("Lookup file identities do not match snapshot")
        self.cache = RowGroupCache(self.snapshot["root"], files, cache_bytes, max_handles)
        self.metrics, self.handles = self.cache.metrics, self.cache.handles

    @property
    def cache_size_bytes(self):
        return self.cache.cache_size_bytes

    def inventory(self, split):
        if split not in self.snapshot["inventories"]:
            raise ValueError("Unknown source split")
        inventory = self.snapshot["inventories"][split]
        if not inventory["sources"]:
            raise ValueError(f"No source coverage for split: {split}")
        return inventory

    def source_refs(self, split):
        version = self.inventory(split)["dataset_version"]
        query, args = ("SELECT * FROM sources", ()) if split == "all" else ("SELECT * FROM sources WHERE split=?", (split,))
        for row in self.database.execute(query + " ORDER BY year, month, ordinal", args):
            yield self.reference(row, version)

    @staticmethod
    def reference(row, version):
        return SourceRecordRef(dataset_version=version, source_file=row["source_file"], source_sha256=row["source_sha256"],
                               csv_member=row["member"], source_row_id=row["ordinal"])

    def source_ref(self, record_id, split):
        inventory = self.inventory(split)
        row = self.database.execute("SELECT * FROM sources WHERE record_id=?", (record_id,)).fetchone()
        if row is None or (split != "all" and row["split"] != split):
            raise ValueError("Source not in requested inventory")
        return self.reference(row, inventory["dataset_version"])

    def raw(self, record_id):
        locator = self.database.execute("SELECT * FROM sources WHERE record_id=?", (record_id,)).fetchone()
        if locator is None:
            raise ValueError("Unknown source record")
        row = self.cache.row(locator["file_id"], locator["row_group"], locator["row_offset"])
        if (row["source_record_id"] != record_id or row["source_row_id"] != locator["ordinal"]
                or row["source_sha256"] != locator["source_sha256"] or row["source_member"] != locator["member"]):
            raise ValueError("Physical source locator mismatch")
        stops = []
        for stop in self.database.execute("SELECT * FROM stops WHERE record_id=? ORDER BY stop_index", (record_id,)):
            value = self.cache.row(stop["file_id"], stop["row_group"], stop["row_offset"])
            if (value["source_record_id"] != record_id or value["source_row_id"] != locator["ordinal"]
                    or value["stop_index"] != stop["stop_index"]):
                raise ValueError("Physical stop locator mismatch")
            stops.append(value)
        if len(stops) != locator["stop_count"]:
            raise ValueError("Missing indexed diversion stops")
        return row, stops

    def get(self, source, inventory=None):
        record_id = digest([source.source_sha256, source.csv_member, source.source_row_id])
        row = self.database.execute("SELECT * FROM sources WHERE record_id=?", (record_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown source reference")
        allowed = [value["dataset_version"] for name, value in self.snapshot["inventories"].items()
                   if name in ("all", row["split"])]
        if inventory is not None:
            sources = inventory["sources"]
            known = self.snapshot["inventories"]["all"]["sources"]
            if (inventory["dataset_version"] != digest(sources) or not sources
                    or len({(item["year"], item["month"]) for item in sources}) != len(sources)
                    or any(item not in known for item in sources)):
                raise ValueError("Invalid selected source inventory")
            if any(item["sha256"] == source.source_sha256 and item["csv_member"] == source.csv_member for item in sources):
                allowed.append(inventory["dataset_version"])
        if source.dataset_version not in allowed or source.source_file != row["source_file"]:
            raise ValueError("Source reference inventory mismatch")
        flight, stops = self.raw(record_id)
        for name in ("scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"):
            flight[name] = fact(flight[name])
        for stop in stops:
            stop["landing"], stop["departure"] = fact(stop["landing"]), fact(stop["departure"])
        return FlightEvidence(source, freeze(flight), tuple(freeze(stop) for stop in stops))

    def close(self):
        self.cache.close()
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
