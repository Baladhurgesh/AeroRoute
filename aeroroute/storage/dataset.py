from pathlib import Path
from types import MappingProxyType

from ..domain.evidence import FlightEvidence, TimeFact
from ..domain.records import SourceRecordRef
from .artifacts import verify_dataset, verify_derived
from .identity import digest
from .lookup import readonly_database
from .parquet import RowGroupCache


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def fact(value):
    return TimeFact(value=value["value"], status=value["status"], method=value["method"],
                    issues=tuple(value["issues"]), checks=tuple(value["checks"]), evidence=tuple(value["evidence"]))


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

    def _record_key(self, record_id):
        return bytes.fromhex(record_id) if self.lookup_manifest["identity"]["lookup_version"] == 2 else record_id

    def source_ref(self, record_id, split):
        inventory = self.inventory(split)
        row = self.database.execute("SELECT * FROM sources WHERE record_id=?", (self._record_key(record_id),)).fetchone()
        if row is None or (split != "all" and row["split"] != split):
            raise ValueError("Source not in requested inventory")
        return self.reference(row, inventory["dataset_version"])

    def raw(self, record_id):
        locator = self.database.execute("SELECT * FROM sources WHERE record_id=?", (self._record_key(record_id),)).fetchone()
        if locator is None:
            raise ValueError("Unknown source record")
        row = self.cache.row(locator["file_id"], locator["row_group"], locator["row_offset"])
        if (row["source_record_id"] != record_id or row["source_row_id"] != locator["ordinal"]
                or row["source_sha256"] != locator["source_sha256"] or row["source_member"] != locator["member"]):
            raise ValueError("Physical source locator mismatch")
        stops = []
        for stop in self.database.execute("SELECT * FROM stops WHERE record_id=? ORDER BY stop_index", (self._record_key(record_id),)):
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
        row = self.database.execute("SELECT * FROM sources WHERE record_id=?", (self._record_key(record_id),)).fetchone()
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
