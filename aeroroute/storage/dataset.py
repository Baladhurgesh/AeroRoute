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
    def __init__(self, snapshot_path, lookup_path, cache_bytes=64 * 1024 ** 2, max_handles=16, verified_snapshot=None):
        self.snapshot = verify_dataset(snapshot_path) if verified_snapshot is None else verified_snapshot
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
        self._archives = {}

    @property
    def lookup_version(self):
        return self.lookup_manifest["identity"]["lookup_version"]

    @property
    def cache_size_bytes(self):
        return self.cache.cache_size_bytes

    def archive(self, archive_id):
        row = self._archives.get(archive_id)
        if row is None:
            row = self.database.execute("SELECT * FROM archives WHERE id=?", (archive_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown source archive")
            self._archives[archive_id] = row
        return row

    def inventory(self, split):
        if split not in self.snapshot["inventories"]:
            raise ValueError("Unknown source split")
        inventory = self.snapshot["inventories"][split]
        if not inventory["sources"]:
            raise ValueError(f"No source coverage for split: {split}")
        return inventory

    def source_refs(self, split):
        version = self.inventory(split)["dataset_version"]
        if self.lookup_version < 3:
            query, args = ("SELECT * FROM sources", ()) if split == "all" else ("SELECT * FROM sources WHERE split=?", (split,))
            for row in self.database.execute(query + " ORDER BY year, month, ordinal", args):
                yield self.reference(row, version)
            return
        query, args = ("SELECT * FROM archives", ()) if split == "all" else ("SELECT * FROM archives WHERE split=?", (split,))
        for archive in self.database.execute(query + " ORDER BY year, month, id", args):
            for ordinal in self._ordinals(archive["id"]):
                yield self.reference({**dict(archive), "ordinal": ordinal}, version)

    def _ordinals(self, archive_id):
        ranges = list(self.database.execute(
            "SELECT first_ordinal, last_ordinal FROM ranges WHERE archive_id=? ORDER BY first_ordinal", (archive_id,)))
        explicit = list(self.database.execute("SELECT ordinal FROM explicit_rows WHERE archive_id=? ORDER BY ordinal", (archive_id,)))
        range_index = explicit_index = 0
        while range_index < len(ranges) or explicit_index < len(explicit):
            if range_index < len(ranges) and (explicit_index >= len(explicit) or ranges[range_index]["first_ordinal"] <= explicit[explicit_index]["ordinal"]):
                first, last = ranges[range_index]["first_ordinal"], ranges[range_index]["last_ordinal"]
                yield from range(first, last + 1)
                range_index += 1
            else:
                yield explicit[explicit_index]["ordinal"]
                explicit_index += 1

    def _locate(self, source_sha256, member, ordinal):
        if self.lookup_version < 3:
            return self.database.execute(
                "SELECT * FROM sources WHERE source_sha256=? AND member=? AND ordinal=?",
                (source_sha256, member, ordinal)).fetchone()
        archive = self.database.execute(
            "SELECT * FROM archives WHERE source_sha256=? AND member=?", (source_sha256, member)).fetchone()
        if archive is None:
            return None
        ranged = self.database.execute(
            "SELECT * FROM ranges WHERE archive_id=? AND first_ordinal<=? AND last_ordinal>=?",
            (archive["id"], ordinal, ordinal)).fetchone()
        if ranged is not None:
            counted = self.database.execute(
                "SELECT stop_count FROM stop_counts WHERE archive_id=? AND ordinal=?", (archive["id"], ordinal)).fetchone()
            return {**dict(archive), "ordinal": ordinal, "file_id": ranged["file_id"], "row_group": ranged["row_group"],
                    "row_offset": ordinal - ranged["first_ordinal"], "stop_count": 0 if counted is None else counted["stop_count"]}
        explicit = self.database.execute(
            "SELECT * FROM explicit_rows WHERE archive_id=? AND ordinal=?", (archive["id"], ordinal)).fetchone()
        if explicit is None:
            return None
        counted = self.database.execute(
            "SELECT stop_count FROM stop_counts WHERE archive_id=? AND ordinal=?", (archive["id"], ordinal)).fetchone()
        return {**dict(archive), "ordinal": ordinal, "file_id": explicit["file_id"], "row_group": explicit["row_group"],
                "row_offset": explicit["row_offset"], "stop_count": 0 if counted is None else counted["stop_count"]}

    def _read_located(self, located, columns=None):
        row = self.cache.row(located["file_id"], located["row_group"], located["row_offset"], columns)
        record_id = digest([located["source_sha256"], located["member"], located["ordinal"]])
        if (row["source_record_id"] != record_id or row["source_row_id"] != located["ordinal"]
                or row["source_sha256"] != located["source_sha256"] or row["source_member"] != located["member"]):
            raise ValueError("Physical source locator mismatch")
        if self.lookup_version >= 3:
            stops = self.database.execute(
                "SELECT * FROM stops WHERE archive_id=? AND ordinal=? ORDER BY stop_index", (located["id"], located["ordinal"]))
        else:
            stops = self.database.execute(
                "SELECT * FROM stops WHERE record_id=? ORDER BY stop_index", (self._record_key(record_id),))
        values = []
        for stop in stops:
            value = self.cache.row(stop["file_id"], stop["row_group"], stop["row_offset"])
            if (value["source_record_id"] != record_id or value["source_row_id"] != located["ordinal"]
                    or value["stop_index"] != stop["stop_index"]):
                raise ValueError("Physical stop locator mismatch")
            values.append(value)
        if len(values) != located["stop_count"]:
            raise ValueError("Missing indexed diversion stops")
        return row, values

    @staticmethod
    def reference(row, version):
        return SourceRecordRef(dataset_version=version, source_file=row["source_file"], source_sha256=row["source_sha256"],
                               csv_member=row["member"], source_row_id=row["ordinal"])

    def _record_key(self, record_id):
        return bytes.fromhex(record_id) if self.lookup_manifest["identity"]["lookup_version"] == 2 else record_id

    def source_ref(self, record_id, split):
        if self.lookup_version >= 3:
            raise ValueError("Physical source locator requires archive coordinates")
        inventory = self.inventory(split)
        row = self.database.execute("SELECT * FROM sources WHERE record_id=?", (self._record_key(record_id),)).fetchone()
        if row is None or (split != "all" and row["split"] != split):
            raise ValueError("Source not in requested inventory")
        return self.reference(row, inventory["dataset_version"])

    def reference_at(self, source_sha256, member, ordinal, split):
        inventory = self.inventory(split)
        located = self._locate(source_sha256, member, ordinal)
        if located is None or (split != "all" and located["split"] != split):
            raise ValueError("Source not in requested inventory")
        return self.reference(located, inventory["dataset_version"])

    def raw(self, record_id):
        if self.lookup_version >= 3:
            raise ValueError("Physical source locator requires archive coordinates")
        locator = self.database.execute("SELECT * FROM sources WHERE record_id=?", (self._record_key(record_id),)).fetchone()
        if locator is None:
            raise ValueError("Unknown source record")
        return self._read_located(locator)

    def raw_at(self, source_sha256, member, ordinal, columns=None):
        located = self._locate(source_sha256, member, ordinal)
        if located is None:
            raise ValueError("Unknown source record")
        return self._read_located(located, columns)

    def get(self, source, inventory=None):
        if self.lookup_version < 3:
            record_id = digest([source.source_sha256, source.csv_member, source.source_row_id])
            located = self.database.execute("SELECT * FROM sources WHERE record_id=?", (self._record_key(record_id),)).fetchone()
        else:
            located = self._locate(source.source_sha256, source.csv_member, source.source_row_id)
        if located is None:
            raise ValueError("Unknown source reference")
        allowed = [value["dataset_version"] for name, value in self.snapshot["inventories"].items()
                   if name in ("all", located["split"])]
        if inventory is not None:
            sources = inventory["sources"]
            known = self.snapshot["inventories"]["all"]["sources"]
            if (inventory["dataset_version"] != digest(sources) or not sources
                    or len({(item["year"], item["month"]) for item in sources}) != len(sources)
                    or any(item not in known for item in sources)):
                raise ValueError("Invalid selected source inventory")
            if any(item["sha256"] == source.source_sha256 and item["csv_member"] == source.csv_member for item in sources):
                allowed.append(inventory["dataset_version"])
        if source.dataset_version not in allowed or source.source_file != located["source_file"]:
            raise ValueError("Source reference inventory mismatch")
        flight, stops = self._read_located(located)
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
