import sqlite3
from contextlib import closing
from pathlib import Path

import pyarrow.parquet as pq

from .artifacts import publish, staging, verify_dataset
from .identity import digest


FLIGHT_INDEX_COLUMNS = ("source_record_id", "source_sha256", "source_member", "source_file", "source_row_id")
STOP_INDEX_COLUMNS = ("source_record_id", "source_row_id", "stop_index")


def readonly_database(path):
    database = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA cache_size=-8192")
    return database


def index_plan(ordinals):
    if not ordinals:
        return "empty", None, None
    first = ordinals[0]
    if type(first) is int and all(value == first + index for index, value in enumerate(ordinals)):
        return "range", first, first + len(ordinals) - 1
    return "explicit", None, None


def _identities(table, source):
    ordinals = table.column("source_row_id").to_pylist()
    records = table.column("source_record_id").to_pylist()
    hashes = table.column("source_sha256").to_pylist()
    members = table.column("source_member").to_pylist()
    files = table.column("source_file").to_pylist()
    expected = (source["sha256"], source["csv_member"], source["path"])
    for index, ordinal in enumerate(ordinals):
        if (type(ordinal) is not int or not 1 <= ordinal <= source["row_count"]
                or (hashes[index], members[index], files[index]) != expected
                or records[index] != digest([source["sha256"], source["csv_member"], ordinal])):
            raise ValueError("Invalid source ordinal or record identity")
    return ordinals


def build_lookup(snapshot_path, output):
    snapshot = verify_dataset(snapshot_path)
    identity = {"kind": "lookup", "schema_version": 1, "lookup_version": 3,
                "snapshot": snapshot["manifest"]["dataset_version"], "locator": "row-group-range-v1"}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    with closing(sqlite3.connect(stage / "lookup.sqlite")) as database, database:
        database.execute("PRAGMA cache_size=-8192")
        database.executescript("""
            CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE, sha256 TEXT NOT NULL);
            CREATE TABLE archives (id INTEGER PRIMARY KEY, source_sha256 TEXT, member TEXT,
                source_file TEXT, split TEXT, year INTEGER, month INTEGER, UNIQUE(source_sha256, member));
            CREATE TABLE ranges (archive_id INTEGER NOT NULL, file_id INTEGER NOT NULL, row_group INTEGER NOT NULL,
                first_ordinal INTEGER NOT NULL, last_ordinal INTEGER NOT NULL, PRIMARY KEY (archive_id, first_ordinal));
            CREATE TABLE explicit_rows (archive_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, file_id INTEGER NOT NULL,
                row_group INTEGER NOT NULL, row_offset INTEGER NOT NULL, PRIMARY KEY (archive_id, ordinal)) WITHOUT ROWID;
            CREATE TABLE stop_counts (archive_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, stop_count INTEGER NOT NULL,
                PRIMARY KEY (archive_id, ordinal)) WITHOUT ROWID;
            CREATE TABLE stops (archive_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, stop_index INTEGER NOT NULL,
                file_id INTEGER NOT NULL, row_group INTEGER NOT NULL, row_offset INTEGER NOT NULL,
                PRIMARY KEY (archive_id, ordinal, stop_index)) WITHOUT ROWID;
        """)
        for partition in snapshot["partitions"]:
            source, manifest, path = partition["source"], partition["manifest"], partition["path"]
            archive_id = database.execute(
                "INSERT INTO archives(source_sha256, member, source_file, split, year, month) VALUES (?, ?, ?, ?, ?, ?)",
                (source["sha256"], source["csv_member"], source["path"], partition["group"], source["year"], source["month"])).lastrowid
            flight_id = database.execute("INSERT INTO files(path, sha256) VALUES (?, ?)",
                ((path / "flights.parquet").relative_to(snapshot["root"]).as_posix(), manifest["artifacts"]["flights.parquet"]["sha256"])).lastrowid
            stop_file_id = database.execute("INSERT INTO files(path, sha256) VALUES (?, ?)",
                ((path / "diversion_stops.parquet").relative_to(snapshot["root"]).as_posix(),
                 manifest["artifacts"]["diversion_stops.parquet"]["sha256"])).lastrowid
            covered = 0
            with pq.ParquetFile(path / "flights.parquet") as parquet:
                for group in range(parquet.num_row_groups):
                    table = parquet.read_row_group(group, columns=list(FLIGHT_INDEX_COLUMNS))
                    ordinals = _identities(table, source)
                    kind, first, last = index_plan(ordinals)
                    if kind == "empty":
                        continue
                    if kind == "range":
                        if first <= covered:
                            raise ValueError("Source ordinal ranges overlap")
                        database.execute("INSERT INTO ranges VALUES (?, ?, ?, ?, ?)", (archive_id, flight_id, group, first, last))
                        covered = last
                    else:
                        rows = []
                        for offset, ordinal in enumerate(ordinals):
                            if ordinal <= covered:
                                raise ValueError("Source ordinal ranges overlap")
                            rows.append((archive_id, ordinal, flight_id, group, offset))
                            covered = max(covered, ordinal)
                        database.executemany("INSERT INTO explicit_rows VALUES (?, ?, ?, ?, ?)", rows)
                    del table
            with pq.ParquetFile(path / "diversion_stops.parquet") as parquet:
                for group in range(parquet.num_row_groups):
                    table = parquet.read_row_group(group, columns=list(STOP_INDEX_COLUMNS))
                    ordinals = table.column("source_row_id").to_pylist()
                    records = table.column("source_record_id").to_pylist()
                    indexes = table.column("stop_index").to_pylist()
                    for offset, ordinal in enumerate(ordinals):
                        if type(ordinal) is not int or not 1 <= indexes[offset] <= 5:
                            raise ValueError("Invalid diversion stop index")
                        if records[offset] != digest([source["sha256"], source["csv_member"], ordinal]):
                            raise ValueError("Invalid source ordinal or record identity")
                        located = database.execute(
                            "SELECT 1 FROM ranges WHERE archive_id=? AND first_ordinal<=? AND last_ordinal>=? "
                            "UNION ALL SELECT 1 FROM explicit_rows WHERE archive_id=? AND ordinal=? LIMIT 1",
                            (archive_id, ordinal, ordinal, archive_id, ordinal)).fetchone()
                        if located is None:
                            raise ValueError("Orphan diversion stop")
                        database.execute(
                            "INSERT INTO stop_counts VALUES (?, ?, 1) ON CONFLICT(archive_id, ordinal) DO UPDATE SET stop_count=stop_count+1",
                            (archive_id, ordinal))
                        database.execute("INSERT INTO stops VALUES (?, ?, ?, ?, ?, ?)",
                                         (archive_id, ordinal, indexes[offset], stop_file_id, group, offset))
                    del table
        mismatch = database.execute("""
            SELECT stop_count, (SELECT count(*) FROM stops s WHERE s.archive_id=c.archive_id AND s.ordinal=c.ordinal)
            FROM stop_counts c
            WHERE stop_count != (SELECT count(*) FROM stops s WHERE s.archive_id=c.archive_id AND s.ordinal=c.ordinal)
            LIMIT 1
        """).fetchone()
        if mismatch is not None:
            raise ValueError("Missing indexed diversion stops")
        database.commit()
    return publish(stage, final, identity)
