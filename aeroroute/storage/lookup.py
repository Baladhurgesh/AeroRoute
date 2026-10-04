import sqlite3
from contextlib import closing
from pathlib import Path

import pyarrow.parquet as pq

from .artifacts import publish, staging, verify_dataset
from .identity import digest


def readonly_database(path):
    database = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA cache_size=-8192")
    return database


def build_lookup(snapshot_path, output):
    snapshot = verify_dataset(snapshot_path)
    identity = {"kind": "lookup", "schema_version": 1, "lookup_version": 2,
                "snapshot": snapshot["manifest"]["dataset_version"]}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    with closing(sqlite3.connect(stage / "lookup.sqlite")) as database, database:
        database.execute("PRAGMA cache_size=-8192")
        database.executescript("""
            CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE, sha256 TEXT NOT NULL);
            CREATE TABLE archives (id INTEGER PRIMARY KEY, source_sha256 TEXT, member TEXT,
                source_file TEXT, split TEXT, year INTEGER, month INTEGER, UNIQUE(source_sha256, member));
            CREATE TABLE source_rows (record_id BLOB PRIMARY KEY, archive_id INTEGER, ordinal INTEGER,
                file_id INTEGER, row_group INTEGER, row_offset INTEGER, stop_count INTEGER DEFAULT 0,
                UNIQUE(archive_id, ordinal)) WITHOUT ROWID;
            CREATE VIEW sources AS SELECT r.*, a.source_sha256, a.member, a.source_file, a.split, a.year, a.month
                FROM source_rows r JOIN archives a ON a.id=r.archive_id;
            CREATE TABLE stops (record_id BLOB, stop_index INTEGER, file_id INTEGER,
                row_group INTEGER, row_offset INTEGER, PRIMARY KEY(record_id, stop_index)) WITHOUT ROWID;
        """)
        for partition in snapshot["partitions"]:
            source, manifest, path = partition["source"], partition["manifest"], partition["path"]
            archive_id = database.execute("INSERT INTO archives(source_sha256, member, source_file, split, year, month) VALUES (?, ?, ?, ?, ?, ?)",
                (source["sha256"], source["csv_member"], source["path"], partition["group"], source["year"], source["month"])).lastrowid
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
                                database.execute("INSERT INTO source_rows VALUES (?, ?, ?, ?, ?, ?, 0)",
                                    (bytes.fromhex(record_id), archive_id, ordinal, file_id, group, offset))
                            else:
                                if not 1 <= row["stop_index"] <= 5:
                                    raise ValueError("Invalid diversion stop index")
                                changed = database.execute("UPDATE source_rows SET stop_count=stop_count+1 WHERE record_id=?", (bytes.fromhex(record_id),))
                                if changed.rowcount != 1:
                                    raise ValueError("Orphan diversion stop")
                                database.execute("INSERT INTO stops VALUES (?, ?, ?, ?, ?)",
                                                 (bytes.fromhex(record_id), row["stop_index"], file_id, group, offset))
        database.commit()
    return publish(stage, final, identity)
