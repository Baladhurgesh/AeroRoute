import platform
import resource
import shutil
import sqlite3
import sys
import time
import uuid
from collections import Counter
from contextlib import closing
from importlib import metadata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..acquisition.archives import read_source_rows
from ..reference.airports import AirportResolver
from ..storage.artifacts import artifact_metadata, verify_partition
from ..storage.identity import digest, file_hash, load_json, write_json
from ..storage.schemas import AIRPORTS_SCHEMA, FLIGHTS_SCHEMA, STOPS_SCHEMA
from .quality import candidate_key_collisions, collect_candidate_keys, collect_quality, new_quality
from .rows import normalize_row


NORMALIZATION_VERSION = 1
TRANSFORM_MODULES = (
    "acquisition/archives.py", "domain/evidence.py", "domain/scalars.py",
    "normalization/parsing.py", "normalization/pipeline.py", "normalization/quality.py",
    "normalization/rows.py", "normalization/times.py", "reference/airports.py",
    "reference/timezones.py", "storage/artifacts.py", "storage/fields.py",
    "storage/identity.py", "storage/schemas.py",
)


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def transform_identity(tolerance_minutes):
    root = Path(__file__).resolve().parents[1]
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "code_sha256": digest({name: file_hash(root / name) for name in TRANSFORM_MODULES}),
        "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in ("pyarrow", "airportsdata", "tzdata")}},
        "timestamp_tolerance_minutes": tolerance_minutes,
        "scheduled_2400_policy": "unresolved_pending_date_convention",
        "timezone_source": "explicit_pinned_tzdata_resources",
        "csv_record_policy": "python_csv_strict_no_blank_skipping_v1",
    }


def source_identity(entry):
    return {key: entry[key] for key in ("year", "month", "path", "sha256", "csv_member", "row_count", "size_bytes")}


def flush_batch(flights, stops, flight_writer, stop_writer, database):
    if not flights:
        return
    flight_writer.write_table(pa.Table.from_pylist(flights, schema=FLIGHTS_SCHEMA))
    if stops:
        stop_writer.write_table(pa.Table.from_pylist(stops, schema=STOPS_SCHEMA))
    collect_candidate_keys(flights, database)
    flights.clear()
    stops.clear()


def normalize_month(raw_dir, entry, output_dir, batch_size=4096, tolerance_minutes=0, overrides=None, verify_only=False):
    start = time.perf_counter()
    if batch_size <= 0 or tolerance_minutes < 0:
        raise ValueError("Batch size must be positive and tolerance nonnegative")
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    source_path = (raw_dir / entry["path"]).resolve()
    if not source_path.is_relative_to(raw_dir) or entry.get("status") != "validated":
        raise ValueError("Input must be a validated archive beneath the raw directory")
    if source_path.stat().st_size != entry["size_bytes"] or file_hash(source_path) != entry["sha256"]:
        raise ValueError("Raw source size/checksum mismatch")
    base = {"source": source_identity(entry), "transform": transform_identity(tolerance_minutes)}
    parent = output_dir / "partitions" / f"year={entry['year']}" / f"month={entry['month']:02d}"
    if parent.exists():
        for path in sorted(parent.iterdir()):
            if path.name.startswith(".") or not (path / "partition_manifest.json").exists():
                continue
            manifest = load_json(path / "partition_manifest.json")
            if manifest.get("base") != base:
                continue
            resolver = AirportResolver(overrides)
            resolver.replay_requests(manifest["airport_requests"])
            if resolver.reference_digest() != manifest["reference_digest"]:
                continue
            verify_partition(path)
            return {"path": path, "reused": True, "elapsed_seconds": time.perf_counter() - start,
                    "manifest": manifest, "quality": load_json(path / "partition_quality.json")}
    if verify_only:
        raise ValueError(f"No compatible completed partition for {entry['year']}-{entry['month']:02d}")
    existing = output_dir
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < 2 * 1024 ** 3:
        raise OSError("Insufficient disk reserve for a pilot partition (2 GiB required)")
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / (".pending-" + uuid.uuid4().hex)
    stage.mkdir()
    resolver = AirportResolver(overrides)
    quality = new_quality(entry)
    flights, stops = [], []
    with closing(sqlite3.connect("")) as database, \
            pq.ParquetWriter(stage / "flights.parquet", FLIGHTS_SCHEMA, compression="zstd") as flight_writer, \
            pq.ParquetWriter(stage / "diversion_stops.parquet", STOPS_SCHEMA, compression="zstd") as stop_writer:
        database.execute("PRAGMA cache_size=-8192")
        database.execute("CREATE TABLE candidate_keys (key BLOB PRIMARY KEY, count INTEGER, first_row INTEGER) WITHOUT ROWID")
        for ordinal, raw in read_source_rows(source_path, entry["csv_member"]):
            row, row_stops = normalize_row(raw, ordinal, entry, resolver, tolerance_minutes)
            collect_quality(quality, row, row_stops)
            flights.append(row)
            stops.extend(row_stops)
            if len(flights) >= batch_size:
                flush_batch(flights, stops, flight_writer, stop_writer, database)
        flush_batch(flights, stops, flight_writer, stop_writer, database)
        quality["candidate_key_collisions"] = candidate_key_collisions(database)
    for actual, expected in (("row_count", "row_count"), ("cancelled_rows", "cancelled_rows"), ("diverted_rows", "diverted_rows")):
        if entry.get(expected) is not None and quality[actual] != entry[expected]:
            raise ValueError(f"Source preservation check failed for {actual}; unpublished staging files retained at {stage}")
    airport_rows = resolver.airport_rows()
    pq.write_table(pa.Table.from_pylist(airport_rows, schema=AIRPORTS_SCHEMA), stage / "airports.parquet", compression="zstd")
    quality["airport_mapping_entries"] = dict(Counter(row["status"] for row in airport_rows))
    quality["unresolved_airports"] = [{key: row[key] for key in ("airport_id", "code", "status", "expected_state", "issues", "occurrences")}
                                      for row in airport_rows if row["status"] != "resolved"]
    quality["normalization_elapsed_seconds"] = time.perf_counter() - start
    quality["process_peak_rss_bytes"] = peak_rss_bytes()
    quality["memory_measurement"] = "process_lifetime_high_water_mark_including_previous_months"
    quality["parquet_bytes"] = sum((stage / name).stat().st_size for name in ("flights.parquet", "diversion_stops.parquet", "airports.parquet"))
    write_json(stage / "partition_quality.json", quality)
    reference_digest = resolver.reference_digest()
    fingerprint = digest({"base": base, "reference_digest": reference_digest})
    manifest = {
        "schema_version": 1, "status": "complete", "fingerprint": fingerprint, "base": base,
        "reference_digest": reference_digest, "airport_requests": resolver.serialized_requests(),
        "artifacts": {name: artifact_metadata(stage / name) for name in
                      ("flights.parquet", "diversion_stops.parquet", "airports.parquet", "partition_quality.json")},
    }
    write_json(stage / "partition_manifest.json", manifest)
    final = parent / fingerprint
    if final.exists():
        raise FileExistsError(f"Partition already exists; refusing to overwrite: {final}")
    stage.rename(final)
    verify_partition(final)
    return {"path": final, "reused": False, "elapsed_seconds": time.perf_counter() - start, "manifest": manifest, "quality": quality}
