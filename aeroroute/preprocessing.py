import argparse
import csv
import io
import platform
import resource
import shutil
import sqlite3
import sys
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date
from importlib import metadata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .airport_timezones import AirportResolver, pinned_zone
from .data_snapshots import artifact_metadata, digest, file_hash, load_json, verify_partition, write_json
from .flight_times import TimeFact, arrival_time, departure_time, diversion_sequence, schedule_times, unknown
from .preprocessing_schema import (
    AIRPORTS_SCHEMA, BOOLEAN_FIELDS, FLIGHTS_SCHEMA, INTEGER_FIELDS, STOPS_SCHEMA,
    STRING_FIELDS, TIME_FIELDS, parse_cells, parse_integer,
)


ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_VERSION = 1


def read_source_rows(path, member):
    with zipfile.ZipFile(path) as archive, archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
        reader = csv.reader(text, strict=True)
        original = next(reader, [])
        headers = [column.strip() for column in original]
        named = [column for column in headers if column]
        if not named or len(named) != len(set(named)):
            raise ValueError("Missing or duplicate normalized CSV headers")
        for ordinal, cells in enumerate(reader, 1):
            if len(cells) != len(headers):
                raise ValueError(f"CSV data record {ordinal} has {len(cells)} fields; expected {len(headers)}")
            if any(value.strip() for field, value in zip(headers, cells) if not field):
                raise ValueError(f"Nonempty unnamed CSV field at record {ordinal}")
            yield ordinal, {field: value for field, value in zip(headers, cells) if field}


def outcome_category(row):
    cancelled, diverted = row["cancelled"], row["diverted"]
    if cancelled is None or diverted is None or (cancelled and diverted):
        return "unknown"
    if cancelled:
        return "cancelled"
    if not diverted:
        return "ordinary"
    reached = row["diversion_reached_destination"]
    return "diverted_reached" if reached is True else "diverted_stranded" if reached is False else "diverted_unknown"


def mapping_zone(mapping):
    return pinned_zone(mapping["time_zone"]) if mapping["time_zone"] else None


def normalize_row(raw, ordinal, source, resolver, tolerance_minutes):
    row = parse_cells(raw)
    row.update(source_row_id=ordinal, source_sha256=source["sha256"], source_file=source["path"],
               source_member=source["csv_member"],
               source_record_id=digest([source["sha256"], source["csv_member"], ordinal]))
    category = outcome_category(row)
    row["outcome_category"] = category
    flags = []
    origin = resolver.resolve(row["origin_airport_id"], row["origin"], row["flight_date"], row["origin_state_name"])
    destination = resolver.resolve(row["destination_airport_id"], row["destination"], row["flight_date"], row["destination_state_name"])
    for role, mapping in (("origin", origin), ("destination", destination)):
        row[f"{role}_time_zone"] = mapping["time_zone"]
        if mapping["status"] != "resolved":
            flags.append(f"{role}_mapping_{mapping['status']}")
    origin_zone, destination_zone = mapping_zone(origin), mapping_zone(destination)
    scheduled_departure, scheduled_arrival = schedule_times(
        row["flight_date"], row["scheduled_departure_clock"], row["scheduled_arrival_clock"],
        row["scheduled_elapsed_minutes"], origin_zone, destination_zone, tolerance_minutes)
    actual_departure = departure_time(scheduled_departure, row["departure_delay_minutes"], row["actual_departure_clock"],
                                     origin_zone, category == "cancelled", tolerance_minutes)
    if category == "cancelled":
        actual_arrival = TimeFact(status="not_applicable", method="cancelled_no_arrival")
    elif category in ("ordinary", "diverted_reached"):
        elapsed = row["actual_elapsed_minutes"] if category == "ordinary" else row["diversion_elapsed_minutes"]
        delay = row["arrival_delay_minutes"] if category == "ordinary" else row["diversion_arrival_delay_minutes"]
        actual_arrival = arrival_time(actual_departure, scheduled_arrival, elapsed, delay,
                                      row["actual_arrival_clock"], destination_zone, tolerance_minutes)
    else:
        actual_arrival = unknown("stranded_gate_arrival_not_reported" if category == "diverted_stranded" else "outcome_unresolved")
    facts = (scheduled_departure, scheduled_arrival, actual_departure, actual_arrival)
    row.update({name: fact.as_dict() for name, fact in zip(TIME_FIELDS, facts)})
    stops, clocks = [], []
    for index in range(1, 6):
        prefix = f"Div{index}"
        fields = ("Airport", "AirportID", "AirportSeqID", "WheelsOn", "WheelsOff", "TotalGTime", "LongestGTime")
        values = {field: (raw.get(prefix + field) or "").strip() or None for field in fields}
        if not any(value is not None for value in values.values()):
            continue
        airport_id = parse_integer(values["AirportID"], prefix + "AirportID", row["cell_issues"])
        mapping = resolver.resolve(airport_id, values["Airport"], row["flight_date"])
        if mapping["status"] != "resolved":
            flags.append("diversion_mapping_unresolved")
        stops.append({
            "source_record_id": row["source_record_id"], "source_row_id": ordinal, "stop_index": index,
            "airport_id": airport_id, "airport": values["Airport"],
            "airport_seq_id": parse_integer(values["AirportSeqID"], prefix + "AirportSeqID", row["cell_issues"]),
            "landing_clock": values["WheelsOn"], "departure_clock": values["WheelsOff"],
            "total_ground_minutes": parse_integer(values["TotalGTime"], prefix + "TotalGTime", row["cell_issues"]),
            "longest_ground_minutes": parse_integer(values["LongestGTime"], prefix + "LongestGTime", row["cell_issues"]),
            "time_zone": mapping["time_zone"], "mapping_status": mapping["status"],
        })
        clocks.append((values["WheelsOn"], values["WheelsOff"], mapping_zone(mapping)))
    for stop, (landing, departure) in zip(stops, diversion_sequence(clocks, actual_departure.value, actual_arrival.value)):
        stop.update(landing=landing.as_dict(), departure=departure.as_dict())
    if stops and not row["diverted"]:
        flags.append("diversion_stops_without_diversion_flag")
    count = row["diversion_landings"]
    complete_stops = (count is not None and 1 <= count <= 5 and [stop["stop_index"] for stop in stops] == list(range(1, count + 1))
                      and all(stop["airport_id"] is not None and stop["airport"] for stop in stops))
    if row["diverted"] and not complete_stops:
        flags.append("incomplete_diversion_stop_sequence")
    if category in ("ordinary", "diverted_reached"):
        endpoint_id, endpoint = row["destination_airport_id"], row["destination"]
        endpoint_method = "reported_scheduled_destination"
    elif category == "cancelled":
        endpoint_id, endpoint = row["origin_airport_id"], row["origin"]
        endpoint_method = "cancelled_at_origin"
    elif category == "diverted_stranded" and complete_stops:
        endpoint_id, endpoint = stops[-1]["airport_id"], stops[-1]["airport"]
        endpoint_method = "reported_final_diversion_stop"
    else:
        endpoint_id, endpoint, endpoint_method = None, None, "unresolved"
    row.update(endpoint_airport_id=endpoint_id, endpoint=endpoint,
               endpoint_status=endpoint_method if endpoint_id is not None and endpoint else "unresolved",
               endpoint_issues=[] if endpoint_id is not None and endpoint else ["endpoint_not_supported"],
               quality_flags=sorted(set(flags)))
    return row, stops


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def transform_identity(tolerance_minutes):
    modules = ("preprocessing.py", "preprocessing_schema.py", "flight_times.py", "airport_timezones.py", "data_snapshots.py")
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "code_sha256": digest({name: file_hash(Path(__file__).with_name(name)) for name in modules}),
        "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in ("pyarrow", "airportsdata", "tzdata")}},
        "timestamp_tolerance_minutes": tolerance_minutes,
        "scheduled_2400_policy": "unresolved_pending_date_convention",
        "timezone_source": "explicit_pinned_tzdata_resources",
        "csv_record_policy": "python_csv_strict_no_blank_skipping_v1",
    }


def source_identity(entry):
    return {key: entry[key] for key in ("year", "month", "path", "sha256", "csv_member", "row_count", "size_bytes")}


def fact_summary(fact):
    return {key: ([value.isoformat() for value in item] if key == "evidence" else item.isoformat() if key == "value" and item else item)
            for key, item in fact.items()}


def collect_quality(report, row, stops):
    category = row["outcome_category"]
    report["row_count"] += 1
    report["cancelled_rows"] += row["cancelled"] is True
    report["diverted_rows"] += row["diverted"] is True
    report["outcome_categories"][category] += 1
    report["duplicate_flags"][row["duplicate_flag"] or "missing"] += 1
    report["endpoint_resolution"][row["endpoint_status"]] += 1
    report["quality_flags"].update(row["quality_flags"])
    if any("mapping" in flag for flag in row["quality_flags"]):
        report["airport_mapping_affected_flight_rows"] += 1
    for issue in row["cell_issues"]:
        report["invalid_cells"][f"{issue['field']}:{issue['code']}"] += 1
    for name in (*INTEGER_FIELDS, *STRING_FIELDS, *BOOLEAN_FIELDS):
        if row[name] is None:
            report["null_values"][name] += 1
    for name in TIME_FIELDS:
        fact = row[name]
        counters = report["time_resolution"][category][name]
        counters["statuses"][fact["status"]] += 1
        counters["methods"][fact["method"]] += 1
        counters["issues"].update(fact["issues"])
        counters["checks"].update(fact["checks"])
    for stop in stops:
        for name in ("landing", "departure"):
            report["diversion_stop_resolution"][name]["statuses"][stop[name]["status"]] += 1
            report["diversion_stop_resolution"][name]["issues"].update(stop[name]["issues"])
    sample_categories = []
    if row["diverted"]:
        sample_categories.append(category)
    if len(stops) > 1:
        sample_categories.append("multi_stop")
    if any(row[name]["status"] == "contradiction" for name in TIME_FIELDS):
        sample_categories.append("timestamp_contradiction")
    for sample_category in sample_categories:
        samples = report["samples"][sample_category]
        if len(samples) < 5:
            samples.append({
                "source_record_id": row["source_record_id"], "source_row_id": row["source_row_id"],
                "flight_date": row["flight_date"].isoformat() if row["flight_date"] else None,
                "origin": row["origin"], "destination": row["destination"], "endpoint": row["endpoint"],
                "endpoint_status": row["endpoint_status"], "stop_count": len(stops),
                "reported_diversion_landings": row["diversion_landings"],
                "facts": {name: fact_summary(row[name]) for name in TIME_FIELDS},
                "stops": [{"index": stop["stop_index"], "airport": stop["airport"],
                           "landing_clock": stop["landing_clock"], "departure_clock": stop["departure_clock"],
                           "landing": fact_summary(stop["landing"]), "departure": fact_summary(stop["departure"])} for stop in stops],
            })


def new_quality(entry):
    return {
        "year": entry["year"], "month": entry["month"], "source_row_count": entry["row_count"],
        "source_cancelled_rows": entry.get("cancelled_rows"), "source_diverted_rows": entry.get("diverted_rows"),
        "row_count": 0, "cancelled_rows": 0, "diverted_rows": 0, "airport_mapping_affected_flight_rows": 0,
        "outcome_categories": Counter(), "duplicate_flags": Counter(), "endpoint_resolution": Counter(),
        "quality_flags": Counter(), "invalid_cells": Counter(), "null_values": Counter(),
        "time_resolution": defaultdict(lambda: defaultdict(lambda: {key: Counter() for key in ("statuses", "methods", "issues", "checks")})),
        "diversion_stop_resolution": defaultdict(lambda: {"statuses": Counter(), "issues": Counter()}),
        "samples": defaultdict(list), "source_columns": entry.get("columns", []),
    }


def flush_batch(flights, stops, flight_writer, stop_writer, database):
    if not flights:
        return
    flight_writer.write_table(pa.Table.from_pylist(flights, schema=FLIGHTS_SCHEMA))
    if stops:
        stop_writer.write_table(pa.Table.from_pylist(stops, schema=STOPS_SCHEMA))
    keys = []
    for row in flights:
        values = [row["flight_date"].isoformat() if row["flight_date"] else None, row["operating_carrier"],
                  row["operating_flight_number"], row["origin_airport_id"], row["destination_airport_id"], row["scheduled_departure_clock"]]
        if all(value is not None for value in values):
            keys.append((bytes.fromhex(digest(values)), row["source_row_id"]))
    database.executemany("INSERT INTO candidate_keys VALUES (?, 1, ?) ON CONFLICT(key) DO UPDATE SET count = count + 1", keys)
    database.commit()
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
        quality["candidate_key_collisions"] = {
            "groups": database.execute("SELECT count(*) FROM candidate_keys WHERE count > 1").fetchone()[0],
            "additional_rows": database.execute("SELECT coalesce(sum(count-1),0) FROM candidate_keys WHERE count > 1").fetchone()[0],
            "first_source_row_examples": [row[0] for row in database.execute("SELECT first_row FROM candidate_keys WHERE count > 1 ORDER BY first_row LIMIT 10")],
            "policy": "diagnostics_only_no_deduplication",
        }
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


def split_configuration(historical_years, evaluation_years):
    historical, evaluation = sorted(set(historical_years)), sorted(set(evaluation_years))
    if set(historical) & set(evaluation) or any(year < 2018 for year in historical + evaluation):
        raise ValueError("Experiment years must be disjoint and within Marketing Carrier coverage")
    return {"historical_years": historical, "evaluation_years": evaluation, "membership_location": "manifest_only"}


def publish_dataset(results, output_dir, historical_years=tuple(range(2020, 2025)), evaluation_years=(2025,)):
    output_dir = Path(output_dir).resolve()
    references = [{"year": result["manifest"]["base"]["source"]["year"], "month": result["manifest"]["base"]["source"]["month"],
                   "path": result["path"].relative_to(output_dir).as_posix(), "fingerprint": result["manifest"]["fingerprint"],
                   "manifest_sha256": file_hash(result["path"] / "partition_manifest.json")} for result in results]
    references.sort(key=lambda item: (item["year"], item["month"]))
    configuration = split_configuration(historical_years, evaluation_years)
    identity = {"schema_version": 1, "partitions": references, "experiment_split": configuration}
    version = digest(identity)
    destination = output_dir / "datasets" / version
    if destination.exists():
        existing = load_json(destination / "dataset_manifest.json")
        if existing["identity"] != identity:
            raise ValueError("Existing dataset identity mismatch")
        for name, expected in existing["artifacts"].items():
            if artifact_metadata(destination / name) != expected:
                raise ValueError(f"Dataset artifact mismatch: {name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / (".pending-" + uuid.uuid4().hex)
    stage.mkdir()
    inventories = {"all": [], "historical": [], "evaluation": [], "unassigned": []}
    for result in results:
        source = result["manifest"]["base"]["source"]
        inventories["all"].append(source)
        group = "historical" if source["year"] in configuration["historical_years"] else "evaluation" if source["year"] in configuration["evaluation_years"] else "unassigned"
        inventories[group].append(source)
    for group, sources in inventories.items():
        sources.sort(key=lambda source: (source["year"], source["month"]))
        write_json(stage / f"{group}_source_inventory.json", {"dataset_version": digest(sources), "sources": sources})
    airport_rows = []
    for result in results:
        airport_rows.extend(pq.ParquetFile(result["path"] / "airports.parquet").read().to_pylist())
    pq.write_table(pa.Table.from_pylist(airport_rows, schema=AIRPORTS_SCHEMA), stage / "airports.parquet", compression="zstd")
    quality = {"scope": "explicit_month_pilot", "months": [result["quality"] for result in results],
               "row_count": sum(result["quality"]["row_count"] for result in results),
               "cancelled_rows": sum(result["quality"]["cancelled_rows"] for result in results),
               "diverted_rows": sum(result["quality"]["diverted_rows"] for result in results),
               "parquet_bytes": sum(result["quality"]["parquet_bytes"] for result in results),
               "evaluation_note": "2025 schedules with historical sampled outcomes are not actual 2025 performance evaluation."}
    write_json(stage / "quality_report.json", quality)
    artifacts = {path.name: artifact_metadata(path) for path in stage.iterdir()}
    write_json(stage / "dataset_manifest.json", {"status": "complete", "dataset_version": version, "identity": identity, "artifacts": artifacts})
    stage.rename(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description="Normalize explicitly selected BTS months without filtering source records.")
    parser.add_argument("--periods", nargs="+", required=True, help="Explicit source months, e.g. 2020-01 2020-03 2025-01")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/bts_marketing")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--tolerance-minutes", type=int, default=0)
    parser.add_argument("--airport-overrides", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--historical-years", nargs="*", type=int, default=list(range(2020, 2025)))
    parser.add_argument("--evaluation-years", nargs="*", type=int, default=[2025])
    args = parser.parse_args(argv)
    split_configuration(args.historical_years, args.evaluation_years)
    manifest = load_json(args.raw_dir / "manifest.json")
    periods = sorted(set(args.periods))
    for period in periods:
        day = date.fromisoformat(period + "-01")
        if day.year < 2018 or period != day.strftime("%Y-%m") or period not in manifest["files"]:
            raise ValueError(f"Requested month is unavailable or invalid: {period}")
    results = []
    output_dir = args.output_dir.resolve()
    for period in periods:
        print(f"{period}: {'verifying' if args.verify_only else 'normalizing/checking reusable partition'}", flush=True)
        result = normalize_month(args.raw_dir, manifest["files"][period], output_dir, args.batch_size,
                                 args.tolerance_minutes, args.airport_overrides, args.verify_only)
        results.append(result)
        quality = result["quality"]
        print(f"  {'reused' if result['reused'] else 'normalized'} {quality['row_count']:,} records; "
              f"{quality['parquet_bytes']:,} Parquet bytes; operation {result['elapsed_seconds']:.2f}s", flush=True)
        print(f"  cancelled={quality['cancelled_rows']:,}; diverted={quality['diverted_rows']:,}; "
              f"timezone-affected rows={quality['airport_mapping_affected_flight_rows']:,}", flush=True)
        print(f"  {result['path']}", flush=True)
    if not args.verify_only:
        path = publish_dataset(results, output_dir, args.historical_years, args.evaluation_years)
        print(f"Dataset quality report: {path / 'quality_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, csv.Error, zipfile.BadZipFile) as error:
        print(f"Preprocessing failed: {error}", file=sys.stderr)
        sys.exit(1)
