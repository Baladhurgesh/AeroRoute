from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..storage.artifacts import artifact_metadata, verify_dataset, verify_partition
from ..storage.identity import digest, file_hash, load_json
from ..storage.fields import BOOLEAN_FIELDS, INTEGER_FIELDS, STRING_FIELDS


def audit_flights(path):
    counts = {"row_count": 0, "cancelled_rows": 0, "diverted_rows": 0,
              "cancelled_nulls": 0, "diverted_nulls": 0, "source_ordinals_contiguous": True}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=["source_row_id", "cancelled", "diverted"]):
        expected = pa.array(range(counts["row_count"] + 1, counts["row_count"] + batch.num_rows + 1), type=pa.int64())
        if not batch.column("source_row_id").equals(expected):
            raise ValueError(f"Source ordinals changed in {path}")
        counts["row_count"] += batch.num_rows
        for name in ("cancelled", "diverted"):
            counts[name + "_rows"] += pc.sum(batch.column(name)).as_py() or 0
            counts[name + "_nulls"] += batch.column(name).null_count
    return counts


def field_rates(category_count, facts):
    result = {}
    for name, metrics in facts.items():
        statuses = metrics["statuses"]
        if sum(statuses.values()) != category_count:
            raise ValueError(f"Resolution counts do not reconcile for {name}")
        applicable = category_count - statuses.get("not_applicable", 0)
        result[name] = {
            "total_rows": category_count, "applicable_rows": applicable,
            "resolved_pct_of_applicable": round(100 * statuses.get("resolved", 0) / applicable, 6) if applicable else None,
            **metrics,
        }
    return result


def build_report(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    verified = verify_dataset(dataset_dir / "dataset_manifest.json")
    root = verified["root"]
    dataset = verified["manifest"]
    if dataset.get("status") != "complete" or digest(dataset["identity"]) != dataset["dataset_version"]:
        raise ValueError("Dataset manifest identity/status mismatch")
    for name, metadata in dataset["artifacts"].items():
        if Path(name).name != name or artifact_metadata(dataset_dir / name) != metadata:
            raise ValueError(f"Dataset artifact mismatch: {name}")
    report = {"report_schema_version": 1, "dataset_version": dataset["dataset_version"],
              "experiment_split": dataset["identity"]["experiment_split"], "months": [], "source_schema_groups": {},
              "scope": "selected_month_pilot_not_full_six_year_normalization", "totals": {},
              "evaluation_caveat": "Newer schedules with older sampled outcomes do not measure actual held-out-year performance.",
              "timing_policy": "Zero-minute cross-check tolerance unless the partition configuration explicitly states otherwise; unknowns are not imputed."}
    totals = Counter()
    expected_fields = set(INTEGER_FIELDS.values()) | set(STRING_FIELDS.values()) | set(BOOLEAN_FIELDS.values())
    for reference in dataset["identity"]["partitions"]:
        partition = (root / reference["path"]).resolve()
        if not partition.is_relative_to(root) or file_hash(partition / "partition_manifest.json") != reference["manifest_sha256"]:
            raise ValueError("Partition reference integrity failure")
        manifest = verify_partition(partition)
        quality = load_json(partition / "partition_quality.json")
        audit = audit_flights(partition / "flights.parquet")
        for field in ("row_count", "cancelled_rows", "diverted_rows"):
            if audit[field] != quality[field] or audit[field] != quality["source_" + field]:
                raise ValueError(f"Raw/normalized/audit counts do not reconcile for {field}")
            totals[field] += audit[field]
        totals["parquet_bytes"] += quality["parquet_bytes"]
        totals["airport_mapping_affected_flight_rows"] += quality["airport_mapping_affected_flight_rows"]
        period = f"{reference['year']}-{reference['month']:02d}"
        headers = quality["source_columns"]
        schema_hash = digest(headers)
        group = report["source_schema_groups"].setdefault(schema_hash, {"periods": [], "columns": headers,
                                                                     "named_column_count": sum(bool(name.strip()) for name in headers)})
        group["periods"].append(period)
        resolutions = {category: field_rates(count, quality["time_resolution"][category])
                       for category, count in quality["outcome_categories"].items()}
        schedule = {}
        for name in ("scheduled_departure", "scheduled_arrival"):
            statuses = Counter()
            issues = Counter()
            for facts in quality["time_resolution"].values():
                statuses.update(facts[name]["statuses"])
                issues.update(facts[name]["issues"])
            schedule[name] = {"statuses": dict(statuses), "issues": dict(issues),
                              "resolved_pct_all_rows": round(100 * statuses.get("resolved", 0) / audit["row_count"], 6)}
        report["months"].append({
            "period": period, "partition": reference["path"], "audit": audit,
            "all_source_records_and_disruptions_preserved": True,
            "schedule_resolution": schedule, "outcome_resolution": resolutions,
            "endpoint_resolution": quality["endpoint_resolution"],
            "diversion_stop_resolution": quality["diversion_stop_resolution"],
            "timezone_mapping": {"affected_flight_rows": quality["airport_mapping_affected_flight_rows"],
                                 "affected_flight_pct": round(100 * quality["airport_mapping_affected_flight_rows"] / audit["row_count"], 6),
                                 "entries": quality["airport_mapping_entries"], "unresolved": quality["unresolved_airports"]},
            "invalid_cells": quality["invalid_cells"], "null_values": quality["null_values"],
            "quality_flags": quality["quality_flags"], "duplicate_flags": quality["duplicate_flags"],
            "candidate_key_collisions": quality["candidate_key_collisions"],
            "missing_selected_source_columns": sorted(expected_fields - {name.strip() for name in headers}),
            "runtime_seconds": quality["normalization_elapsed_seconds"],
            "peak_rss_bytes": quality["process_peak_rss_bytes"], "memory_measurement": quality["memory_measurement"],
            "parquet_bytes": quality["parquet_bytes"], "examples": quality["samples"],
            "transform": manifest["base"]["transform"],
        })
    report["totals"] = dict(totals)
    report["totals"]["normalization_seconds"] = sum(month["runtime_seconds"] for month in report["months"])
    report["totals"]["peak_reported_process_rss_bytes"] = max(month["peak_rss_bytes"] for month in report["months"])
    return report
