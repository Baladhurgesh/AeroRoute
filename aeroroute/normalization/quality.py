from collections import Counter, defaultdict

from ..storage.fields import BOOLEAN_FIELDS, INTEGER_FIELDS, STRING_FIELDS, TIME_FIELDS
from ..storage.identity import digest


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


def collect_candidate_keys(flights, database):
    keys = []
    for row in flights:
        values = [row["flight_date"].isoformat() if row["flight_date"] else None, row["operating_carrier"],
                  row["operating_flight_number"], row["origin_airport_id"], row["destination_airport_id"], row["scheduled_departure_clock"]]
        if all(value is not None for value in values):
            keys.append((bytes.fromhex(digest(values)), row["source_row_id"]))
    database.executemany("INSERT INTO candidate_keys VALUES (?, 1, ?) ON CONFLICT(key) DO UPDATE SET count = count + 1", keys)
    database.commit()


def candidate_key_collisions(database):
    return {
        "groups": database.execute("SELECT count(*) FROM candidate_keys WHERE count > 1").fetchone()[0],
        "additional_rows": database.execute("SELECT coalesce(sum(count-1),0) FROM candidate_keys WHERE count > 1").fetchone()[0],
        "first_source_row_examples": [row[0] for row in database.execute("SELECT first_row FROM candidate_keys WHERE count > 1 ORDER BY first_row LIMIT 10")],
        "policy": "diagnostics_only_no_deduplication",
    }
