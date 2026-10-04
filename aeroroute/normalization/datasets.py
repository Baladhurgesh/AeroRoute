import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..storage.artifacts import artifact_metadata
from ..storage.identity import digest, file_hash, load_json, write_json
from ..storage.schemas import AIRPORTS_SCHEMA


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
