import uuid
from pathlib import Path

import pyarrow.parquet as pq

from .identity import digest, file_hash, load_json, write_json
from .schemas import AIRPORTS_SCHEMA, FLIGHTS_SCHEMA, STOPS_SCHEMA


def artifact_metadata(path):
    result = {"sha256": file_hash(path), "size_bytes": path.stat().st_size}
    if path.suffix == ".parquet":
        result["row_count"] = pq.ParquetFile(path).metadata.num_rows
    return result


def verify_partition(path):
    path = Path(path)
    manifest = load_json(path / "partition_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
        raise ValueError(f"Incomplete or unsupported partition: {path}")
    expected = digest({"base": manifest["base"], "reference_digest": manifest["reference_digest"]})
    if expected != manifest["fingerprint"] or path.name != expected:
        raise ValueError(f"Partition identity mismatch: {path}")
    for name, metadata in manifest["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("Artifact names must be local basenames")
        if artifact_metadata(path / name) != metadata:
            raise ValueError(f"Artifact checksum/metadata mismatch: {path / name}")
    if manifest["artifacts"]["flights.parquet"]["row_count"] != manifest["base"]["source"]["row_count"]:
        raise ValueError("Source and normalized row counts differ")
    return manifest


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes root: {relative}")
    return path


def verify_artifacts(root, artifacts, required=()):
    if not set(required) <= set(artifacts):
        raise ValueError("Missing required artifacts")
    for name, expected in artifacts.items():
        path = contained_path(root, name)
        if artifact_metadata(path) != expected:
            raise ValueError(f"Artifact checksum/metadata mismatch: {path}")


def verify_dataset(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_json(manifest_path)
    identity = manifest["identity"]
    if manifest.get("status") != "complete" or identity.get("schema_version") != 1:
        raise ValueError("Incomplete or unsupported dataset")
    version = digest(identity)
    if version != manifest["dataset_version"] or manifest_path.parent.name != version:
        raise ValueError("Normalized snapshot identity mismatch")
    groups = ("all", "historical", "evaluation", "unassigned")
    verify_artifacts(manifest_path.parent, manifest["artifacts"],
                     [*(f"{group}_source_inventory.json" for group in groups), "airports.parquet", "quality_report.json"])
    split = identity["experiment_split"]
    historical, evaluation = split["historical_years"], split["evaluation_years"]
    if (any(type(year) is not int or year < 2018 for year in historical + evaluation)
            or len(set(historical + evaluation)) != len(historical + evaluation)):
        raise ValueError("Invalid or overlapping split years")
    root = manifest_path.parent.parent.parent
    partitions, periods = [], set()
    expected = {group: [] for group in groups}
    for ref in identity["partitions"]:
        period = (ref["year"], ref["month"])
        if period in periods or not 1 <= period[1] <= 12:
            raise ValueError("Duplicate or invalid source period")
        periods.add(period)
        path = contained_path(root, ref["path"])
        partition_path = path / "partition_manifest.json"
        if file_hash(partition_path) != ref["manifest_sha256"]:
            raise ValueError("Partition manifest hash mismatch")
        partition = load_json(partition_path)
        verify_artifacts(path, partition["artifacts"],
                         ("flights.parquet", "diversion_stops.parquet", "airports.parquet", "partition_quality.json"))
        verify_partition(path)
        source = partition["base"]["source"]
        if period != (source["year"], source["month"]) or ref["fingerprint"] != partition["fingerprint"]:
            raise ValueError("Partition source identity mismatch")
        for name, schema in (("flights.parquet", FLIGHTS_SCHEMA), ("diversion_stops.parquet", STOPS_SCHEMA),
                             ("airports.parquet", AIRPORTS_SCHEMA)):
            if not pq.ParquetFile(path / name).schema_arrow.equals(schema):
                raise ValueError(f"Unsupported schema: {name}")
        group = "historical" if period[0] in historical else "evaluation" if period[0] in evaluation else "unassigned"
        expected["all"].append(source)
        expected[group].append(source)
        partitions.append({"path": path, "manifest": partition, "group": group, "source": source})
    if not partitions:
        raise ValueError("Dataset has no partitions")
    inventories = {}
    for group in groups:
        inventory = load_json(manifest_path.parent / f"{group}_source_inventory.json")
        sources = sorted(expected[group], key=lambda item: (item["year"], item["month"]))
        if inventory != {"dataset_version": digest(sources), "sources": sources}:
            raise ValueError(f"Source inventory mismatch: {group}")
        inventories[group] = inventory
    return {"manifest": manifest, "path": manifest_path, "root": root,
            "partitions": partitions, "inventories": inventories}


def verify_derived(path, kind=None):
    path = Path(path)
    manifest = load_json(path / "manifest.json")
    identity = manifest["identity"]
    if (manifest.get("status") != "complete" or identity.get("schema_version") != 1
            or (kind is not None and identity.get("kind") != kind)
            or digest(identity) != path.name):
        raise ValueError("Incomplete or mismatched derived artifact")
    required = {
        "lookup": ("lookup.sqlite",), "services": ("services.sqlite", "quality.json"),
        "schedule": ("schedule.sqlite",), "pools": ("pools.sqlite", "quality.json", "fitting_inventory.json"),
        "replay": ("replay.sqlite",), "pilot": ("pilot_report.json", "scenario.json"),
        "episode": ("episode.json", "verification.json", "experiment.json"),
    }
    artifact_kind = identity.get("kind")
    version_field = "lookup_version" if artifact_kind == "lookup" else "builder_version"
    supported_versions = (1, 2) if artifact_kind in ("lookup", "services") else (1,)
    if artifact_kind not in required or (artifact_kind != "pilot" and identity.get(version_field) not in supported_versions):
        raise ValueError("Unsupported derived artifact version")
    verify_artifacts(path, manifest["artifacts"], required[artifact_kind])
    return manifest


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
