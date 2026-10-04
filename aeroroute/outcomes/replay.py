import sqlite3
from pathlib import Path

from ..domain.records import HistoricalSampleRef
from ..storage.artifacts import publish, staging, verify_derived
from ..storage.identity import digest
from ..storage.lookup import readonly_database
from .draws import SampledEvidence, receipt, validated_flight


def prepare_replay(schedule, output):
    catalog = schedule.catalog
    catalog.require_ready("evaluation")
    identity = {"kind": "replay", "schema_version": 1, "builder_version": 1,
                "schedule_sha256": schedule.reference.sha256, "catalog": catalog.path.name,
                "inventory": catalog.store.inventory("evaluation")["dataset_version"]}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    with sqlite3.connect(stage / "replay.sqlite") as database:
        database.execute("CREATE TABLE replay (service_id TEXT PRIMARY KEY, source_id TEXT, pool_id TEXT, sha256 TEXT)")
        for row in schedule.database.execute("SELECT service_id FROM flights ORDER BY service_id"):
            service = catalog.get(row[0])
            if service["split"] != "evaluation":
                continue
            source = catalog.store.source_ref(service["evidence_source"], "evaluation")
            sha256 = digest([[source.source_sha256, source.csv_member, source.source_row_id]])
            pool_id = digest({**identity, "service_id": row[0], "source": source.to_dict(), "membership_sha256": sha256})
            database.execute("INSERT INTO replay VALUES (?, ?, ?, ?)", (row[0], service["evidence_source"], pool_id, sha256))
        database.commit()
    database.close()
    return publish(stage, final, identity)


class ExactReplayProvider:
    def __init__(self, schedule, path):
        self.schedule, self.path = schedule, Path(path)
        self.manifest = verify_derived(path, "replay")
        if (self.manifest["identity"]["schedule_sha256"] != schedule.reference.sha256
                or self.manifest["identity"]["catalog"] != schedule.catalog.path.name
                or self.manifest["identity"]["inventory"] != schedule.catalog.store.inventory("evaluation")["dataset_version"]):
            raise ValueError("Replay references a different held-out schedule")
        schedule.catalog.require_ready("evaluation")
        self.database = readonly_database(self.path / "replay.sqlite")

    def sample(self, flight, *, episode_id=None, step_index=None):
        validated_flight(self.schedule, flight)
        member = self.database.execute("SELECT * FROM replay WHERE service_id=?", (flight.flight_id,)).fetchone()
        if member is None:
            raise ValueError("Selected departure has no authorized replay mapping")
        catalog = self.schedule.catalog
        service = catalog.get(flight.flight_id)
        if service["split"] != "evaluation" or service["evidence_source"] != member["source_id"]:
            raise ValueError("Replay source differs from resolved service")
        source = catalog.store.source_ref(member["source_id"], "evaluation")
        if digest([[source.source_sha256, source.csv_member, source.source_row_id]]) != member["sha256"]:
            raise ValueError("Replay singleton membership mismatch")
        evidence = catalog.store.get(source)
        reference = HistoricalSampleRef(source=source, flight_date=evidence.flight["flight_date"],
                    matching_pool_id=member["pool_id"], matching_pool_sha256=member["sha256"],
                    matching_pool_size=1, fallback_level="exact_replay")
        return SampledEvidence(reference, evidence, receipt({"mode": "exact_replay", "service_instance_id": flight.flight_id,
            "source_record_id": member["source_id"], "pool_id": member["pool_id"], "membership_sha256": member["sha256"]}, episode_id, step_index),
            catalog.supports(flight.flight_id))

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
