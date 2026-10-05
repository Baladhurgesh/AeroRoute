import sqlite3
from pathlib import Path

from ..domain.records import HistoricalSampleRef, SourceRecordRef
from ..storage.artifacts import publish, staging, verify_derived
from ..storage.identity import digest
from ..storage.lookup import readonly_database
from .draws import SampledEvidence, receipt, validated_flight


def prepare_replay(schedule, output):
    catalog = schedule.catalog
    identity = {"kind": "replay", "schema_version": 1, "builder_version": 1,
                "schedule_sha256": schedule.reference.sha256, "catalog": catalog.path.name,
                "inventory": catalog.store.inventory("evaluation")["dataset_version"]}
    final, stage = staging(output, identity)
    if stage is None:
        return final
    archives = {(row["source_sha256"], row["member"]): dict(row) for row in catalog.store.database.execute("SELECT * FROM archives")}
    version = identity["inventory"]
    with sqlite3.connect(stage / "replay.sqlite") as database:
        database.execute("PRAGMA cache_size=-8192")
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("CREATE TABLE replay (service_id TEXT NOT NULL, source_id TEXT, pool_id TEXT, sha256 TEXT)")
        pending, count = [], 0
        print("Preparing exact replay from the schedule index", flush=True)
        for row in schedule.database.execute(
                "SELECT service_id, source_sha256, source_member, source_ordinal, has_evidence FROM flights"):
            archive = archives[(row["source_sha256"], row["source_member"])]
            if archive["split"] != "evaluation":
                continue
            ordinal = row["source_ordinal"]
            evidence = digest([archive["source_sha256"], archive["member"], ordinal]) if row["has_evidence"] else None
            source = SourceRecordRef(dataset_version=version, source_file=archive["source_file"], source_sha256=archive["source_sha256"],
                                     csv_member=archive["member"], source_row_id=ordinal)
            membership = digest([[source.source_sha256, source.csv_member, source.source_row_id]])
            pool_id = digest({**identity, "service_id": row["service_id"], "source": source.to_dict(), "membership_sha256": membership})
            pending.append((row["service_id"], evidence, pool_id, membership))
            count += 1
            if len(pending) >= 65536:
                database.executemany("INSERT INTO replay VALUES (?, ?, ?, ?)", pending)
                pending.clear()
                if count % 262144 == 0:
                    print(f"Replay mappings prepared: {count}", flush=True)
        if pending:
            database.executemany("INSERT INTO replay VALUES (?, ?, ?, ?)", pending)
        database.execute("CREATE UNIQUE INDEX replay_service ON replay(service_id)")
        print(f"Replay mappings prepared: {count}", flush=True)
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
        source = catalog.store.reference_at(service["source_sha256"], service["source_member"], service["source_ordinal"], "evaluation")
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
