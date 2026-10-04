from dataclasses import replace

from ..domain.records import HistoricalSampleRef
from .draws import SAMPLER_VERSION, SampledEvidence, receipt, uniform_index, validated_flight
from .pools import NoMatchingPool


class EmpiricalProvider:
    def __init__(self, pools, scenario):
        if scenario.pool_set != pools.path.name or scenario.schedule_sha256 != pools.schedule.reference.sha256:
            raise ValueError("Scenario binding differs from prepared pool set")
        self.pools, self.scenario = pools, scenario

    def sample(self, flight, *, episode_id=None, step_index=None):
        validated_flight(self.pools.schedule, flight)
        database = self.pools.database
        target = database.execute("SELECT pool_id FROM targets WHERE service_id=?", (flight.flight_id,)).fetchone()
        if target is None or target[0] is None:
            raise NoMatchingPool("No historical matching pool for scheduled service")
        pool = database.execute("SELECT * FROM pools WHERE pool_id=?", (target[0],)).fetchone()
        key = {"seed": self.scenario.seed, "scenario_id": self.scenario.scenario_id,
               "scenario_sha256": self.scenario.sha256, "service_instance_id": flight.flight_id,
               "pool_id": pool["pool_id"], "membership_sha256": pool["sha256"], "sampler_version": SAMPLER_VERSION}
        ordinal = uniform_index(key, pool["size"])
        member = database.execute("SELECT * FROM members WHERE pool_id=? AND ordinal=?", (pool["pool_id"], ordinal)).fetchone()
        catalog = self.pools.catalog
        historical = catalog.get(member["service_id"])
        if historical["split"] != "historical" or historical["evidence_source"] != member["record_id"]:
            raise ValueError("Pool member differs from resolved fitting service")
        source = replace(catalog.store.source_ref(member["record_id"], "historical"), dataset_version=self.pools.inventory["dataset_version"])
        if (source.source_sha256, source.csv_member, source.source_row_id) != (
                member["source_sha256"], member["source_member"], member["source_ordinal"]):
            raise ValueError("Selected source differs from hashed pool membership")
        evidence = catalog.store.get(source, inventory=self.pools.inventory)
        reference = HistoricalSampleRef(source=source, flight_date=evidence.flight["flight_date"],
                    matching_pool_id=pool["pool_id"], matching_pool_sha256=pool["sha256"],
                    matching_pool_size=pool["size"], fallback_level=pool["level"])
        return SampledEvidence(reference, evidence, receipt({**key, "ordinal": ordinal, "source_record_id": member["record_id"]}, episode_id, step_index),
                               catalog.supports(member["service_id"]))
