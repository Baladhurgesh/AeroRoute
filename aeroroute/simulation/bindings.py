from ..domain.records import EpisodeProvenance
from ..outcomes.draws import SAMPLER_VERSION
from ..outcomes.providers import EmpiricalProvider
from ..outcomes.replay import ExactReplayProvider


SIMULATOR_VERSION = "sequential-flight-v1"


def provenance(schedule, provider, policy):
    if isinstance(provider, EmpiricalProvider):
        if provider.pools.schedule.reference != schedule.reference:
            raise ValueError("Provider and environment schedules differ")
        return EpisodeProvenance(simulator_version=SIMULATOR_VERSION, sampler_version=SAMPLER_VERSION,
            seed=provider.scenario.seed, schedule_ref=schedule.reference,
            outcome_dataset_version=provider.pools.inventory["dataset_version"],
            sampler_parameters=(("mode", "empirical"), ("pool_set", provider.pools.path.name),
                                ("scenario_id", provider.scenario.scenario_id), ("scenario_sha256", provider.scenario.sha256)),
            transition_parameters=policy.parameters())
    if isinstance(provider, ExactReplayProvider):
        if provider.schedule.reference != schedule.reference:
            raise ValueError("Provider and environment schedules differ")
        return EpisodeProvenance(simulator_version=SIMULATOR_VERSION, sampler_version="exact-replay-v1", seed=0,
            schedule_ref=schedule.reference, outcome_dataset_version=provider.manifest["identity"]["inventory"],
            sampler_parameters=(("mode", "replay"), ("replay_id", provider.path.name)), transition_parameters=policy.parameters())
    raise ValueError("Unsupported outcome provider")
