import hashlib
from dataclasses import asdict, dataclass

from ..domain.evidence import FlightEvidence
from ..domain.records import HistoricalSampleRef
from ..storage.identity import canonical_bytes, digest


SAMPLER_VERSION = "sha256-service-rejection-v1"


def uniform_index(key, size):
    if type(size) is not int or not 0 < size <= 2 ** 256:
        raise ValueError("Invalid pool size")
    limit = 2 ** 256 - (2 ** 256 % size)
    counter = 0
    while True:
        value = int.from_bytes(hashlib.sha256(canonical_bytes([SAMPLER_VERSION, key, counter])).digest(), "big")
        if value < limit:
            return value % size
        counter += 1


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    pool_set: str
    schedule_sha256: str
    sampler_version: str = SAMPLER_VERSION

    def __post_init__(self):
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip() or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Invalid scenario key")
        if self.sampler_version != SAMPLER_VERSION:
            raise ValueError("Unsupported sampler version")

    @classmethod
    def create(cls, scenario_id, seed, pools):
        return cls(scenario_id, seed, pools.path.name, pools.schedule.reference.sha256)

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class DrawReceipt:
    draw: tuple[tuple[str, str], ...]
    trace: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SampledEvidence:
    historical_sample: HistoricalSampleRef
    evidence: FlightEvidence
    receipt: DrawReceipt
    supporting_source_ids: tuple[str, ...]


def validated_flight(schedule, flight):
    if flight != schedule.get_flight(flight.flight_id):
        raise ValueError("Selected flight differs from the authorized schedule")


def receipt(parameters, episode_id=None, step_index=None):
    if step_index is not None and (type(step_index) is not int or step_index < 0):
        raise ValueError("Invalid trace step index")
    trace = tuple((key, str(value)) for key, value in (("episode_id", episode_id), ("step_index", step_index)) if value is not None)
    return DrawReceipt(tuple((key, canonical_bytes(value).decode()) for key, value in sorted(parameters.items())), trace)
