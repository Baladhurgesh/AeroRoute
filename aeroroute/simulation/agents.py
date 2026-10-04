from collections.abc import Callable, Mapping
from dataclasses import dataclass
import random

import numpy as np

from ..domain.records import EpisodeRecord


EncodedObservation = Mapping[str, np.ndarray]
Agent = Callable[[EncodedObservation], int]


def _available_actions(observation: EncodedObservation) -> tuple[np.ndarray, list[int]]:
    if not isinstance(observation, Mapping) or not {"flights", "action_mask"} <= observation.keys():
        raise ValueError("Expected an encoded observation with flights and action_mask")
    flights = np.asarray(observation["flights"])
    mask = np.asarray(observation["action_mask"])
    if flights.ndim != 2 or flights.shape[1] != 5 or not np.issubdtype(flights.dtype, np.number):
        raise ValueError("flights must be a numeric array with shape (max_flights, 5)")
    if np.iscomplexobj(flights):
        raise ValueError("flights must contain real numbers")
    if mask.shape != (len(flights) + 1,) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("action_mask must be binary with shape (max_flights + 1,)")
    actions = np.flatnonzero(mask).tolist()
    if not actions:
        raise ValueError("Observation has no valid masked action")
    if mask[-1]:
        if len(actions) != 1:
            raise ValueError("Terminal acknowledgement cannot be offered alongside flight actions")
    else:
        offered = flights[actions]
        if not np.all(np.isfinite(offered)) or not np.all((offered[:, 4] == 0) | (offered[:, 4] == 1)):
            raise ValueError("Offered flights must be finite with binary is_trip_destination values")
    return flights, actions


class RandomAgent:
    def __init__(self, seed: int | None = 0):
        if seed is not None and type(seed) is not int:
            raise ValueError("Agent seed must be an integer or None")
        self._random = random.Random(seed)

    def __call__(self, observation: EncodedObservation) -> int:
        _, actions = _available_actions(observation)
        return self._random.choice(actions)


class EarliestArrivalAgent:
    def __call__(self, observation: EncodedObservation) -> int:
        flights, actions = _available_actions(observation)
        if actions == [len(flights)]:
            return actions[0]
        return min(actions, key=lambda index: (
            -flights[index, 4], flights[index, 3], flights[index, 2], index,
        ))


def make_agent(name: str, *, seed: int | None = 0) -> Agent:
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "earliest-arrival":
        return EarliestArrivalAgent()
    raise ValueError(f"Unknown agent {name!r}; expected 'random' or 'earliest-arrival'")


@dataclass(frozen=True)
class RolloutResult:
    record: EpisodeRecord
    total_reward: float
    terminated: bool
    truncated: bool


def run_episode(env, agent: Agent, *, seed=None, options=None) -> RolloutResult:
    observation, _ = env.reset(seed=seed, options=options)
    total_reward = 0.0
    while True:
        action = agent(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            record = env.record
            if not isinstance(record, EpisodeRecord):
                raise ValueError("Finished environment must expose an EpisodeRecord as env.record")
            return RolloutResult(record, total_reward, bool(terminated), bool(truncated))
