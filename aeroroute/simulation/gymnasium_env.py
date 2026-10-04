from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..domain.records import EpisodeRecord, TerminationReason, TripRequest
from .environment import FlightEnvironment, Observation
from .rubric import reliability_reward


class FlightGymEnv(gym.Env):
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, environment: FlightEnvironment, request: TripRequest, *,
                 max_flights: int = 1000, episode_id: str = "episode"):
        if not isinstance(environment, FlightEnvironment):
            raise ValueError("environment must be a FlightEnvironment")
        self._validate_request(request, episode_id)
        if type(max_flights) is not int or max_flights < 1:
            raise ValueError("max_flights must be a positive integer")
        self.environment = environment
        self.request = request
        self.episode_id = episode_id
        self.max_flights = max_flights
        self.action_space = spaces.Discrete(max_flights + 1)
        self.observation_space = spaces.Dict({
            "flights": spaces.Box(-np.inf, np.inf, shape=(max_flights, 5), dtype=np.float64),
            "state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
            "time_known": spaces.MultiBinary(2),
            "action_mask": spaces.MultiBinary(max_flights + 1),
        })
        self._has_reset = False
        self._finished = False
        self._acknowledgement_pending = False

    @property
    def record(self) -> EpisodeRecord | None:
        return self.environment.record

    @staticmethod
    def _validate_request(request, episode_id):
        if type(request) is not TripRequest:
            raise ValueError("request must be a validated TripRequest")
        if type(episode_id) is not str or not episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")

    def reset(self, *, seed=None, options=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        if options is not None and type(options) is not dict:
            raise ValueError("options must be a dictionary or None")
        options = {} if options is None else options
        if options.keys() - {"request", "episode_id"}:
            raise ValueError("options only supports request and episode_id")
        request = options.get("request", self.request)
        episode_id = options.get("episode_id", self.episode_id)
        self._validate_request(request, episode_id)
        _, observation, info = self._update(
            self.environment.reset, request, episode_id=episode_id, acknowledge_terminal=True,
        )
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self._has_reset = True
        self._finished = False
        self._acknowledgement_pending = self.environment.observation.done
        return observation, info

    def step(self, action):
        if not self._has_reset or self._finished:
            raise ValueError("Reset an episode before taking an action")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer flight slot")
        action = int(action)
        if action < 0 or action > self.max_flights:
            raise ValueError("action is outside the action space")
        if self._acknowledgement_pending:
            if action != self.max_flights:
                raise ValueError("Only the terminal acknowledgement action is available")
            current = self.environment.observation
            observation = self._encode(current)
            info = self._info(current)
            truncated = current.termination_reason in (
                TerminationReason.TIME_LIMIT, TerminationReason.DECISION_LIMIT,
            )
            self._acknowledgement_pending = False
            self._finished = True
            return observation, reliability_reward(self.environment.record), not truncated, truncated, info
        flights = self.environment.observation.flights
        if action == self.max_flights or action >= len(flights):
            raise ValueError("action is not an offered flight slot")
        result, observation, info = self._update(self.environment.step, flights[action].flight_id)
        self._finished = result.terminated or result.truncated
        return observation, result.reward, result.terminated, result.truncated, info

    def _update(self, operation: Callable, *args, acknowledge_terminal=False, **kwargs):
        previous = self.environment.__dict__.copy()
        try:
            result = operation(*args, **kwargs)
            current = self.environment.observation
            observation = self._encode(current, acknowledge_terminal=acknowledge_terminal)
            info = self._info(current)
        except Exception:
            self.environment.__dict__.clear()
            self.environment.__dict__.update(previous)
            raise
        return result, observation, info

    def _encode(self, observation: Observation, *, acknowledge_terminal=False):
        if len(observation.flights) > self.max_flights:
            raise ValueError(
                f"Offered {len(observation.flights)} flights exceeds max_flights={self.max_flights}"
            )
        start = observation.request.start_at

        def minutes(timestamp):
            return 0.0 if timestamp is None else (timestamp - start).total_seconds() / 60.0

        flights = np.zeros((self.max_flights, 5), dtype=np.float64)
        for index, flight in enumerate(observation.flights):
            flights[index] = (
                flight.origin.airport_id,
                flight.destination.airport_id,
                minutes(flight.scheduled_departure_at),
                minutes(flight.scheduled_arrival_at),
                float(flight.destination == observation.request.destination),
            )
        state = np.array((
            0 if observation.current_airport is None else observation.current_airport.airport_id,
            observation.request.destination.airport_id,
            minutes(observation.decision_at),
            minutes(observation.ready_to_board_at),
            minutes(observation.request.arrival_deadline),
            self.environment.settings.max_duration_minutes,
            len(self.environment.steps),
            float(observation.done),
        ), dtype=np.float64)
        mask = np.zeros(self.max_flights + 1, dtype=np.int8)
        if observation.done:
            if acknowledge_terminal:
                mask[self.max_flights] = 1
        else:
            mask[:len(observation.flights)] = 1
        return {
            "flights": flights,
            "state": state,
            "time_known": np.array((observation.decision_at is not None,
                                    observation.ready_to_board_at is not None), dtype=np.int8),
            "action_mask": mask,
        }

    def _info(self, observation: Observation):
        parameters = dict(self.environment.provenance.sampler_parameters)
        return {
            "flight_ids": tuple(flight.flight_id for flight in observation.flights),
            "termination_reason": (None if observation.termination_reason is None
                                   else observation.termination_reason.value),
            "provider_mode": self.environment.mode,
            "scenario_id": parameters.get("scenario_id"),
            "scenario_seed": (self.environment.provenance.seed
                              if self.environment.mode == "empirical" else None),
            "reset_seed_scope": "adapter_rng_and_action_space_only",
            "provider_reseeded": False,
        }

    def close(self):
        pass
