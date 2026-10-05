# AeroRoute

AeroRoute is a sequential flight-choice environment built from U.S. Bureau of Transportation Statistics (BTS) marketing-carrier on-time records, January 2020 through December 2025. The traveler starts at an origin airport. At each step your policy picks the next bookable flight. A historical outcome then advances the trip: the flight arrives, is cancelled, is diverted, or is missed because it departed before the traveler was ready. The episode ends when the traveler reaches the destination, runs out of time, or hits the decision limit.

The reward is a reliability score from 0 to 1, paid once at the end of the episode.

## Install

```bash
uv sync --extra rl
```

Python 3.11 or newer. The `rl` extra installs Gymnasium and NumPy, which the training interface needs. Core data commands work without it.

Raw and processed data stay under `data/` and are not in Git.

## Get the processed data

The snapshot, lookup, service catalog, 2025 schedule, replay map, 2020–2024 pools, pilots, and example episodes are published at `s3://simpleclosure/aeroroute/data/processed`. The download is about 18 GB. Raw monthly ZIPs are not included.

```bash
aws s3 sync --no-sign-request s3://simpleclosure/aeroroute/data/processed data/processed
```

The commands below use those paths. Re-running `build_data_access --stage pilot` rebuilds them instead of using this copy.

## Flow

```text
BTS monthly ZIPs
  -> normalized snapshot
  -> pilot bundle (schedule + frozen outcomes)
  -> Gymnasium env
  -> your policy
```

**1. Download the archives.** Resumable. Files that already pass checksums are skipped.

```bash
uv run python -m aeroroute.cli.download --start-year 2020 --end-year 2025
```

**2. Normalize selected months** into an immutable snapshot. By default, 2020–2024 are the historical fitting years and 2025 is held out.

```bash
uv run python -m aeroroute.cli.normalize --periods 2020-01 2020-03 2020-11 2025-01
```

Pass every `YYYY-MM` you want. The command prints the snapshot directory. Its `dataset_manifest.json` is the `--snapshot` path below.

This checkout already has all 72 months normalized. The snapshot is `data/processed/datasets/61ee17f6e23b1ddbe9fb1cf997fd8c25cf1d6c4ecc0b3ef92714d0e093e7da54/dataset_manifest.json`. The 2025 schedule, the 2020–2024 outcome pools, and exact 2025 replay are already published under `data/processed/access/`. Re-running `build_data_access --stage pilot` starts that build over. Use the pilots below.

**3. Run the earliest-arrival baseline.** One pilot freezes one empirical scenario. This one is Kansas City to Tampa, scenario `train-scenario`, seed 0. The first open checksums the snapshot and the service catalog, so it takes longer than the episode itself.

```bash
uv run --extra rl python -m aeroroute.cli.simulate \
  --snapshot data/processed/datasets/61ee17f6e23b1ddbe9fb1cf997fd8c25cf1d6c4ecc0b3ef92714d0e093e7da54/dataset_manifest.json \
  --pilot data/processed/access/pilots/5aa0aefbbcee9ff8886302e204ac532183e08206623a02c866848399fb3e6e48 \
  --mode empirical \
  --interface gymnasium \
  --agent earliest-arrival
```

The checked run takes one flight and scores **1.0**. Its episode is `data/processed/episodes/96d5073ca16b71c5b58bbb94c005f6b58383d55ce77d43f199adbbb0f4958f5a`.

`--agent random --agent-seed 42` samples a legal action each step. The command writes `experiment.json`, `episode.json`, and `verification.json` under `data/processed/episodes/`.

## Review examples

`data/processed/access/pilots/review_cases.json` lists two or three earliest-arrival episodes for each outcome. Each row names the pilot, the episode, the route, and the seed. Replay (`--mode replay`) scores the 2025 flight’s own recorded outcome. These examples use empirical draws from 2020–2024.

| Outcome | Route | Seeds | Reward | What the trace shows |
| --- | --- | --- | --- | --- |
| On time | MCI → TPA | 0, 0, 3 | 1.0 | One flight arrives by the deadline |
| Cancellation | MCI → TPA, BNA → ATL | 47, 58, 11 | 0.05, 0.05, 0.9 | Two trips stop unresolved after the cancellation. Nashville cancels, then a later flight arrives |
| Missed boarding | MCI → TPA | 2, 7, 8 | 0.05 | The first flight left before the traveler was ready, and the next outcome is unresolved |
| Late arrival | BIL → SLC | 99, 112, 311 | 0.75 | The flight arrives after the deadline |
| Diversion | MIA → LAX, CLT → IAH | 160, 78 | 0.9, 0.65 | Miami still makes the deadline. Charlotte arrives late |

The two on-time seeds of 0 are different scenarios: `train-scenario` and `review-on-time`. A seed only fixes the draw together with its scenario id.

## Train a policy

Open the same pilot, wrap it in `FlightGymEnv`, and step it like any other Gymnasium environment. `policy.act` is yours. It must return an integer slot whose `action_mask` entry is 1.

```python
from contextlib import ExitStack
from pathlib import Path

from aeroroute.catalog.schedule import ScheduleStore
from aeroroute.catalog.services import ServiceCatalog
from aeroroute.domain.records import EpisodeSettings, TripRequest
from aeroroute.outcomes.draws import Scenario
from aeroroute.outcomes.pools import PreparedPools
from aeroroute.outcomes.providers import EmpiricalProvider
from aeroroute.simulation.environment import FlightEnvironment
from aeroroute.simulation.gymnasium_env import FlightGymEnv
from aeroroute.simulation.transitions import TransitionPolicy
from aeroroute.storage.artifacts import contained_path, verify_derived
from aeroroute.storage.dataset import DatasetStore
from aeroroute.storage.identity import load_json

pilot = Path("data/processed/access/pilots/5aa0aefbbcee9ff8886302e204ac532183e08206623a02c866848399fb3e6e48")
snapshot = Path("data/processed/datasets/61ee17f6e23b1ddbe9fb1cf997fd8c25cf1d6c4ecc0b3ef92714d0e093e7da54/dataset_manifest.json")
manifest = verify_derived(pilot, "pilot")
refs, root = manifest["identity"]["references"], pilot.parent.parent

with ExitStack() as stack:
    store = stack.enter_context(DatasetStore(
        snapshot, contained_path(root, "lookups/" + refs["lookup"])))
    catalog = stack.enter_context(ServiceCatalog(
        store, contained_path(root, "services/" + refs["services"])))
    schedule = stack.enter_context(ScheduleStore(
        catalog, contained_path(root, "schedules/" + refs["schedule"])))
    pools = stack.enter_context(PreparedPools(
        catalog, schedule, contained_path(root, "pools/" + refs["pools"])))
    provider = EmpiricalProvider(pools, Scenario(**load_json(pilot / "scenario.json")))

    environment = FlightEnvironment(
        schedule, provider,
        EpisodeSettings(max_duration_minutes=24 * 60, max_decisions=8),
        TransitionPolicy(),
    )
    episode = Path("data/processed/episodes/96d5073ca16b71c5b58bbb94c005f6b58383d55ce77d43f199adbbb0f4958f5a")
    request = TripRequest.from_dict(load_json(episode / "experiment.json")["request"])
    env = FlightGymEnv(environment, request, max_flights=1000, episode_id="train-0")

    observation, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
    trace = env.record
```

Swap trips without rebuilding the environment:

```python
observation, info = env.reset(options={"request": next_request, "episode_id": "train-1"})
```

`env.close()` releases the adapter only. Close the `with` block to release the schedule and outcome files. Leave the inner `FlightEnvironment` alone while an adapter episode is in progress.

`max_flights` is the width of the action vector. If a search offers more flights than that, the step raises. Raise the limit for the experiment you declared. Slots past the real offer list are padded and illegal.

### What the policy sees

`observation` is a dict of the traveler’s state and the flights on offer: airport ids, scheduled times, and the legal-action mask.

| Key | Shape | Meaning |
| --- | --- | --- |
| `flights` | `(max_flights, 5)` float64 | origin airport id, destination airport id, scheduled departure minutes after trip start, scheduled arrival minutes after trip start, `1` if that leg’s destination is the trip destination |
| `state` | `(8,)` float64 | current airport id (`0` if unknown), trip destination id, decision minutes, ready-to-board minutes, deadline minutes, episode length in minutes, decisions already taken, done flag |
| `time_known` | `(2,)` int8 | `1` when decision time is known, `1` when ready-to-board time is known |
| `action_mask` | `(max_flights + 1,)` int8 | `1` on legal actions |

Times are minutes after `request.start_at`. `start_at` is when the traveler is present at the origin, before the initial boarding buffer (default 60 minutes). After a completed flight, the next choice waits out the connection buffer (default 45 minutes).

`info["flight_ids"]` maps occupied rows back to schedule ids, in offer order. Use it for logs. The policy input is the arrays above.

### What the policy does

Actions are integers.

- `0 .. len(offered) - 1` selects that flight. The mask is 1 only for flights the traveler can board.
- `max_flights` (the last slot) is legal only when the episode is already over at reset, for example when the search has no flights. Step it once to receive the terminal reward.

Sample with the mask: `env.action_space.sample(mask=observation["action_mask"])`. An unmasked sample can land on padding and raise.

### What the policy is paid

Every nonterminal step returns `0`. The last step returns:

| Criterion | Weight |
| --- | --- |
| Reached the requested destination | 60 |
| Arrived by `arrival_deadline` | 25 |
| No cancellation, diversion, or missed boarding | 10 |
| Elapsed time known and inside the episode limit | 5 |

Each criterion scores 0 or 1. The reward is the weighted sum divided by 100. A clean on-time arrival scores `1.0`. The deadline is a grade. The episode itself stops on arrival, the time limit, or the decision limit.

### Outcome modes

Use `--mode empirical` (and `EmpiricalProvider`) for training. Each chosen flight draws one historical outcome from matching services in the fitting years. The draw is shifted onto the flight the traveler actually selected.

Use `--mode replay` when you want the held-out departure’s own recorded outcome, including an incomplete one.

One pilot freezes one scenario. The same flight in that scenario always yields the same evidence, including across `reset`. `reset(seed=...)` only seeds adapter randomness. To train on other historical draws, build more pilots with different `--seed` and `--scenario-id` values and rotate those providers.

A 2025 schedule scored with 2020–2024 sampled outcomes is a training distribution. It is a separate question from scoring 2025 flights on their own realized outcomes.

## Check a finished episode

```bash
uv run python -m aeroroute.cli.simulate \
  --snapshot data/processed/datasets/61ee17f6e23b1ddbe9fb1cf997fd8c25cf1d6c4ecc0b3ef92714d0e093e7da54/dataset_manifest.json \
  --pilot data/processed/access/pilots/5aa0aefbbcee9ff8886302e204ac532183e08206623a02c866848399fb3e6e48 \
  --verify-episode data/processed/episodes/96d5073ca16b71c5b58bbb94c005f6b58383d55ce77d43f199adbbb0f4958f5a
```

The verifier rebuilds the offer set, redraws the same outcome, and recomputes the reward. Publish `env.record` through that path when you need a trace you can defend.

## Tests

```bash
uv run --extra rl python -m unittest discover -s tests -v
```
