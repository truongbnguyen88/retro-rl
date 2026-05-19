# Architecture

This document is the deep-dive companion to [`CLAUDE.md`](../CLAUDE.md). Read
that first for the contract; read this when you need to know *why* a module
exists or *what* it owns.

---

## System diagram

```
        ┌────────────────────────────────────────────────┐
        │ stable-retro emulator (Airstriker-Genesis-v0)  │
        │  raw frames: 224×320×3 uint8, ~60 fps          │
        │  raw actions: MultiBinary(12) controller bits  │
        └────────────────────────┬───────────────────────┘
                                 │
        ┌────────────────────────▼───────────────────────┐
        │ env/ — wrapper stack (outer → inner)           │
        │                                                │
        │   ResizeObservation (84×84)                    │
        │   GrayScaleObservation                         │
        │   FrameStack (4)                               │
        │   RewardShapingWrapper  ◄── configs/env.yaml   │
        │   ActionRepeat (k=4)                           │
        │   DiscreteActionWrapper (Discrete(9) → MB(12)) │
        │   AutoFireWrapper (per-emu-frame B tap)        │
        │   retro.RetroEnv (raw)                         │
        └────────────────────────┬───────────────────────┘
                                 │
                                 ▼
        SubprocVecEnv ──► SB3 PPO (CnnPolicy + RetroCNN)
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │ training/                                      │
        │   Trainer ──► PPO.learn(callbacks=[...])       │
        │   EvalAndVideoCallback   (deterministic eval)  │
        │   CheckpointManager      (last-K + best, atomic│
        │                           tmp → os.replace)    │
        │   EntCoefLinearSchedule  (mutates model.ent_coef│
        │                           between rollouts)    │
        └────────────────────────┬───────────────────────┘
                                 │
                outputs/         │       outputs/
                checkpoints/  ◄──┼──►   tensorboard/
                videos/          │       eval/
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │ backend/ (FastAPI on :8000)                    │
        │   GET  /health, /runs, /checkpoints            │
        │   GET  /runs/{name}/metrics  (TB scalars)      │
        │   POST /episodes  → start rollout              │
        │   GET  /episodes/{id}/{frame,state}            │
        │   DELETE /episodes/{id}                        │
        └────────────────────────┬───────────────────────┘
                                 │ HTTP only
                                 ▼
        ┌────────────────────────────────────────────────┐
        │ frontend/ (Streamlit on :8501)                 │
        │   pages/1_Training.py — TB scalar curves       │
        │   pages/2_Play.py     — checkpoint replay      │
        │   pages/3_Compare.py  — multi-run overlay      │
        └────────────────────────────────────────────────┘
```

The dependency direction is strictly top-to-bottom. `frontend/` imports
nothing from `src/`; it talks to the backend over HTTP. This is the seam
that lets us swap either side independently (e.g., a React frontend, a
different policy server) without touching the other.

---

## Module deep-dive

### `env/` — environment and reward shaping

The wrapper stack is the most subtle part of the codebase. Order matters
because each wrapper transforms either the observation, the action, or the
reward — and some wrappers (`AutoFireWrapper`, `ActionRepeat`) need to see
emulator-level frames rather than agent-level decisions.

| Wrapper                  | Layer | Owns                                                       |
|--------------------------|-------|------------------------------------------------------------|
| `AutoFireWrapper`        | innermost (next to raw env) | Toggles the fire button (B) on a frame-level schedule, *inside* `ActionRepeat`'s skip loop. Necessary because Airstriker fires only on the rising edge of B; holding B continuously gives one bullet per life. |
| `DiscreteActionWrapper`  | next  | Maps `Discrete(9)` → `MultiBinary(12)`. Combos are config-driven (`EnvConfig.action_combos`); v5+ encode 8-way movement + NOOP with B=0 (AutoFire handles fire). |
| `ActionRepeat`           | mid   | Repeats the chosen action for `k=4` emulator frames; aggregates reward, picks the last observation. |
| `RewardShapingWrapper`   | mid   | Computes shaped reward from `info` deltas (score, lives, x_progress). Weights live in `configs/env.yaml` under `reward:`. |
| `FrameStack`             | outer | Stacks 4 grayscale frames along the channel axis. |
| `GrayScaleObservation`   | outer | RGB → single-channel luminance. |
| `ResizeObservation`      | outermost | 224×320 → 84×84. |

**Reward shaping** ([`reward_shaping.py`](../src/retro_rl/env/reward_shaping.py))
is config-driven; never hardcoded. Each component (`score_delta`,
`x_progress`, `life_loss`, `death`, `stage_clear`, `survival_bonus`) is a
weight times a delta or constant. Set a weight to 0 to disable the term.
For vertical-scroll games like Airstriker, `x_progress` is disabled and
`info_keys.x_pos` points at a missing key (which yields 0 and one warning).

### `models/` — feature extractors

`RetroCNN` is a thin wrapper over SB3's `NatureCNN` architecture with a
configurable `features_dim`. Lives in `models/` because we want to swap
in LSTM/attention heads later without touching `agents/` or `training/`.

### `agents/` — algorithm wrappers

Currently one production agent (SB3 PPO via `policy_kwargs(features_dim)`)
plus a `RandomAgent` for baselines. `agents/base.py` is a `typing.Protocol`,
not an ABC — both PPO and `RandomAgent` conform structurally without
forcing SB3 into a custom hierarchy.

### `training/` — trainer + callbacks

| Component                | Owns                                                                                       |
|--------------------------|--------------------------------------------------------------------------------------------|
| `Trainer`                | Wires VecEnv + PPO + callbacks; runs `PPO.learn(...)`; persists final model.               |
| `EvalAndVideoCallback`   | Periodic deterministic eval; writes mp4 per checkpoint; emits `eval/*` scalars to TB.      |
| `CheckpointManager`      | Atomic save (tmp + `os.replace`); maintains last-K + best-by-eval-return; JSON sidecar.    |
| `EntCoefLinearSchedule`  | Mutates `model.ent_coef` between rollouts (v6+); linear anneal from `ent_coef`→`ent_coef_final`. |

**`SubprocVecEnv` always**, even at `n_envs=1`. stable-retro hard-limits
one emulator per *process*; if the train emulator lives in the main process,
the eval callback's emulator construction fails. SubprocVecEnv puts train
emulators in workers, leaving the main process free for eval.

### `evaluation/` — eval rollouts and metrics

Used by both `scripts/evaluate.py` (offline eval over N episodes) and the
backend's `/episodes` rollout. `EvalMetrics` is a flat dataclass of
mean/std/min/max return + episode lengths; `evaluator.evaluate(...)` is
the entry point. `utils/video.py` handles mp4 encoding via imageio-ffmpeg.

### `backend/` — FastAPI service

| Module          | Responsibility                                                          |
|-----------------|-------------------------------------------------------------------------|
| `api.py`        | Route handlers + lifespan; dependency-injects singletons from `app.state`. |
| `inference.py`  | `CheckpointRegistry` (filesystem scan), `EpisodeManager` (live rollouts), `MetricsReader` (TB events parser). |
| `models.py`     | Pydantic response models — frozen for the HTTP contract.                |

The backend is a **read-only consumer** of `outputs/`: it never writes
checkpoints, never trains. State is held in `app.state.*` singletons,
constructed at startup, torn down in the lifespan handler.

### `frontend/` — Streamlit dashboard

Three pages, one shared sidebar. **No imports from `src/`** — all backend
contact goes through `frontend/components/api_client.py`, which wraps
`requests` with `st.cache_data` TTLs (5 s health, 10 s catalog, 15 s
metrics; episode endpoints uncached). The Play page's frame stream runs
an explicit `while playing` loop inside one Streamlit rerun, refreshing
an `st.empty()` placeholder at 1–30 FPS.

### `utils/` — config, logging, seeding, video

`utils/` is a leaf — it imports nothing internal. `config.py` defines the
pydantic models that every other module loads YAML through. `seeding.py`
exposes `set_global_seed(seed)` which threads `seed` into env, numpy,
torch, and python random. `logging.py` is structured (one record per
event); never `print` in library code.

---

## Data flow examples

### Training tick

1. `Trainer.run()` constructs `SubprocVecEnv([make_retro_env() × n_envs])`.
2. PPO runs `n_steps=2048` rollouts in parallel across workers.
3. Each worker's env stack: raw retro → AutoFire → DiscreteAction →
   ActionRepeat(4) → RewardShaping → FrameStack(4) → Gray → Resize.
4. PPO computes advantages over the rollout buffer; runs `n_epochs=10`
   of minibatch SGD on policy + value losses.
5. `EntCoefLinearSchedule` updates `model.ent_coef` at `rollout_end`.
6. Every `eval_freq` env steps, `EvalAndVideoCallback` builds a fresh
   single-env eval emulator in the main process, runs N deterministic
   episodes with `seed=eval_seed+ep_i`, writes an mp4, and emits
   `eval/mean_return`, `eval/std_return`, `eval/mean_length` to TB.
7. `CheckpointManager` atomically writes `outputs/checkpoints/<run>/
   step-XXX.zip` + sidecar JSON; updates `last.zip` (symlink) and
   `best.zip` if eval mean return improved.

### Replay (Play page)

1. User picks a checkpoint in `pages/2_Play.py`, clicks Start.
2. `api_client.start_episode(checkpoint_id, ...)` → `POST /episodes`.
3. Backend `EpisodeManager` lazy-loads the SB3 PPO model, builds a
   single-env retro emulator, resets it, returns `{episode_id, ...}`.
4. Frontend enters the streaming loop: `GET /episodes/{id}/frame`
   (which advances one env step + returns PNG bytes) at the chosen FPS.
5. On Stop or episode `done`, frontend calls `DELETE /episodes/{id}`;
   backend disposes the emulator and the model reference is released.

---

## Why these choices

**PPO over DQN.** On-policy + advantage estimation has been more stable
to tune for visual input with the reward shapes we use. DQN remains a
candidate baseline but isn't worth the extra plumbing until PPO ceiling
is clearly hit.

**Custom `RetroCNN` over SB3's `NatureCNN`.** Same architecture; ours
exposes `features_dim` as a config knob and lets us swap in LSTM/
attention heads later without re-plumbing PPO.

**HTTP seam between backend and frontend.** Lets us swap frontends
(React, mobile, CLI) and policy servers independently. Cost is the
serialization overhead; benefit is zero coupling.

**Atomic checkpoint writes (tmp → os.replace).** Eliminates a class of
"crashed mid-save" bugs where a partial zip would be loaded later. The
JSON sidecar holds eval return + step count and survives even if the
zip itself is corrupted.

**`SubprocVecEnv` at all `n_envs`.** stable-retro's per-process emulator
limit makes `DummyVecEnv` unusable for any setup with an eval callback.
The IPC cost at `n_envs=1` is negligible vs env stepping.

---

## Pointers

- Setup, including macOS-specific stable-retro install: [`environment.md`](environment.md).
- Day-to-day training workflow + knobs: [`training.md`](training.md).
- Backend route reference: live at `http://localhost:8000/docs` (OpenAPI).
- Operational contract + module dependency rule: [`CLAUDE.md`](../CLAUDE.md).
