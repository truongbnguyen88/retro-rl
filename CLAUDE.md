# CLAUDE.md — retro-rl

Operational guide for working in this repo. Token-efficient by design; expand only what's load-bearing.

---

## What this repo is

RL agent that plays stable-retro games. Default target: **Airstriker (Genesis)** — a freely-distributable homebrew shooter that ships with the stable-retro distribution, so no separate ROM step. Backend trains and serves the agent; frontend visualizes training + agent play. Backend and frontend communicate **only over the REST API** — never import across that boundary.

---

## Architecture (one-liner per layer)

```
stable-retro env ──► wrappers (preprocess + reward shaping) ──► VecEnv
                                                                  │
                                                                  ▼
                                                           SB3 PPO + CNN policy
                                                                  │
              checkpoints, TB logs, eval videos ◄───────────────  │
                               │
                               ▼
                      FastAPI backend  ◄── HTTP ──  Streamlit frontend
```

---

## Layout

| Path | Responsibility | Imports from |
|------|---------------|--------------|
| `src/retro_rl/env/`        | Env construction + wrappers + reward shaping | `utils/` only |
| `src/retro_rl/models/`     | CNN feature extractors, policy heads          | nothing internal |
| `src/retro_rl/agents/`     | Algorithm wrappers (PPO, baselines)           | `models/`, `utils/` |
| `src/retro_rl/training/`   | Trainer, callbacks, checkpoint manager        | `env/`, `agents/`, `utils/` |
| `src/retro_rl/evaluation/` | Eval rollouts, metrics, video recording       | `env/`, `agents/`, `utils/` |
| `src/retro_rl/backend/`    | FastAPI app                                   | all above (read-only consumer) |
| `src/retro_rl/utils/`      | config, logging, seeding, video               | nothing internal |
| `frontend/`                | Streamlit dashboard                           | **only** backend HTTP API |
| `configs/`                 | YAML hyperparams + env config                 | — |
| `scripts/`                 | CLI entrypoints (`train.py`, `evaluate.py`, `play.py`, `serve.py`) | `src/retro_rl/` |
| `outputs/`                 | run artifacts: checkpoints, TB, videos        | gitignored |
| `roms/`                    | user-supplied ROMs (gitignored; not needed for Airstriker) | — |

**Dependency rule (enforce):** the table is acyclic top-to-bottom; `utils/` is a leaf. If you add a module, place it so no upward import is needed.

---

## Key design decisions

- **PPO over DQN** for the default. On-policy + advantage estimation is more stable to tune for visual input with the reward shapes we use. DQN remains a candidate baseline.
- **Reward shaping is explicit and config-driven** (`configs/env.yaml`). Components: score delta, x-progress (disabled for vertical-scroll games), life loss, death, stage clear. Each weight is a knob; never hardcode in code.
- **Frame stack = 4, gray 84×84, action repeat = 4** — standard Atari preprocessing. Justified by Airstriker's frame rate and the need for temporal context (projectiles, motion direction).
- **Vectorized envs** (SB3 `SubprocVecEnv`, n=8 default) for throughput. PPO needs many parallel rollouts per update.
- **Checkpoint cadence**: every N env steps, keep last-K + best-by-eval-return. Atomic write (tmp + rename).
- **Eval is deterministic** (`deterministic=True` in PPO predict, fixed seed list). Train uses stochastic policy.
- **Frontend never imports backend code.** It calls `http://localhost:8000/...`. This is the seam.
- **Custom feature extractor is `RetroCNN`**, not SB3's `NatureCNN`. Same architecture; ours exposes `features_dim` as a config knob and lets us swap in LSTM/attention heads later without re-plumbing PPO.

---

## Conventions

- Python 3.11+, `pathlib.Path` everywhere, explicit type hints on public functions.
- All config is YAML → pydantic model (`utils/config.py`). No magic strings, no hardcoded paths.
- Logging via `utils/logging.py` (structured). Never `print` in library code.
- Seeds: thread `seed: int` from config → env, numpy, torch, python random. One helper: `utils/seeding.py:set_global_seed`.
- Tests live in `tests/`, mirror `src/retro_rl/` layout. Pytest. Mock the env when the test isn't about env behavior.
- No notebooks in `src/`. Notebooks are for exploration only.

---

## Environment specifics (Airstriker, Genesis)

- Game ID in stable-retro: `Airstriker-Genesis-v0`. ROM ships with `stable-retro` at `<site-packages>/stable_retro/data/stable/Airstriker-Genesis-v0/`; no import step required.
- Action space: discrete (~12 combos via stable-retro's default discretizer).
- Observation: 224×320×3 → wrapper → 84×84×1 grayscale → stack 4.
- Episode end: death (configurable: end on first life-loss vs. exhaust all lives) or hitting `max_episode_steps`.
- Vertical scroll, so `x_progress` shaping is disabled (weight=0) and `info_keys["x_pos"]` points at a missing key (yields 0 contribution + one warning).

### Pointing at a different stable-retro game

1. Update `configs/env.yaml`: `game`, `state`, `scenario`, and `info_keys` to match the new integration's `data.json`.
2. Re-tune the reward weights in `reward:`. Side-scrollers will want `x_progress > 0`.
3. Sanity-check with `python scripts/play_random.py --config configs/env.yaml`.

No code changes needed for a game swap — the env layer is generic.

---

## Commands

```bash
# Setup
pip install -e .                              # editable install

# Sanity-check the env (opens a viewer window)
python scripts/play_random.py --config configs/env.yaml

# Train
python scripts/train.py --config configs/ppo.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint outputs/checkpoints/<run>/best.zip --episodes 20

# Watch agent play (renders mp4 to outputs/videos/)
python scripts/play.py --checkpoint outputs/checkpoints/<run>/best.zip

# Backend
python scripts/serve.py                       # FastAPI on :8000

# Frontend
streamlit run frontend/app.py                 # Streamlit on :8501
```

---

## When implementing

1. **Read first**: before editing a module, read the module + its direct callers.
2. **Incremental**: one module per change. Land env wrappers before agents; agents before trainer; trainer before backend.
3. **Stub then fill**: define the public interface (function signatures + docstring) → write a passing smoke test → implement.
4. **No premature abstraction**. If there's one algorithm, no algorithm registry. Add the registry the second time we need it.
5. **No silent failure**. Surface errors with context (run_id, step, config snapshot path).

---

## What NOT to do

- Don't commit ROMs, checkpoints, TB logs, or videos. `.gitignore` covers these.
- Don't add a config flag for a value that's used once. Inline it.
- Don't couple frontend to backend code via shared imports. HTTP only.
- Don't introduce a new RL library when SB3 covers the case. If we outgrow SB3, that's a deliberate migration, not a drift.
- Don't tune hyperparameters by editing source. They live in `configs/`.

---

## Open questions / risks (live list)

- Throughput on Apple Silicon: stable-retro is CPU-bound for env stepping. Profile early; if blocking, reduce `n_envs` or use a Linux box.
- Airstriker's reward signal may be too sparse for raw score-delta shaping. If learning curves stall, revisit `configs/env.yaml` and consider rewarding enemy kills directly if the integration exposes a kill counter.
- macOS auto-sets `UF_HIDDEN` on files in `.venv/lib/.../site-packages/`, which CPython 3.12.5+ skips for security. The top-level `conftest.py` works around this for tests; production scripts run from the repo root pick up `src/` via cwd.
