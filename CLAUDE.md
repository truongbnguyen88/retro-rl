# CLAUDE.md — contra-rl

Operational guide for working in this repo. Token-efficient by design; expand only what's load-bearing.

---

## What this repo is

RL agent that plays Contra (NES). Backend trains and serves the agent; frontend visualizes training + agent play. Backend and frontend communicate **only over the REST API** — never import across that boundary.

---

## Architecture (one-liner per layer)

```
ROM ──► stable-retro env ──► wrappers (preprocess + reward shaping) ──► VecEnv
                                                                          │
                                                                          ▼
                                                                   SB3 PPO + CNN policy
                                                                          │
                       checkpoints, TB logs, eval videos ◄─────────────── │
                                       │
                                       ▼
                              FastAPI backend  ◄── HTTP ──  Streamlit frontend
```

---

## Layout

| Path | Responsibility | Imports from |
|------|---------------|--------------|
| `src/contra_rl/env/`        | Env construction + wrappers + reward shaping | `utils/` only |
| `src/contra_rl/models/`     | CNN feature extractors, policy heads          | nothing internal |
| `src/contra_rl/agents/`     | Algorithm wrappers (PPO, baselines)           | `models/`, `utils/` |
| `src/contra_rl/training/`   | Trainer, callbacks, checkpoint manager        | `env/`, `agents/`, `utils/` |
| `src/contra_rl/evaluation/` | Eval rollouts, metrics, video recording       | `env/`, `agents/`, `utils/` |
| `src/contra_rl/backend/`    | FastAPI app                                   | all above (read-only consumer) |
| `src/contra_rl/utils/`      | config, logging, seeding, video               | nothing internal |
| `frontend/`                 | Streamlit dashboard                           | **only** backend HTTP API |
| `configs/`                  | YAML hyperparams + env config                 | — |
| `scripts/`                  | CLI entrypoints (`train.py`, `evaluate.py`, `play.py`, `serve.py`) | `src/contra_rl/` |
| `outputs/`                  | run artifacts: checkpoints, TB, videos        | gitignored |
| `roms/`                     | user-supplied NES ROM (gitignored, legal)     | — |

**Dependency rule (enforce):** the table is acyclic top-to-bottom; `utils/` is a leaf. If you add a module, place it so no upward import is needed.

---

## Key design decisions

- **PPO over DQN** for the default. Contra has long horizons + visual input + sparse-ish rewards after shaping → on-policy + advantage estimation is more stable to tune. DQN remains a registered baseline.
- **Reward shaping is explicit and config-driven** (`configs/env.yaml`). Components: score delta, x-progress, life loss, death, stage clear. Each weight is a knob; never hardcode in code.
- **Frame stack = 4, gray 84×84, action repeat = 4** — standard Atari preprocessing. Justified by Contra's frame rate and the need for temporal context (bullets, jumps).
- **Vectorized envs** (SB3 `SubprocVecEnv`, n=8 default) for throughput. PPO needs many parallel rollouts per update.
- **Checkpoint cadence**: every N env steps, keep last-K + best-by-eval-return. Atomic write (tmp + rename).
- **Eval is deterministic** (`deterministic=True` in PPO predict, fixed seed list). Train uses stochastic policy.
- **Frontend never imports backend code.** It calls `http://localhost:8000/...`. This is the seam.

---

## Conventions

- Python 3.11+, `pathlib.Path` everywhere, explicit type hints on public functions.
- All config is YAML → pydantic model (`utils/config.py`). No magic strings, no hardcoded paths.
- Logging via `utils/logging.py` (structured). Never `print` in library code.
- Seeds: thread `seed: int` from config → env, numpy, torch, python random. One helper: `utils/seeding.py:set_global_seed`.
- Tests live in `tests/`, mirror `src/contra_rl/` layout. Pytest. Mock the env when the test isn't about env behavior.
- No notebooks in `src/`. Notebooks are for exploration only.

---

## Environment specifics (Contra NES)

- Game ID in stable-retro: `Contra-Nes` (must run `python -m retro.import roms/` once after placing the ROM).
- Action space: discrete, ~12 useful combos (movement × jump × shoot). Use stable-retro's `Discrete` action setup, not multi-binary, unless we need crouch-shoot timing.
- Observation: 240×256×3 → wrapper → 84×84×1 grayscale → stack 4.
- Default training target: **Stage 1 (jungle, side-scroll)**. Stages 2/3 are top-down and need a different reward shaping pass.
- Episode end: death (configurable: end on first death vs. exhaust all lives) or stage clear.

---

## Commands

```bash
# Setup
pip install -e .                              # editable install (uses pyproject.toml)
python -m retro.import roms/                  # import Contra ROM (one-time)

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

- ROM legality: user must supply. Document in README; never ship one.
- stable-retro Contra integration may need a custom `data.json` / `scenario.json` if the default reward signal is too sparse. Check before training and document the path in `docs/environment.md`.
- Throughput on Apple Silicon: stable-retro is CPU-bound for env stepping. Profile early; if blocking, reduce n_envs or use a Linux box.
