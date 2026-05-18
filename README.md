# retro-rl
(Renamed from contra-rl)

Reinforcement learning agent for [stable-retro](https://github.com/Farama-Foundation/stable-retro) games. PPO + CNN policy with a FastAPI backend and a Streamlit dashboard. Default target: **Airstriker (Genesis)** — a freely-distributable homebrew shooter that ships with stable-retro, so no separate ROM step is needed.

> **Pointing at a different game?** The env layer is config-driven — swap `configs/env.yaml`'s `game`, `state`, and `info_keys` and everything downstream (wrappers, CNN, PPO) works unchanged. ROM legality is the user's responsibility.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1a. Apple Silicon / macOS only — stable-retro needs a source-build patch.
#     See docs/environment.md for the why.
brew install cmake pkg-config
./scripts/install_stable_retro_macos.sh

# 2. Sanity-check the env (opens a viewer window; random-action rollout)
python scripts/play_random.py --config configs/env.yaml

# 3. Train  (Milestone 3 — coming up)
python scripts/train.py --config configs/ppo.yaml

# 4. Watch the trained agent  (Milestone 3)
python scripts/play.py --checkpoint outputs/checkpoints/<run>/best.zip

# 5. Dashboard  (Milestones 5-6)
python scripts/serve.py            # backend on :8000
streamlit run frontend/app.py      # dashboard on :8501
```

---

## Repo layout

See [CLAUDE.md](CLAUDE.md) for the full module map and dependency rules. The short version:

- `src/retro_rl/` — Python package (env, models, agents, training, evaluation, backend, utils)
- `frontend/` — Streamlit dashboard (talks to backend over HTTP only)
- `configs/` — YAML hyperparameters
- `scripts/` — CLI entrypoints
- `outputs/` — runtime artifacts (gitignored)

---

## Status

See [TASKS.md](TASKS.md) for milestone tracking. Currently: env layer + models + agents (Milestones 1–2) complete; training pipeline (Milestone 3) is next.
