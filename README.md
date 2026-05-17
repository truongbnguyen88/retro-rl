# contra-rl

Reinforcement learning agent that plays Contra (NES). PPO + CNN policy on top of [stable-retro](https://github.com/Farama-Foundation/stable-retro), with a FastAPI backend and a Streamlit dashboard.

> **ROM legality** — You must supply your own legally-obtained Contra (NES) ROM. No ROM is included in this repo. Place it under `roms/` and run the import step below.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Import your ROM (one-time)
cp /path/to/your/Contra.nes roms/
python -m retro.import roms/

# 3. Train
python scripts/train.py --config configs/ppo.yaml

# 4. Watch the agent
python scripts/play.py --checkpoint outputs/checkpoints/<run>/best.zip

# 5. Dashboard
python scripts/serve.py            # backend on :8000
streamlit run frontend/app.py      # dashboard on :8501
```

---

## Repo layout

See [CLAUDE.md](CLAUDE.md) for the full module map and dependency rules. The short version:

- `src/contra_rl/` — Python package (env, models, agents, training, evaluation, backend, utils)
- `frontend/` — Streamlit dashboard (talks to backend over HTTP only)
- `configs/` — YAML hyperparameters
- `scripts/` — CLI entrypoints
- `outputs/` — runtime artifacts (gitignored)

---

## Status

See [TASKS.md](TASKS.md) for milestone tracking. Currently: scaffolding complete; environment layer up next.
