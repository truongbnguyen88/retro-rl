# contra-rl

Reinforcement learning agent that plays Contra (NES). PPO + CNN policy on top of [stable-retro](https://github.com/Farama-Foundation/stable-retro), with a FastAPI backend and a Streamlit dashboard.

> **ROM legality** — You must supply your own legally-obtained Contra (NES) ROM. No ROM is included in this repo. Place it under `roms/` and run the import step below.
>
> **Don't have a Contra ROM yet?** The env layer + smoke tests can be validated against **Airstriker (Genesis)**, a freely-distributable homebrew that ships with `stable-retro` (zero extra download). Use `configs/env-airstriker.yaml` instead of `configs/env.yaml` until you can dump your own Contra cartridge. The only clean path to a legal Contra ROM is to dump a cart you own with hardware like the Retrode 2 or INL Retro Dumper.

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
