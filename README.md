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

# 3. Train
python scripts/train.py --config configs/ppo.yaml

# 4. Evaluate a checkpoint
python scripts/evaluate.py --checkpoint outputs/checkpoints/<run>/best.zip --episodes 20

# 5. Backend (Streamlit frontend still in flight — Milestone 6)
python scripts/serve.py            # FastAPI on :8000
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

See [TASKS.md](TASKS.md) for milestone tracking. **Milestones 1–5 complete** (env layer, models, agents, trainer, evaluator, FastAPI backend). Milestone 6 (Streamlit frontend) is next.

## Training notes (Airstriker)

We iterated the action space + reward shaping four times before reaching a stable training setup. Each iteration's diagnosis is preserved in the TASKS.md decisions log; the short version:

| Version | Problem | Fix |
|---------|---------|-----|
| **v1** | `end_on_life_lost: true` + tight clip → every dying episode returned exactly −10; argmax policy collapsed to "move left → die" | Multi-life episodes, looser clip, `survival_bonus`, lower `ent_coef` |
| **v2** | `MultiBinary(12)` action space → SB3 models each button as independent Bernoulli; P(B=1) converged to ~0.14 — enough for stochastic (1174 return) but below the 0.5 threshold for deterministic eval | `Discrete(9)` over hand-curated combos via [`DiscreteActionWrapper`](src/retro_rl/env/wrappers.py) |
| **v3** | "Always-fire" combos hardcoded `B=1`; agent converged to passive corner-hiding (high `survival_bonus` made dodging more rewarding than engaging) | Reduce `survival_bonus`, amplify `score_delta` |
| **v4** | `score_delta` amplification was a no-op against the `+10` clip ceiling | Raised clip ceiling |
| **v5** *(current)* | **Real bug surfaced**: Airstriker fires only on the *rising edge* of B. Holding B continuously fires one bullet per life. v3/v4's "always-fire" combos were actually "fire once, then never again" | [`AutoFireWrapper`](src/retro_rl/env/wrappers.py) taps B at the emulator-frame level (1-on / 3-off); combos encode movement only. Verified empirically with [`scripts/diagnose_fire_button.py`](scripts/diagnose_fire_button.py) |

The takeaway for future game integrations: **never assume the policy can replace the human's fire-button finger.** Many retro shooters use rising-edge fire semantics. If you swap to a different game, run `diagnose_fire_button.py` to verify the fire mechanic before training.
