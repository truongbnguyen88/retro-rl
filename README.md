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

We iterated the action space, reward shaping, optimiser regularisation, and the fire wrapper across seven runs before reaching a stable setup. Each iteration's diagnosis is preserved in the TASKS.md decisions log; the short version:

| Version | Problem | Fix |
|---------|---------|-----|
| **v1** | `end_on_life_lost: true` + tight clip → every dying episode returned exactly −10; argmax policy collapsed to "move left → die" | Multi-life episodes, looser clip, `survival_bonus`, lower `ent_coef` |
| **v2** | `MultiBinary(12)` action space → SB3 models each button as independent Bernoulli; P(B=1) converged to ~0.14 — enough for stochastic (1174 return) but below the 0.5 threshold for deterministic eval | `Discrete(9)` over hand-curated combos via [`DiscreteActionWrapper`](src/retro_rl/env/wrappers.py) |
| **v3** | "Always-fire" combos hardcoded `B=1`; agent converged to passive corner-hiding (high `survival_bonus` made dodging more rewarding than engaging) | Reduce `survival_bonus`, amplify `score_delta` |
| **v4** | `score_delta` amplification was a no-op against the `+10` clip ceiling | Raised clip ceiling |
| **v5** | **Rising-edge fire bug**: Airstriker fires only on the *rising edge* of B. v3/v4's "always-fire" combos held B continuously, firing exactly one bullet per life | [`AutoFireWrapper`](src/retro_rl/env/wrappers.py) taps B at the emulator-frame level (1-on / 3-off). Combos encode movement only. Verified with [`scripts/diagnose_fire_button.py`](scripts/diagnose_fire_button.py). Mean return 193, peak 244 over 2M |
| **v6** | Constant `ent_coef=0.02` against Discrete(9) max entropy `ln(9) ≈ 2.197` kept the policy near-uniform once advantages shrank; eval return peaked at 244 (1.4M) then *regressed* to 214 by 2M with `approx_kl → 0` | Linear `ent_coef` schedule 0.02 → 0.001 over total_timesteps via new [`EntCoefLinearSchedule`](src/retro_rl/training/callbacks.py). Mean rose to 222, peak to 275 over 1.6M, but still ceilinged because v5's tap-fire was actually *too fast* |
| **v7** *(current)* | **Bullet-array saturation**: AutoFire at period=4 (15 Hz) jammed Airstriker's player-bullet sprite slots within ~0.6s of each life. Game silently dropped B for the remaining 2-5s, so v5/v6 training data only ever covered the first ~5s of Level 1 | Drop AutoFire to `period=24` (2.5 Hz, ~25 bullets per life). Slot array never saturates → fire is visibly active end-to-end → policy can train on deeper level coverage. Carries v6's entropy schedule |

The takeaways for future retro-shooter integrations:

1. **Never assume the policy can replace the human's fire-button finger.** Rising-edge fire semantics are common. Run `diagnose_fire_button.py` to verify the mechanic before training.
2. **Many retro shooters cap on-screen bullets with a sprite array, and saturating it makes the game silently ignore further fire input.** Run [`scripts/diagnose_fire_rate_vs_state.py`](scripts/diagnose_fire_rate_vs_state.py) to sweep tap rates and pick one that doesn't saturate.
3. **Validate fire-mechanic fixes by extracting individual eval-video frames** ([`scripts/diagnose_video_kill_frames.py`](scripts/diagnose_video_kill_frames.py)) and confirming bullets are visibly active throughout each life — not just at the start. Visible-bullet-density is more diagnostic than the score curve, especially on the first checkpoint.
