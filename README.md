# retro-rl

Reinforcement learning agent for [stable-retro](https://github.com/Farama-Foundation/stable-retro) games. PPO / RecurrentPPO + CNN policy with a FastAPI backend and a Streamlit dashboard. Default target: **Airstriker (Genesis)** — a freely-distributable homebrew shooter that ships with stable-retro, so no separate ROM step is needed.

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

# 5. Backend + frontend
python scripts/serve.py            # FastAPI on :8000
streamlit run frontend/app.py      # Streamlit dashboard on :8501
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

See [TASKS.md](TASKS.md) for milestone tracking. **Milestones 1–7 complete** (env layer, models, agents, trainer, evaluator, FastAPI backend, Streamlit frontend, docs + CI + pre-commit). Remaining work is experimental training iteration.

| Run | Algorithm | Best return | Status |
|-----|-----------|-------------|--------|
| v8 | PPO + RetroCNN | 4271 @ 3.3M | ✅ complete |
| v9 | PPO + IMPALA ResNet + VecNormalize | **7168 @ 2.7M** | ✅ **best — clears the game** |
| v10 | RecurrentPPO + LSTM + IMPALA, frame_stack=1 | 1993 @ 4M | ❌ retired (cold-start blindness + clip drift) |
| v11 | RecurrentPPO + LSTM + IMPALA, frame_stack=4 | 5325 @ 5.7M | ✅ complete (+25% vs v8, −26% vs v9) |
| v12 | PPO + **temporal-attention** extractor, frame_stack=8 | 500K smoke | 🔄 **smoke running** |

**Current line of work (v12):** v10/v11 tested whether an LSTM's recurrent memory could beat the stateless v9 baseline — both fell short, capped by the LSTM's `(h, c)=0` cold-start at every episode reset (a structural eval-time penalty). v12 swaps the LSTM for **self-attention over the K-frame window** ([`models/attention.py`](src/retro_rl/models/attention.py)): no hidden state to initialize ⇒ no cold-start, and it runs on **standard PPO** (no recurrent machinery). See the v12 row below and the TASKS.md decisions log for the full rationale.

## Training notes (Airstriker)

For a from-scratch walkthrough of the v9 training pipeline — observation inputs, the IMPALA ResNet backbone, the PPO learning loop, and GAE — see [docs/v9_procedure_pipeline.md](docs/v9_procedure_pipeline.md).

We iterated the action space, reward shaping, optimiser regularisation, fire wrapper, and now recurrent architecture across many runs before reaching the current setup. Each iteration's diagnosis is in the TASKS.md decisions log; the short version:

| Version | Problem | Fix |
|---------|---------|-----|
| **v1** | `end_on_life_lost: true` + tight clip → every dying episode returned exactly −10; argmax policy collapsed to "move left → die" | Multi-life episodes, looser clip, `survival_bonus`, lower `ent_coef` |
| **v2** | `MultiBinary(12)` action space → SB3 models each button as independent Bernoulli; P(B=1) converged to ~0.14 — enough for stochastic (1174 return) but below the 0.5 threshold for deterministic eval | `Discrete(9)` over hand-curated combos via [`DiscreteActionWrapper`](src/retro_rl/env/wrappers.py) |
| **v3** | "Always-fire" combos hardcoded `B=1`; agent converged to passive corner-hiding (high `survival_bonus` made dodging more rewarding than engaging) | Reduce `survival_bonus`, amplify `score_delta` |
| **v4** | `score_delta` amplification was a no-op against the `+10` clip ceiling | Raised clip ceiling |
| **v5** | **Rising-edge fire bug**: Airstriker fires only on the *rising edge* of B. v3/v4's "always-fire" combos held B continuously, firing exactly one bullet per life | [`AutoFireWrapper`](src/retro_rl/env/wrappers.py) taps B at the emulator-frame level (1-on / 3-off). Combos encode movement only. Verified with [`scripts/diagnose_fire_button.py`](scripts/diagnose_fire_button.py). Mean return 193, peak 244 over 2M |
| **v6** | Constant `ent_coef=0.02` against Discrete(9) max entropy `ln(9) ≈ 2.197` kept the policy near-uniform once advantages shrank; eval return peaked at 244 (1.4M) then *regressed* to 214 by 2M with `approx_kl → 0` | Linear `ent_coef` schedule 0.02 → 0.001 over total_timesteps via new [`EntCoefLinearSchedule`](src/retro_rl/training/callbacks.py). Mean rose to 222, peak to 275 over 1.6M, but still ceilinged because v5's tap-fire was actually *too fast* |
| **v7** | **Bullet-array saturation**: AutoFire at period=4 (15 Hz) jammed Airstriker's player-bullet sprite slots within ~0.6s of each life. Game silently dropped B for the remaining 2-5s, so v5/v6 training data only ever covered the first ~5s of Level 1 | Drop AutoFire to `period=24` (2.5 Hz, ~25 bullets per life). Slot array never saturates → fire is visibly active end-to-end → policy can train on deeper level coverage. Carries v6's entropy schedule. Peak eval return **1248** @ 1.5M, episode length 1820 (+3.3×) |
| **v8** | v7 confirmed the saturation fix but late-run `approx_kl → 0` at 2M suggested the entropy schedule annealed too fast over 2M; agent still under-prioritised survival | `period=18` (3.33 Hz), survival reward tripled (`survival_bonus` 0.01→0.03) + life-loss penalty doubled (−10→−20), `total_timesteps=4M` so `ent_coef` anneals at half the rate. Best eval return **4271** @ 3.3M (+3.4× over v7) |
| **v9** | v8 replay shows 2–3-frame action "shaking" (LEFT↔RIGHT logit oscillation) and a possible representational ceiling from the Nature-CNN | Bundles three structural changes — `action_repeat` 4→8 (7.5 Hz decisions), Nature-CNN → **IMPALA ResNet** ([`models/impala.py`](src/retro_rl/models/impala.py)), `auto_fire.period` 18→12 (5 Hz) — plus two stability fixes a 3-rung 500K smoke ladder surfaced: `learning_rate` 2.5e-4→1e-4 and **`normalize_reward: true`** (SB3 VecNormalize, reward only). VecNorm was decisive: EV 0.4→0.9, value_loss ~800→~0.1, smoke eval return @500K ~245→2608. Best eval return **7168 @ 2.7M** (+1.68× over v8). Configs: [`ppo_v9.yaml`](configs/ppo_v9.yaml) + [`env_v9.yaml`](configs/env_v9.yaml) |
| **v10** *(retired)* | Hypothesis: "LSTM owns all temporal memory, frame_stack=1 is sufficient." RecurrentPPO + LSTM(256) + IMPALA + `frame_stack=1`, 4M steps | **Three failure modes** all confirmed: (1) `frame_stack=1` cold-start blindness — `h=0,c=0` at episode reset means single-frame obs contains no velocity; agent flies blind for first several steps (mean_length stuck at 555). (2) `clip_fraction=0.363` LSTM re-execution drift stuck since 1.2M. (3) Stochastic/deterministic policy gap (rollout 3311 vs eval 1993). Final best return: 1993 — significantly below v9 |
| **v11** | v10's failure modes: cold-start blindness + LSTM drift | `frame_stack 1→4` (fast motion handled by stack; LSTM handles slow spawn cycles — complementary timescales), `n_epochs 4→2` (halves LSTM re-execution drift per rollout), `lr 1e-4→8e-5`, `total_timesteps 4M→6M`. The fixes worked (clip_fraction healthy, early survival up) but the eval curve stayed **noisy/bimodal** and best return **5325 @ 5.7M** — **+25% over v8 but −26% vs v9, despite 50% more compute**. The big evals only appeared in the last 1M steps once `ent_coef→0.001` closed the stochastic/deterministic gap. Root cause that capped it: the LSTM cold-start is *structural* — `(h,c)=0` each reset, so "bad" episodes never recover (the bimodality). **Clean negative result: recurrence does not beat stateless IMPALA PPO on this game.** Configs: [`ppo_v11.yaml`](configs/ppo_v11.yaml) + [`env_v11.yaml`](configs/env_v11.yaml) |
| **v12** *(smoke running)* | Can attention-based temporal aggregation beat conv-over-channels *without* the LSTM's cold-start cost? | Replace the LSTM with a **`TemporalAttentionExtractor`** ([`models/attention.py`](src/retro_rl/models/attention.py)): shared per-frame Nature-CNN tokenizer → learned positional embedding → 2-layer/4-head pre-LN `TransformerEncoder` (full self-attention over the K=8 frame window) → last-token readout. **No hidden state ⇒ no cold-start** (FrameStack pads the window at reset). Runs on **standard PPO** — registered as `"temporal_attn"` in the extractor registry, so trainer/eval/backend are unchanged. Config is a thin overlay on the v9 best run (swap extractor + `frame_stack 4→8`; keep VecNormalize/lr=1e-4/ent-schedule/4M). Smoke GO gate: EV>0.85, clip_fraction<0.25, return ~2608 @ 500K (v9 smoke bar), fps≥~25. Configs: [`ppo_v12.yaml`](configs/ppo_v12.yaml) + [`env_v12.yaml`](configs/env_v12.yaml) |

The takeaways for future retro-shooter integrations:

1. **Never assume the policy can replace the human's fire-button finger.** Rising-edge fire semantics are common. Run `diagnose_fire_button.py` to verify the mechanic before training.
2. **Many retro shooters cap on-screen bullets with a sprite array, and saturating it makes the game silently ignore further fire input.** Run [`scripts/diagnose_fire_rate_vs_state.py`](scripts/diagnose_fire_rate_vs_state.py) to sweep tap rates and pick one that doesn't saturate.
3. **Validate fire-mechanic fixes by extracting individual eval-video frames** ([`scripts/diagnose_video_kill_frames.py`](scripts/diagnose_video_kill_frames.py)) and confirming bullets are visibly active throughout each life — not just at the start. Visible-bullet-density is more diagnostic than the score curve, especially on the first checkpoint.
4. **A deeper net or a higher `action_repeat` can stall the value function via return-target *scale*, not learning rate.** v9's IMPALA + `action_repeat=8` produced large, high-variance per-step rewards; the value head pinned at `explained_variance ≈ 0` (LR too high) and, once the LR was lowered, ceilinged at ~0.4 with `value_loss ≈ 800`. Lowering LR further would only learn slower. The fix was reward normalization (`VecNormalize`, `norm_reward=True`, `norm_obs=False`) — rescaling returns to ~unit variance made the regression well-conditioned (EV→0.9, `value_loss`→~0.1) and the cleaner value baseline unblocked policy learning. Eval stays on a bare env reporting raw returns, so the metric remains comparable; the running stats are checkpointed as a `.pkl` sidecar for resume.
5. **`frame_stack` and LSTM are not redundant — they operate at different timescales.** v10 tested the "LSTM owns all memory" hypothesis by setting `frame_stack=1`. The LSTM's `h=0,c=0` at episode reset provides zero velocity context; a single grayscale frame contains no motion information. The result was cold-start blindness every life (mean_length 555 vs v9's 4500). v11's fix: restore `frame_stack=4` for fast motion (4 frames ≈ 533ms window) and let the LSTM handle slow temporal patterns (spawn cycles, threat timing). For games with rapid relative motion between objects, frame stacking is necessary input preprocessing, not a redundant memory mechanism.
6. **RecurrentPPO with `n_epochs > 2` accumulates LSTM re-execution drift.** Each optimization epoch re-runs the LSTM with updated weights, widening the gap between the current policy π_θ and the rollout policy π_θ_old used to compute importance weights. At `n_epochs=4`, v10's `clip_fraction` locked at 0.363 from 1.2M to 4M steps — the oscillation never resolved. Halving to `n_epochs=2` (v11) immediately brought clip_fraction into 0.04–0.12 range. For recurrent policies, treat `n_epochs=2` as the default and only increase if clip_fraction is consistently low.
7. **An LSTM's eval-time cold-start can be a structural ceiling, not a tuning problem.** v11 fixed v10's clip drift and early-survival issues yet still lost to stateless v9 by 26% with 50% more compute. The tell was a persistently **bimodal eval curve** (episodes either long or dead-on-arrival) — `(h, c)=0` at every reset means episodes where the LSTM never re-accumulates context just die. No amount of extra steps fixes this; it's inherent to resetting recurrent state per episode. The same effect showed up operationally: stochastic training rollouts (warm state, continuous) cleared the game while deterministic checkpoint playback (cold state) died at stage 3. If you need temporal modeling, prefer a mechanism with **no state to initialize** — a fixed-window transformer (v12) attends over an always-populated frame buffer and sidesteps the cold-start entirely. Before reaching for recurrence at all, confirm the task actually needs memory beyond frame stacking: v9 clears the game with a 533ms (4-frame) window, which is why v10/v11 were a net negative.
