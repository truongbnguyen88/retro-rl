# Training guide

How to train, what to tune, and what a healthy curve looks like. Read
[`architecture.md`](architecture.md) first if you don't already know what
the wrapper stack does.

---

## Quick start

```bash
# One-time setup
pip install -e ".[dev]"
# macOS Apple Silicon only — see docs/environment.md
bash scripts/install_stable_retro_macos.sh

# Sanity-check the env stack
python scripts/play_random.py --config configs/env.yaml

# Train (writes to outputs/checkpoints/<run_name>/)
python scripts/train.py --config configs/ppo.yaml

# Watch TB live
tensorboard --logdir outputs/tensorboard/

# Or use the dashboard
python scripts/serve.py                    # backend on :8000
streamlit run frontend/app.py              # frontend on :8501
```

A single run on an 8-core machine takes ~12 hours for 4M env steps with
`n_envs=8`. Apple Silicon throughput is CPU-bound on stable-retro, so
more cores ≈ more rollouts/second.

---

## Config layout

Two YAML files, both loaded into pydantic models in [`utils/config.py`](../src/retro_rl/utils/config.py):

- **[`configs/ppo.yaml`](../configs/ppo.yaml)** — PPO hyperparameters,
  schedules, eval/checkpoint cadence, paths.
- **[`configs/env.yaml`](../configs/env.yaml)** — game id, preprocessing,
  reward shaping weights, AutoFire period, discrete action combos.

Override either by editing the YAML directly. Don't add code knobs — if a
hyperparameter is going to be swept, it lives in the config.

---

## The knobs that actually matter

### Reward shaping (`configs/env.yaml` → `reward:`)

The reward signal *is* the policy. Most failed runs in this repo's history
were reward-shape problems, not optimizer problems.

| Field             | Effect                                                              | Typical range          |
|-------------------|---------------------------------------------------------------------|------------------------|
| `score_delta`     | Per-kill bonus = raw score delta × weight, clipped to `clip`.       | 1.0 (kills hit ceiling) |
| `survival_bonus`  | Per-step bonus while alive. Encourages long episodes.                | 0.01–0.03               |
| `life_loss`       | One-shot penalty per life lost (3 lives in Airstriker).             | -10 to -25              |
| `death`           | One-shot penalty on final life. Combined with `life_loss` on death. | -25 to -100             |
| `clip`            | Per-step reward clamp. Asymmetric is fine.                          | `[-50, 10]`             |

**Trap to watch for**: if `clip[0]` is above the combined worst-case step
penalty, every dying episode returns the same flat number and the value
function gets no gradient signal about *when* the agent died. v1 hit this
exactly. Rule of thumb: `clip[0] ≤ life_loss + death + (survival × max_ep_len)`.

### Action space (`configs/env.yaml` → `action_combos`, `auto_fire`)

Airstriker fires only on the **rising edge** of button B; holding B
continuously gives one bullet per life. `AutoFireWrapper` toggles B on a
frame-level schedule independent of the policy:

| `auto_fire.period` | Fire rate | Outcome                                                  |
|--------------------|-----------|----------------------------------------------------------|
| 2  (30 Hz)         | jammed    | Bullet sprite array saturates, game stops responding.    |
| 4  (15 Hz)         | jammed    | Same as above; v5/v6 hit this and capped at ~275 return. |
| 18 (3.33 Hz)       | healthy   | v8 default. Fast enough for kills, slow enough to avoid jam. |
| 24 (2.5 Hz)        | healthy   | v7 default. Every bullet visibly separated.              |
| 60 (1 Hz)          | safe      | Very slow; works but throughput-limited.                 |

`action_combos` encodes Discrete(N) → MultiBinary(12) controller bits.
With AutoFire handling B, combos should leave `B=0` and only set the
directional bits. v5+ uses 9 movement combos (NOOP + 4 cardinal + 4
diagonal).

### PPO hyperparameters (`configs/ppo.yaml` → `ppo:`)

| Field           | What it does                                                          | Default       |
|-----------------|-----------------------------------------------------------------------|---------------|
| `learning_rate` | Adam LR. Linearly schedules to 0 over `total_timesteps`.              | 2.5e-4        |
| `n_steps`       | Rollout length per env. `n_steps × n_envs` is the PPO update batch.   | 128           |
| `batch_size`    | Minibatch size for SGD; must divide `n_steps × n_envs`.               | 256           |
| `n_epochs`      | SGD passes over each rollout.                                         | 4             |
| `gamma`         | Discount.                                                             | 0.99          |
| `gae_lambda`    | GAE λ for advantage estimation.                                       | 0.95          |
| `clip_range`    | PPO surrogate clip. Linear-scheduled to 0.                            | 0.1           |
| `ent_coef`      | Entropy bonus initial value.                                          | 0.02          |
| `ent_coef_final`| Linear-anneal target (v6+). `None` keeps `ent_coef` constant.         | 0.001         |
| `vf_coef`       | Value loss weight.                                                    | 0.5           |
| `max_grad_norm` | Gradient clip.                                                        | 0.5           |

**Entropy schedule matters.** Constant `ent_coef` against Discrete(9)
max-entropy `ln(9) ≈ 2.2` drags the policy back toward uniform once
advantages shrink. v5 (constant `ent_coef=0.02`) flatlined at 2M with
`approx_kl → 0`. v6+ uses a linear anneal to 0.001 by `total_timesteps`.

Pace the anneal: if `total_timesteps=2M`, the floor is reached at 2M
exactly — which is also when collapse tends to start. Doubling
`total_timesteps` (v8: 4M) halves the anneal rate, keeping `ent_coef`
at ~0.013 by the equivalent 2M mark.

---

## Expected curves

A healthy Airstriker run shows these patterns in TB:

| Series                | Healthy shape                                                            |
|-----------------------|--------------------------------------------------------------------------|
| `rollout/ep_rew_mean` | Climbs roughly monotonically. Plateaus before `eval/mean_return` does. |
| `eval/mean_return`    | Lags rollout by ~50–200K steps (deterministic vs stochastic gap).        |
| `eval/mean_length`    | Climbs from ~400 (dies fast) to ~1800 (survives most of Level 1).        |
| `train/approx_kl`     | Drifts down from ~0.02 to ~0.003 over the run. **Hitting 0 means dead.** |
| `train/entropy_loss`  | Climbs slowly (less negative); matches `ent_coef` schedule.              |
| `train/clip_fraction` | Settles in 0.05–0.20 range. Out-of-range suggests `clip_range` issues.   |
| `train/ent_coef`      | Visible as a straight line if the linear schedule is active.             |

Reference numbers (Airstriker, this codebase, 4M-step run):

| Run     | Best `eval/mean_return` | Best `eval/mean_length` | Notes                                  |
|---------|-------------------------|-------------------------|----------------------------------------|
| v5      | 244 @ 1.4M              | ~420                    | Constant `ent_coef`; collapsed late.   |
| v6      | 275 @ ~1.5M             | ~635                    | Added ent_coef schedule; fire-rate cap |
| v7      | 1248 @ 1.5M             | 1820                    | Period 4→24 fixed bullet saturation.   |
| v8 (in flight) | TBD              | TBD                     | Period 18, survival-weighted, 4M.      |

---

## Diagnosing failure modes

| Symptom                                          | Likely cause                                                                                 | Where to look                          |
|--------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------|
| `eval/std_return = 0`                            | Deterministic save-state + deterministic policy → identical eval trajectory.                | Add per-episode seeds or `sticky_action_prob > 0`. |
| `eval/mean_return` flat at clip floor            | Episodes are dying early and `clip[0]` is masking the time-of-death signal.                  | `configs/env.yaml` → `reward.clip`     |
| `eval/mean_return ≪ rollout/ep_rew_mean`         | Deterministic argmax differs significantly from sampled policy. Usually a discretization problem. | `action_combos`, `ent_coef`            |
| `train/approx_kl → 0`                            | Policy collapsed; gradients vanished. Often paired with a flat `eval` curve.                 | `ent_coef` (raise floor or extend schedule) |
| Agent fires once per life                        | AutoFire isn't engaged, OR a combo holds B=1 continuously (rising-edge issue).               | `auto_fire.period`, `action_combos`    |
| Agent fires but scores 0                         | Fire rate too high → bullet array saturated.                                                 | `auto_fire.period` (try ≥ 18)           |
| Throughput < 1000 env-steps/s on 8 cores         | `n_envs` too low OR worker processes are oversubscribed.                                     | `n_envs`, system load                  |

---

## Evaluation + replay

```bash
# Offline eval over N deterministic episodes
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/ppo_airstriker_v7/best.zip \
  --episodes 20

# Render an mp4 of a single rollout
python scripts/play.py \
  --checkpoint outputs/checkpoints/ppo_airstriker_v7/best.zip

# Interactive replay (dashboard) — preferred for ad-hoc inspection
python scripts/serve.py & streamlit run frontend/app.py
```

The dashboard's **Play** page lets you scrub through any checkpoint at
1–30 FPS and watch the policy decide live. The **Compare** page overlays
return curves across multiple runs — useful for ablations.

---

## Pointers

- Architecture and module deep-dive: [`architecture.md`](architecture.md).
- Environment install / stable-retro macOS gotcha: [`environment.md`](environment.md).
- Backend route reference: live at `http://localhost:8000/docs`.
- Full run history with rationales: [`../TASKS.md`](../TASKS.md) → "Decisions log".
