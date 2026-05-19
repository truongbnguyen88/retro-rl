# TASKS.md — retro-rl

Source of truth for what's done, in flight, and queued. Update as work lands. Keep it concise — one line per task; expand only in linked docs.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Milestone 0 — Repo scaffolding ✅

- [x] Plan architecture and module boundaries
- [x] Create directory skeleton (`src/`, `frontend/`, `configs/`, `scripts/`, `tests/`, `outputs/`, `roms/`)
- [x] Write `CLAUDE.md` (repo-level operating guide)
- [x] Write `TASKS.md` (this file)
- [x] `.gitignore` covers ROMs, checkpoints, TB logs, videos, venv
- [x] `pyproject.toml` with editable install + dependency groups
- [x] `requirements.txt` mirroring pyproject for ease
- [x] Stub `README.md` with setup + quickstart
- [x] Default YAML configs (`default.yaml`, `ppo.yaml`, `env.yaml`)

## Milestone 1 — Environment layer ✅

- [x] `utils/config.py`: pydantic models for env/ppo/training config; YAML loader w/ `extends:`
- [x] `utils/seeding.py`: `set_global_seed` (python/numpy/torch/PYTHONHASHSEED)
- [x] `utils/logging.py`: structured logger factory; per-run rotating file
- [x] `env/retro_env.py`: stable-retro factory function; ROM import check w/ actionable error
- [x] `env/wrappers.py`: grayscale, resize 84×84, frame stack 4, action-repeat, max-episode-steps, sticky-action option, end-on-life-lost
- [x] `env/reward_shaping.py`: configurable shaping (score Δ, x-progress, life loss, death, stage clear); pure function over info dict; per-integration `info_keys` override
- [x] `env/__init__.py`: `make_env(config)` + `make_env_fn(config, rank)` public entrypoints
- [x] `tests/test_env.py`: **17 passed** (all env-layer wrappers + Airstriker smoke green on any clean install)
- [x] `configs/env.yaml`: Airstriker-Genesis-v0 default config — smoke test runs end-to-end
- [x] `scripts/install_stable_retro_macos.sh`: source-build script with zlib patch for Apple Silicon
- [x] `scripts/play_random.py`: visual sanity-check CLI — random-action rollouts with viewer window
- [x] `docs/environment.md`: rationale for the macOS arm64 build path

## Milestone 2 — Models + agents ✅

- [x] `models/cnn.py`: `RetroCNN` Nature-CNN feature extractor (SB3-compatible, configurable `features_dim`, uint8→float normalization inside `forward`)
- [x] `models/policies.py`: `policy_kwargs(features_dim)` helper wiring `RetroCNN` into SB3 `CnnPolicy` (no subclass — premature abstraction avoided)
- [x] `agents/base.py`: `Agent` Protocol (`predict`, `save`, `load`); structural — both SB3 PPO and `RandomAgent` conform
- [x] `agents/ppo.py`: `build_ppo(vec_env, cfg)` factory with linear schedules on lr + clip_range
- [x] `agents/random_agent.py`: uniform-random baseline; JSON sidecar persistence; handles single + batched image obs
- [x] `tests/test_models.py`: **6 passed** — CNN forward shapes (default + custom `features_dim`, FS=1, uint8 extremes), space validation, policy_kwargs wiring
- [x] `tests/test_agents.py`: **11 passed** — RandomAgent (predict shapes, reproducibility, save/load, Protocol conformance), PPO factory (construction, CNN wiring, policy gate, predict, save/load roundtrip), linear schedule
- [x] `conftest.py`: top-level pytest hook adding `src/` to `sys.path` — workaround for macOS auto-applying `UF_HIDDEN` to venv `.pth` files (CPython 3.12.5+ skips hidden `.pth` for security)

## Milestone 3 — Training pipeline ✅

- [x] `training/checkpoint.py`: `CheckpointManager` — atomic save (tmp + `os.replace`), JSON sidecar (`run_name`, `step`, `eval_return`, `kind`, `config_snapshot_path`, `timestamp`), last-K pruning (best excluded), best-tracker restored from disk on init
- [x] `training/callbacks.py`: `PeriodicCheckpointCallback` (rotates step ckpts) + `EvalAndVideoCallback` (deterministic rollouts, TB scalars `eval/{mean_return,std_return,mean_length}`, mp4 of first episode per cycle, updates best via manager)
- [x] `training/trainer.py`: `train(cfg, resume_from=None) -> Path` — config snapshot, SubprocVecEnv (always, even at `n_envs=1`), `build_ppo` or `PPO.load`, callback wiring, final ckpt save, returns best (or latest)
- [x] `scripts/train.py`: argparse CLI — `--config`, `--resume`
- [x] `configs/ppo_smoke.yaml`: extends `ppo.yaml`; `total_timesteps=10000`, `n_envs=1`, eval every 2500, ckpt every 5000
- [x] `tests/test_training.py`: **14 passed** — CheckpointManager unit tests (atomic, sidecar, best-tracking, last-K rotation, restore from disk)
- [x] Smoke acceptance: `python scripts/train.py --config configs/ppo_smoke.yaml` completes; `outputs/checkpoints/ppo_airstriker_smoke/{best,step-*}.{zip,json}` + `outputs/tensorboard/ppo_airstriker_smoke_*/events.*` + `outputs/videos/ppo_airstriker_smoke/eval-step-*.mp4` all produced

## Milestone 4 — Evaluation ✅

- [x] `evaluation/evaluator.py`: `evaluate(agent, env, n_episodes, ...)` → `(EvalMetrics, frames)`; death + stage-clear tracking from info dict; first-episode frame capture
- [x] `evaluation/metrics.py`: `EpisodeResult` + `EvalMetrics` frozen dataclasses; `compute_metrics` — mean/std/min/max return, mean episode length, stage-clear rate, deaths-per-episode
- [x] `utils/video.py`: `write_mp4(frames, path, fps)` — atomic write (tmp + rename); extracted from inline callback code
- [x] `scripts/evaluate.py`: argparse CLI — `--checkpoint`, `--config`, `--episodes`, `--seed`, `--no-video`, `--output-dir`; emits `metrics.json` + `episode_0.mp4`
- [x] `training/callbacks.py`: refactored to use `utils.video.write_mp4` (removed inline `_write_mp4`)
- [x] `tests/test_evaluation.py`: **21 passed** — `compute_metrics` (correctness, frozen, empty guard), `evaluate` (episode count, returns, death/truncation distinction, stage-clear key mapping, first-episode-only frames), `write_mp4` (creates file, no tmp residue, empty guard, string path)

## Milestone 5 — Backend (FastAPI) ✅

- [x] `backend/models.py`: 11 pydantic schemas — `HealthResponse`, `CheckpointInfo`/`CheckpointList`, `EpisodeStartRequest`/`EpisodeStartResponse`/`EpisodeState`, `RunInfo`/`RunList`, `MetricPoint`/`MetricSeries`/`RunMetrics`, `ErrorResponse`; all `extra="forbid"`
- [x] `backend/inference.py`: `CheckpointResolver` (id↔path, snapshot→EnvConfig, `list_all()`), `AgentRegistry` (LRU-bounded PPO cache, thread-safe), `EpisodeRuntime` (per-rollout: step/state/frame_png/close with per-instance lock), `EpisodeRegistry` (thread-safe map). PIL for PNG encode; `_json_safe` for numpy→JSON in info dicts
- [x] `backend/api.py`: 8 routes — `GET /health`, `GET /checkpoints`, `GET /runs`, `GET /runs/{name}/metrics` (lazy TB EventAccumulator, picks latest log dir per run), `POST /episodes` (201, eager env build), `GET /episodes/{id}/state`, `GET /episodes/{id}/frame` (advances one step, returns image/png), `DELETE /episodes/{id}` (204). Lifespan-managed singletons on `app.state`; CORS pre-wired for Streamlit
- [x] `scripts/serve.py`: argparse CLI — `--host`, `--port`, `--checkpoint-root`, `--tensorboard-root`, `--agent-cache-size`, `--log-level`; same sys.path + PYTHONPATH shim as `train.py`; verified end-to-end via real HTTP boot + clean SIGTERM shutdown
- [x] `tests/test_backend.py`: **24 passed** — `/health`, `/checkpoints` (2), `/runs` (aggregation + config_snapshot detection), `/runs/{name}/metrics` (4: happy + two 404 branches + latest-log-dir selection via synthesized TB events), `POST /episodes` (5: happy + unknown ckpt + bad id format + 422 paths), `/state` (2), `/frame` (3: advances state, PNG Content-Type + magic bytes, idempotent after done), `DELETE` (1), `AgentRegistry` LRU (2), `EpisodeRegistry` semantics (3). Uses `dependency_overrides` + monkey-patching of `EpisodeRuntime` to avoid real stable-retro env construction in CI
- [x] `pyproject.toml`: added `httpx>=0.27` to `[dev]` deps (FastAPI TestClient dependency)
- [ ] `docs/api.md`: deferred to M7. OpenAPI auto-doc at `/docs` and `/redoc` covers the route reference live; checked-in doc only worth adding if the API stabilises and we want a static reference

## Milestone 6 — Frontend (Streamlit)

- [ ] `frontend/app.py`: landing page, backend health probe, theme config
- [ ] `frontend/pages/1_Training.py`: pick a run → live-poll TB metrics via backend → plotly charts
- [ ] `frontend/pages/2_Play.py`: pick a checkpoint → request episode → stream frames into `st.image`
- [ ] `frontend/pages/3_Compare.py`: multi-run overlay of return curves
- [ ] `frontend/components/plots.py`: shared plotly helpers
- [ ] `.streamlit/config.toml`: dark theme, wide layout

## Milestone 7 — Polish

- [ ] `docs/architecture.md`: diagram + module deep-dive
- [ ] `docs/training.md`: how to train, common knobs, expected curves
- [ ] CI: GitHub Actions running lint + tests
- [ ] Pre-commit hooks: ruff + black + mypy (advisory)

---

## Currently in flight

- **Training run v8** — `python scripts/train.py --config configs/ppo.yaml` → `ppo_airstriker_v8`, 4M steps, seed=42. Config: period=18 (3.33 Hz), survival_bonus=0.03, life_loss=-20, ent_coef 0.02→0.001 over 4M (half the annealing rate of v7). Eval determinism fix also active (per-episode seeds in `EvalAndVideoCallback`). TB log will be at `outputs/tensorboard/ppo_airstriker_v8_1`.
- Milestone 6 (Streamlit frontend) queued after v8 is launched.

## Next up (queue)

1. **v8 eval at 4M** — compare head-to-head vs v7 (peak 1248, final 865). Decision threshold: if v8 `eval/mean_length` climbs above 2000 outer steps or return above 1500 by 2M, the survival tuning is working. If return plateaus near v7's level, the ceiling is likely gameplay-structural (limited enemy density in Level1).
2. **Milestone 6** — Streamlit frontend (after v8 launched):
   - `frontend/app.py` — landing page + health probe
   - `frontend/pages/1_Training.py` — run picker, live TB metrics via backend, plotly charts
   - `frontend/pages/2_Play.py` — checkpoint picker → episode → stream frames
   - `frontend/pages/3_Compare.py` — multi-run return overlays (v5–v8 head-to-head)
   - `frontend/components/plots.py` — shared plotly helpers
   - `.streamlit/config.toml` — dark theme, wide layout
3. **Milestone 7** — polish (docs, CI, pre-commit)

## Decisions log

- 2026-05-17 — Chose PPO + SB3 over a from-scratch impl. Rationale: maintainable, debuggable, swappable; we control only the policy net + env. Revisit if SB3's abstractions get in the way.
- 2026-05-17 — Frontend is Streamlit, not React. Rationale: single language, fastest path to a clean dashboard; backend already separated so we can swap frontends without touching core.
- 2026-05-17 — Added `shimmy>=1.3` as a gym↔gymnasium bridge dependency. stable-retro is supposed to be gymnasium-native (≥0.9.2) but the adapter is a cheap insurance policy against version drift.
- 2026-05-17 — Plumbed `info_keys` through `EnvConfig` instead of hardcoding RAM-var names. Lets any stable-retro integration reuse the same shaping code with a YAML-side override.
- 2026-05-17 — stable-retro on macOS Apple Silicon: PyPI arm64 wheels are mislabeled (contain x86_64). Source build also fails because multiple libretro cores ship a vendored zlib whose `zutil.h` defines `fdopen` as `NULL` on any platform with `TARGET_OS_MAC` (i.e. Darwin too, not just Classic Mac OS). Fixed by patching all `zutil.h` copies to also gate on `!defined(__APPLE__)`; automated via [`scripts/install_stable_retro_macos.sh`](scripts/install_stable_retro_macos.sh). All env-layer tests + Airstriker smoke green on macOS 26 + arm64.
- 2026-05-18 — Milestone 2: chose a `policy_kwargs(features_dim)` helper over a custom `ActorCriticPolicy` subclass. Rationale: SB3's `CnnPolicy` already does everything we need once the feature extractor is plugged in; a subclass would only duplicate plumbing. Revisit when a second policy variant (e.g. recurrent) lands. Also chose `typing.Protocol` for `agents/base.py` over an ABC — both SB3 PPO and our `RandomAgent` conform structurally without forcing SB3 into a custom hierarchy.
- 2026-05-18 — macOS auto-hides files in `.venv/lib/.../site-packages/` (sets `UF_HIDDEN` via `xattr com.apple.provenance` + some background daemon). CPython 3.12.5+ skips hidden `.pth` files for security, which breaks the editable install repeatedly even after `chflags nohidden`. Worked around with a top-level `conftest.py` that adds `src/` to `sys.path` directly. Future: investigate whether a launch agent / Spotlight indexer is responsible; for now the conftest is the durable fix for tests, and prod scripts run via `python -m` from the repo root pick up `src/` via cwd.
- 2026-05-18 — **Pivot: project renamed `contra-rl` → `retro-rl`; default target is Airstriker (Genesis), not Contra (NES).** Rationale: no legal path to a Contra ROM available to the user; Airstriker ships free with stable-retro and the entire architecture is already game-agnostic via `info_keys` + config-driven reward shaping. Full rename executed (Python pkg `contra_rl`→`retro_rl`, `ContraCNN`→`RetroCNN`, `make_contra_env`→`make_retro_env`, `configs/env.yaml` now Airstriker, `total_timesteps` reduced 10M→2M for the simpler target). Repo directory + git remote name left unchanged (orthogonal to the code rename; user can update those externally if/when desired).
- 2026-05-18 — M3: trainer always uses `SubprocVecEnv` (even at `n_envs=1`), not `DummyVecEnv`. Rationale: stable-retro hard-limits one emulator per process; if the train emulator lives in the main process, the eval-callback's lazy emulator construction fails with `RuntimeError("Cannot create multiple emulator instances per process")`. SubprocVecEnv puts the train emulator(s) in workers, leaving the main process free for eval. IPC cost at `n_envs=1` is negligible vs env stepping.
- 2026-05-18 — M3: chose thin custom callbacks wrapping `BaseCallback` over SB3's built-in `CheckpointCallback`/`EvalCallback`. Rationale: atomic save (tmp + `os.replace`) + JSON sidecar metadata + best-tracking owned by our `CheckpointManager` (decoupled from SB3 internals). Cost is ~150 LOC of glue; benefit is reproducibility and a clean handoff to the backend in M5 (sidecars are the API).
- 2026-05-18 — M3: known cosmetic issue — stable-retro subproc workers raise `AttributeError: 'CocoaAlternateEventLoop' object has no attribute 'platform_event_loop'` during shutdown (pyglet teardown on macOS). Happens AFTER training success is logged and parent exits 0; does not affect outputs. Upstream pyglet/stable-retro interaction; documented and deferred. Revisit if it ever causes a non-zero exit or blocks CI.
- 2026-05-18 — M3: known stale editable install — `.venv/lib/.../site-packages/__editable__.contra_rl-0.0.1.pth` lingers from pre-rename install and points at a non-existent `contra-rl/src` path. Tests work via `conftest.py` (`sys.path` shim); scripts need `PYTHONPATH=src` until `pip install -e .` is re-run. Fix is one command but orthogonal to M3 — flagged for the user to refresh at convenience.
- 2026-05-18 — **v5 fire mechanic: `AutoFireWrapper` (tap-fire)** (after v3/v4 zero-kill plateau). v3 ran to 200K with `Discrete(9)` always-fire combos (each combo set `B=1`); eval/mean_return dropped from -29.90 (100K, anomalous long episode) to -35.75 (200K) and the stochastic rollout sat flat at the no-kill survival floor. Diagnosis followed two false leads (passive corner-hiding via `survival_bonus`, then clip-ceiling masking `score_delta`) before user observation of the eval video — "aircraft barely shoots" — flagged a deeper issue. **True root cause**: Airstriker fires only on the *rising edge* of button B; holding B continuously fires exactly one bullet at the press event, then nothing. With every combo's `B=1`, the agent was holding B for the entire episode — one bullet per life. Verified empirically with [`scripts/diagnose_fire_button.py`](scripts/diagnose_fire_button.py): held B → score 0 over 600 frames; tap pattern (1 on, 3 off) → consistent score. This also explains why v2 worked despite the Bernoulli threshold — stochastic sampling at P(B=1)≈0.14 naturally toggled B and produced press-release cycles. Fix: new [`AutoFireWrapper`](src/retro_rl/env/wrappers.py) wraps the raw env (innermost in the stack) and overrides the fire bit on a per-emulator-frame schedule (1 frame on, 3 off, period=4) regardless of what the policy or DiscreteActionWrapper emits. Because it wraps the raw env, it sees every frame inside `ActionRepeat`'s skip loop — tap cadence is independent of action_repeat. Combos updated to encode movement only (B=0 throughout, 9 directions); reward shaping reverted to v2-style (`score_delta=1.0`, `clip=[-50, 10]`, `survival_bonus=0.01`) since the score signal now actually flows. Implementation: ~30 LOC wrapper + `AutoFireConfig` pydantic model + `EnvConfig.auto_fire: AutoFireConfig | None`. Tests: 8 new (7 for `AutoFireWrapper` + composition, 1 for `AutoFireConfig` validation), full suite 31 env tests passing. Integration smoke: random movement actions now score 20 in 600 frames vs 0 in v3/v4. v1–v4 artifacts preserved for ablation comparison.
- 2026-05-18 — **v3 action space: `Discrete(9)` always-fire** (after v2 Bernoulli-threshold deadlock). v2 ran to 1.66M steps with the new reward shaping; stochastic rollout reached 1174 return (≈119 kill-events/ep) but deterministic eval froze at exactly -35.75 for 8 consecutive checkpoints (1.0M→1.6M). Root cause: SB3 PPO models `MultiBinary(12)` as 12 independent Bernoulli heads; the deterministic policy fires button B only when `P(B=1) > 0.5`. Backing out from stochastic stats, the policy converged to `P(B=1) ≈ 0.14` — enough for stochastic exploration to discover firing but below the deterministic threshold. Fix: new [`DiscreteActionWrapper`](src/retro_rl/env/wrappers.py) maps `Discrete(N)` → `MultiBinary(12)` via a config-driven combo table. v3 uses 9 actions (`B` + 8-way movement, always-fire), encoding the Airstriker-specific prior that unlimited ammo + hard fire-rate cap → optimal play is always-firing. The policy head is now categorical; deterministic argmax selects fire+move directly without any probability threshold. Implementation: ~50 LOC wrapper + `EnvConfig.action_combos: list[list[int]] | None` field; same reward shaping as v2 (it was correct), same PPO hyperparams. Tests: 6 new (5 for `DiscreteActionWrapper`, 1 for config validation); full suite 99 passing. v1 + v2 artifacts preserved for ablation comparison. **NB: This decision turned out to be only half-right — the "always-fire" intent was correct, but the implementation (hardcoded `B=1`) didn't account for tap-fire semantics; see the v5 entry above.**
- 2026-05-18 — **Reward shaping v2** (after v1 deterministic-policy collapse). At 665K steps of `ppo_airstriker` (v1), `rollout/ep_rew_mean` climbed to ~290 but `eval/mean_return` was pinned at -10.0 with `std=0.0` across 6 checkpoints. Eval videos confirmed the argmax policy moved left and died. Root cause: with `end_on_life_lost: true` + `clip: [-10, 10]` + `life_loss=-25` + `death=-100`, every dying episode returned exactly -10.0 (the clip floor), giving the deterministic policy zero gradient signal about *when* it died. Fixes: (a) added `RewardConfig.survival_bonus` (per-step positive reward on non-terminal frames) — small code change in [`reward_shaping.py`](src/retro_rl/env/reward_shaping.py) + new field in [`config.py`](src/retro_rl/utils/config.py); (b) `end_on_life_lost: false` so the 3-life game staggers life-loss penalties; (c) `clip: [-50.0, 10.0]` so death (-40 combined) actually registers; (d) `ent_coef: 0.01 → 0.02` to slow mean-action collapse; (e) softened `life_loss=-10`, `death=-30` to fit under the new clip. New `run_name=ppo_airstriker_v2`; v1 artifacts preserved for side-by-side comparison.
- 2026-05-19 — **v6 ent_coef linear schedule** (after v5 entropy stagnation). v5 (constant `ent_coef=0.02`) peaked at `eval/mean_return=244` at 1.4M then *de-committed*: by 2M, return regressed to 214 and `train/approx_kl → 0.0000` (gradients vanished). Top-2 action mass dropped from 75% (LEFT+RIGHT sweep) at 1.4M to 51% (diffuse across 5 actions) at 2M. Diagnosis: constant `ent_coef` against Discrete(9) max entropy `ln(9) ≈ 2.197` keeps the policy near-uniform once advantages shrink — the entropy regulariser becomes the dominant gradient term and drags the policy back toward uniform. Fix: new [`EntCoefLinearSchedule`](src/retro_rl/training/callbacks.py) callback mutates `model.ent_coef` on rollout_end so PPO sees the annealed value at every gradient pass; reads `model.num_timesteps` directly so it's robust to which lifecycle hook fires. v6 anneals 0.02 → 0.001 over total_timesteps. Result (at 1.6M, before being killed for v7): mean return 222 vs v5's 193; peak 275 vs 244; kills 17 vs 11 at best; episode length 635 vs 402 (+23% survival); `approx_kl` stayed nonzero at 0.0004 vs v5's 0.0000. The schedule worked, but the underlying fire-rate bug (next entry) ceilinged absolute returns. New config field: `PPOHyperparams.ent_coef_final: float | None = None` (None preserves the old constant behaviour). Tests: 4 new for the callback (endpoints + past-end clamping, linear midpoint, TB logging, invalid args); full training suite 18 tests passing.
- 2026-05-19 — **v8 config: period=18, survival-weighted rewards, 4M steps** (after v7 results). Three changes: (1) `auto_fire.period=18` (3.33 Hz, slightly higher than v7's 2.5 Hz; user observation that v7 fire rate was acceptable and the real gap was survival); (2) reward rebalanced toward survival: `survival_bonus 0.01→0.03` (+3×), `life_loss -10→-20` (×2), combined worst-case per-step still -49.97 within clip floor; (3) `total_timesteps=4M` so ent_coef anneals at half the v7 rate (≈0.010 at 2M vs 0.001 at 2M in v7), addressing the approx_kl→0 collapse observed late in v7. Eval determinism fix also shipped: `EvalAndVideoCallback` now passes `seed=eval_seed+ep_i` per episode (A1 variant); for Airstriker's deterministic save-state, std_return will remain 0 until sticky_action_prob or deterministic=False is added — documented in code.
- 2026-05-19 — **v7 results: period=24 confirmed saturation hypothesis** (completed 2026-05-19, 2M steps). Peak `eval/mean_return=1248` @ 1.5M vs v6 peak 275 (+4.5×). Peak `eval/mean_length=1820` outer steps vs v5/v6 ~420–550 (+3.3×) — agent now survives most of Level 1 rather than dying in the first ~5s. Eval curve: monotone rise 100K(63)→1200K(1220), then volatile: 1300K(814)→1400K(245)→1500K(1248)→1600K(795)→1700K(475)→1800K(974)→1900K(475)→2000K(865). Late variance is uninterpretable until the eval-determinism bug (std=0 throughout) is fixed. `approx_kl → 0.000` at 2M mirrors v5 collapse, suggesting `ent_coef_final=0.001` is too aggressive a floor or 2M is too short for the new difficulty level. Best checkpoint: `outputs/checkpoints/ppo_airstriker_v7/best.zip` (step 1.5M, return 1247.8). v8 direction and eval-bug fix pending user decision.
- 2026-05-19 — **v7 AutoFire period 4 → 24** (after v6 hit the same ~275 ceiling for a different reason). v5 and v6 both topped out because AutoFire at `period=4` (15 Hz) saturates Airstriker's player-bullet sprite array within ~0.6s of each life, after which the game silently drops B presses for the remaining 2-5s. Verified with three diagnostics: [`diagnose_wrapper_fire_trace.py`](scripts/diagnose_wrapper_fire_trace.py) proved the wrapper sends B every 4 emu frames continuously in every life — the game stops responding, not us. [`diagnose_fire_rate_vs_state.py`](scripts/diagnose_fire_rate_vs_state.py) ran a stationary rate sweep at periods 2/4/12/60 and showed the freeze is rate-driven (at 30 Hz the game jams to zero kills, at 1 Hz it scores fine). [`eval_period_sweep.py`](scripts/eval_period_sweep.py) ran an inference-time sweep on v6/best.zip across periods 4/8/12/16/24 — `period=8` produced +77% return (275 → 486) and +55% kills (11 → 17) with no retraining. The user-driven call was to go further to `period=24` (2.5 Hz, every bullet visibly separated end-to-end) — bets that a from-scratch policy can learn deliberate position-then-fire rather than spray, with the upside that fire is visibly active for the entire life so the agent gets to experience deeper parts of Level 1 (v5/v6 training data only ever covered the first ~5s). Carries forward v6's ent_coef schedule. Also fixed a quiet bug along the way: `data.json`'s declared score address is off by 2 bytes (real byte is `0xFF0250`, not `0xFF024E`); diagnostic scripts now read the correct byte (training pipeline was already fine via stable-retro's `info['score']`). v6 killed at 1.6M; v1–v6 artifacts preserved.
