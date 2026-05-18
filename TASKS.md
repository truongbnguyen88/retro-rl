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

- **Full training run in background** — `python scripts/train.py --config configs/ppo.yaml`, 2M steps total. Kicked off 2026-05-18 alongside M5 implementation. `n_envs=8` SubprocVecEnv on Apple Silicon, ~207 fps aggregate → ETA ~2.7h wall. Live progress visible via `GET /runs/ppo_airstriker/metrics` once the backend is up. Producing periodic checkpoints (every 250K steps), eval videos (every 100K), and TB scalar series. At time of M5 landing: ~300K / 2M steps complete; `rollout/ep_rew_mean` climbing (88 → 94 between 100K and 200K); deterministic `eval/mean_return` still pinned at -10 (expected at this stage — argmax policy lags exploration distribution early).
- Milestone 5 landed. Holding before M6 (Streamlit frontend) per direction.

## Next up (queue)

1. Milestone 6 — frontend (Streamlit dashboard against the now-live backend)
2. Decide whether to extend training to 4M based on the 2M run's learning curve
3. Milestone 7 — polish (docs, CI, pre-commit, optional `docs/api.md`)

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
