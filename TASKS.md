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

## Milestone 3 — Training pipeline

- [ ] `training/callbacks.py`: TB logging, periodic eval, video recording on eval, checkpoint manager (last-K + best)
- [ ] `training/checkpoint.py`: atomic save (tmp + rename), metadata sidecar (config snapshot, step, eval return)
- [ ] `training/trainer.py`: build env → build agent → fit loop with callbacks → final eval; resume-from-checkpoint
- [ ] `scripts/train.py`: argparse CLI, loads config, dispatches to trainer
- [ ] Smoke: 10k-step run on Airstriker completes; checkpoint + TB + video produced

## Milestone 4 — Evaluation

- [ ] `evaluation/evaluator.py`: rollout N deterministic episodes, return metrics dict
- [ ] `evaluation/metrics.py`: mean/std return, mean episode length, stage-clear rate, deaths-per-episode
- [ ] `utils/video.py`: write mp4 from frame list (imageio-ffmpeg)
- [ ] `scripts/evaluate.py`: CLI; emits JSON + optional video

## Milestone 5 — Backend (FastAPI)

- [ ] `backend/models.py`: pydantic request/response schemas
- [ ] `backend/inference.py`: lazy-loaded agent registry (path → loaded model)
- [ ] `backend/api.py`: routes — `GET /health`, `GET /checkpoints`, `POST /episodes` (start), `GET /episodes/{id}/frame`, `GET /episodes/{id}/state`, `GET /runs`, `GET /runs/{id}/metrics`
- [ ] `scripts/serve.py`: uvicorn launcher
- [ ] `tests/test_backend.py`: TestClient hits each route; mocks the agent
- [ ] `docs/api.md`: route reference (generated from OpenAPI is fine)

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

_None — Milestone 2 landed + project pivoted to Airstriker / `retro-rl`. Ready to pick Milestone 3._

## Next up (queue)

1. Milestone 3 — training pipeline (Airstriker smoke target)
2. Milestone 4 — evaluation
3. Milestone 5 — backend

## Decisions log

- 2026-05-17 — Chose PPO + SB3 over a from-scratch impl. Rationale: maintainable, debuggable, swappable; we control only the policy net + env. Revisit if SB3's abstractions get in the way.
- 2026-05-17 — Frontend is Streamlit, not React. Rationale: single language, fastest path to a clean dashboard; backend already separated so we can swap frontends without touching core.
- 2026-05-17 — Added `shimmy>=1.3` as a gym↔gymnasium bridge dependency. stable-retro is supposed to be gymnasium-native (≥0.9.2) but the adapter is a cheap insurance policy against version drift.
- 2026-05-17 — Plumbed `info_keys` through `EnvConfig` instead of hardcoding RAM-var names. Lets any stable-retro integration reuse the same shaping code with a YAML-side override.
- 2026-05-17 — stable-retro on macOS Apple Silicon: PyPI arm64 wheels are mislabeled (contain x86_64). Source build also fails because multiple libretro cores ship a vendored zlib whose `zutil.h` defines `fdopen` as `NULL` on any platform with `TARGET_OS_MAC` (i.e. Darwin too, not just Classic Mac OS). Fixed by patching all `zutil.h` copies to also gate on `!defined(__APPLE__)`; automated via [`scripts/install_stable_retro_macos.sh`](scripts/install_stable_retro_macos.sh). All env-layer tests + Airstriker smoke green on macOS 26 + arm64.
- 2026-05-18 — Milestone 2: chose a `policy_kwargs(features_dim)` helper over a custom `ActorCriticPolicy` subclass. Rationale: SB3's `CnnPolicy` already does everything we need once the feature extractor is plugged in; a subclass would only duplicate plumbing. Revisit when a second policy variant (e.g. recurrent) lands. Also chose `typing.Protocol` for `agents/base.py` over an ABC — both SB3 PPO and our `RandomAgent` conform structurally without forcing SB3 into a custom hierarchy.
- 2026-05-18 — macOS auto-hides files in `.venv/lib/.../site-packages/` (sets `UF_HIDDEN` via `xattr com.apple.provenance` + some background daemon). CPython 3.12.5+ skips hidden `.pth` files for security, which breaks the editable install repeatedly even after `chflags nohidden`. Worked around with a top-level `conftest.py` that adds `src/` to `sys.path` directly. Future: investigate whether a launch agent / Spotlight indexer is responsible; for now the conftest is the durable fix for tests, and prod scripts run via `python -m` from the repo root pick up `src/` via cwd.
- 2026-05-18 — **Pivot: project renamed `contra-rl` → `retro-rl`; default target is Airstriker (Genesis), not Contra (NES).** Rationale: no legal path to a Contra ROM available to the user; Airstriker ships free with stable-retro and the entire architecture is already game-agnostic via `info_keys` + config-driven reward shaping. Full rename executed (Python pkg `contra_rl`→`retro_rl`, `ContraCNN`→`RetroCNN`, `make_contra_env`→`make_retro_env`, `configs/env.yaml` now Airstriker, `total_timesteps` reduced 10M→2M for the simpler target). Repo directory + git remote name left unchanged (orthogonal to the code rename; user can update those externally if/when desired).
