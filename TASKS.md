# TASKS.md — contra-rl

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
- [x] Stub `README.md` with setup, ROM legality note, quickstart
- [x] Default YAML configs (`default.yaml`, `ppo.yaml`, `env.yaml`)

## Milestone 1 — Environment layer ✅

- [x] `utils/config.py`: pydantic models for env/ppo/training config; YAML loader w/ `extends:`
- [x] `utils/seeding.py`: `set_global_seed` (python/numpy/torch/PYTHONHASHSEED)
- [x] `utils/logging.py`: structured logger factory; per-run rotating file
- [x] `env/contra_env.py`: stable-retro factory function; ROM import check w/ actionable error
- [x] `env/wrappers.py`: grayscale, resize 84×84, frame stack 4, action-repeat, max-episode-steps, sticky-action option, end-on-life-lost
- [x] `env/reward_shaping.py`: configurable shaping (score Δ, x-progress, life loss, death, stage clear); pure function over info dict; per-integration `info_keys` override
- [x] `env/__init__.py`: `make_env(config)` + `make_env_fn(config, rank)` public entrypoints
- [x] `tests/test_env.py`: **17 passed, 1 skipped** (Contra smoke gated on user ROM dump)
- [x] `configs/env-airstriker.yaml`: Airstriker-Genesis-v0 stand-in config — smoke test runs end-to-end
- [x] `scripts/install_stable_retro_macos.sh`: source-build script with zlib patch for Apple Silicon
- [x] `scripts/play_random.py`: visual sanity-check CLI — random-action rollouts with viewer window
- [x] `docs/environment.md`: rationale for the macOS arm64 build path

## Milestone 2 — Models + agents

- [ ] `models/cnn.py`: Nature-CNN feature extractor (SB3-compatible)
- [ ] `models/policies.py`: custom `ActorCriticPolicy` wiring CNN → action head
- [ ] `agents/base.py`: minimal `Agent` protocol (`predict`, `save`, `load`)
- [ ] `agents/ppo.py`: thin wrapper around SB3 PPO with our policy + config plumbing
- [ ] `agents/random_agent.py`: uniform-random baseline (for sanity floor)
- [ ] `tests/test_models.py`: forward-pass shape tests on dummy obs
- [ ] `tests/test_agents.py`: save/load roundtrip; predict on random obs

## Milestone 3 — Training pipeline

- [ ] `training/callbacks.py`: TB logging, periodic eval, video recording on eval, checkpoint manager (last-K + best)
- [ ] `training/checkpoint.py`: atomic save (tmp + rename), metadata sidecar (config snapshot, step, eval return)
- [ ] `training/trainer.py`: build env → build agent → fit loop with callbacks → final eval; resume-from-checkpoint
- [ ] `scripts/train.py`: argparse CLI, loads config, dispatches to trainer
- [ ] Smoke: 10k-step run on Stage 1 completes; checkpoint + TB + video produced

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
- [ ] `docs/environment.md`: stable-retro setup, ROM import, reward shaping rationale
- [ ] CI: GitHub Actions running lint + tests (no ROM-dependent tests)
- [ ] Pre-commit hooks: ruff + black + mypy (advisory)

---

## Currently in flight

_None — Milestone 1 landed; ready to pick Milestone 2._

## Next up (queue)

1. Milestone 2 — models + agents (ROM-independent)
2. Milestone 3 — training pipeline (Airstriker now runs locally on Apple Silicon; Contra unblocks once user dumps cart)
3. Milestone 4 — evaluation

## Decisions log

- 2026-05-17 — Chose PPO + SB3 over a from-scratch impl. Rationale: maintainable, debuggable, swappable; we control only the policy net + env. Revisit if SB3's abstractions get in the way.
- 2026-05-17 — Frontend is Streamlit, not React. Rationale: single language, fastest path to a clean dashboard; backend already separated so we can swap frontends without touching core.
- 2026-05-17 — Default target is Stage 1 only. Stages 2–3 are top-down and need separate shaping; defer until Stage 1 trains reliably.
- 2026-05-17 — Added `shimmy>=1.3` as a gym↔gymnasium bridge dependency. stable-retro is supposed to be gymnasium-native (≥0.9.2) but the adapter is a cheap insurance policy against version drift.
- 2026-05-17 — Plumbed `info_keys` through `EnvConfig` instead of hardcoding Contra's RAM-var names. Lets Airstriker (and any future integration) reuse the same shaping code with a YAML-side override.
- 2026-05-17 — stable-retro on macOS Apple Silicon: PyPI arm64 wheels are mislabeled (contain x86_64). Source build also fails because multiple libretro cores ship a vendored zlib whose `zutil.h` defines `fdopen` as `NULL` on any platform with `TARGET_OS_MAC` (i.e. Darwin too, not just Classic Mac OS). Fixed by patching all `zutil.h` copies to also gate on `!defined(__APPLE__)`; automated via [`scripts/install_stable_retro_macos.sh`](scripts/install_stable_retro_macos.sh). 17/17 unit tests + Airstriker smoke now green on macOS 26 + arm64.
