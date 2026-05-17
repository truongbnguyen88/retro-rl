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

## Milestone 1 — Environment layer

- [ ] `utils/config.py`: pydantic models for env/ppo/training config; YAML loader
- [ ] `utils/seeding.py`: `set_global_seed` (python/numpy/torch/env)
- [ ] `utils/logging.py`: structured logger factory; per-run log file
- [ ] `env/contra_env.py`: stable-retro factory function; ROM import check; surfaces a Gymnasium env
- [ ] `env/wrappers.py`: grayscale, resize 84×84, frame stack 4, action-repeat, max-episode-steps, sticky-action option
- [ ] `env/reward_shaping.py`: configurable shaping (score Δ, x-progress, life loss, death, stage clear); pure function over RAM/info dict
- [ ] `env/__init__.py`: `make_env(config)` public entrypoint
- [ ] `tests/test_env.py`: env factory smoke test (skips if ROM missing), shape/dtype assertions on obs, deterministic seed test

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

_None — Milestone 0 just landed; pick Milestone 1 next._

## Next up (queue)

1. Milestone 1 — env layer (blocks everything else)
2. Milestone 2 — models + agents
3. Milestone 3 — training pipeline

## Decisions log

- 2026-05-17 — Chose PPO + SB3 over a from-scratch impl. Rationale: maintainable, debuggable, swappable; we control only the policy net + env. Revisit if SB3's abstractions get in the way.
- 2026-05-17 — Frontend is Streamlit, not React. Rationale: single language, fastest path to a clean dashboard; backend already separated so we can swap frontends without touching core.
- 2026-05-17 — Default target is Stage 1 only. Stages 2–3 are top-down and need separate shaping; defer until Stage 1 trains reliably.
