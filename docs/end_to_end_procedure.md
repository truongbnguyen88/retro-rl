# retro-rl — End-to-End Procedure Guide

A deep walkthrough of the entire project, Milestone 0 → 6. Milestones 1–4 are
expanded into full RL theory: what each concept means, why it exists, and how
the code implements it. Milestones 5–6 document the service and UI layers.

---

## Table of Contents

1. [Milestone 0 — Repo Scaffolding](#milestone-0--repo-scaffolding)
2. [Milestone 1 — Environment Layer](#milestone-1--environment-layer)
3. [Milestone 2 — Models + Agents](#milestone-2--models--agents)
4. [Milestone 3 — Training Pipeline](#milestone-3--training-pipeline)
5. [Milestone 4 — Evaluation](#milestone-4--evaluation)
6. [Milestone 5 — Backend (FastAPI)](#milestone-5--backend-fastapi)
7. [Milestone 6 — Frontend (Streamlit)](#milestone-6--frontend-streamlit)
8. [Appendix: v10 — RecurrentPPO + LSTM (discontinued)](#appendix-v10--recurrentppo--lstm-discontinued)
9. [Appendix: v12 — Temporal Attention (transformer over the frame window)](#appendix-v12--temporal-attention-transformer-over-the-frame-window)

---

## Milestone 0 — Repo Scaffolding

### 0.1 Project goal

Train a reinforcement learning agent to play **Airstriker** (Sega Genesis), a
vertical-scroll shooter that ships freely with the `stable-retro` library —
no ROM acquisition needed. The full architecture:

```
stable-retro emulator
    └─ wrapper stack (preprocessing + reward shaping)
        └─ VecEnv (8 parallel envs for rollout throughput)
            └─ PPO (on-policy RL algorithm)
                └─ Actor-Critic policy (IMPALA CNN backbone)

Training artifacts: checkpoints / TensorBoard logs / eval videos
    └─ FastAPI backend (REST API)
        └─ Streamlit frontend (dashboard + agent play viewer)
```

### 0.2 Directory layout

| Path | Responsibility |
|------|---------------|
| `src/retro_rl/env/` | Env factory, wrappers, reward shaping |
| `src/retro_rl/models/` | CNN feature extractors |
| `src/retro_rl/agents/` | PPO factory, random baseline, agent Protocol |
| `src/retro_rl/training/` | Trainer, callbacks, checkpoint manager |
| `src/retro_rl/evaluation/` | Deterministic rollouts, metrics, video |
| `src/retro_rl/backend/` | FastAPI app (serves training artifacts) |
| `src/retro_rl/utils/` | Config, logging, seeding, video |
| `frontend/` | Streamlit dashboard (HTTP-only, no shared imports) |
| `configs/` | YAML hyperparameter files |
| `scripts/` | CLI entrypoints: `train.py`, `evaluate.py`, `play.py`, `serve.py` |
| `outputs/` | Run artifacts (gitignored): checkpoints, TB logs, videos |

**Dependency rule:** the table is acyclic top-to-bottom. `utils/` is a pure
leaf. The frontend never imports backend code — HTTP only.

### 0.3 Config system

All configuration is **YAML → pydantic**. Defined in
[`src/retro_rl/utils/config.py`](../src/retro_rl/utils/config.py).

```
configs/
  env.yaml        ← EnvConfig (game, wrappers, reward shaping)
  ppo.yaml        ← TrainConfig (EnvConfig reference + PPO hyperparams)
  env_v9.yaml     ← extends: env.yaml  (overrides action_repeat, auto_fire.period)
  ppo_v9.yaml     ← extends: ppo.yaml  (overrides LR, extractor, normalize_reward)
```

**Deep-merge inheritance** via `extends:` allows version-specific overrides
without duplicating unmodified knobs. The loader
(`load_train_config` / `load_env_config`) handles chains recursively and
replaces the string `env_config:` path with a parsed `EnvConfig` instance.

Key pydantic models:

| Model | Fields |
|---|---|
| `RewardConfig` | `score_delta`, `x_progress`, `life_loss`, `death`, `stage_clear`, `survival_bonus`, `clip` |
| `AutoFireConfig` | `button_index`, `period` |
| `EnvConfig` | game/state/scenario, wrappers (grayscale, resize, frame_stack, action_repeat), reward, action_combos, auto_fire |
| `PPOHyperparams` | lr, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm |
| `TrainConfig` | run_name, seed, env, n_envs, algorithm, features_extractor, features_dim, normalize_reward, ppo, eval, checkpoint |

All models use `extra="forbid"` so typos in YAML are caught at load time, not
silently ignored at runtime.

---

## Milestone 1 — Environment Layer

This milestone establishes the **interface between the Airstriker emulator and
the RL agent**. Understanding it requires knowing what a Markov Decision Process
is, why raw game observations are unsuitable for learning, and what each wrapper
does to fix that.

### 1.1 RL fundamentals: MDP and observations

Reinforcement learning operates on a **Markov Decision Process (MDP)**:

```
State s ─► Agent picks action a ─► Environment transitions to s' and emits reward r
```

The formal components:

| Symbol | Meaning |
|---|---|
| `S` | State space — all possible game states |
| `A` | Action space — all moves available to the agent |
| `r(s, a)` | Reward function — signal telling the agent how good an action was |
| `p(s' | s, a)` | Transition function — probability of next state |
| `γ ∈ [0, 1)` | Discount factor — how much future rewards are worth today |

The **Markov property** — that the future depends only on the current state, not
history — is critical. It means the agent only needs the current state to make
optimal decisions.

**Problem for games:** Airstriker is actually a **Partially Observable MDP
(POMDP)**. The true Markov state lives in the emulator's RAM (enemy positions,
bullet velocities, spawn timers). The agent sees only **pixels** (the rendered
frame). A single frame is not Markov: you cannot tell from one image whether a
bullet is moving up or down. The wrapper stack exists to convert the raw
emulator output into something closer to a Markov state.

#### The discount factor γ in depth

`γ ∈ [0, 1)` controls how much the agent values future reward relative to
immediate reward. A reward `r` received `k` steps in the future is worth
`γ^k · r` today. The objective the agent maximizes is the **discounted return**:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ_{k=0}^∞ γ^k · r_{t+k}
```

Two reasons γ < 1:
1. **Mathematical:** for infinite-horizon problems, an undiscounted sum can
   diverge. Discounting guarantees `G_t` is finite (geometric series).
2. **Practical:** distant rewards are uncertain (the policy might die first,
   the world might change). Discounting expresses "a bird in the hand."

**Effective horizon.** The useful intuition is that γ defines a soft time
horizon of roughly `1/(1−γ)` steps — beyond that, rewards are discounted into
near-irrelevance:

| γ | `1/(1−γ)` | Effective horizon |
|---|---|---|
| 0.9 | 10 | very short-sighted |
| 0.99 (v9) | 100 | ~100 decision steps ahead |
| 0.999 | 1000 | very far-sighted, hard to learn |

At v9's `γ=0.99` and `action_repeat=8`, 100 decision steps ≈ 800 emulator
frames ≈ **13 seconds** of game time. The agent optimizes over roughly a
13-second horizon — long enough to value "dodge now to survive the next wave,"
short enough that credit assignment stays tractable. Too high a γ and the
credit-assignment problem (which of the last 1000 actions caused this reward?)
becomes intractably noisy; too low and the agent can't plan past the immediate
threat.

#### Terminology you must keep straight

These terms get conflated constantly. Pinning them down now prevents confusion
throughout:

| Term | Meaning | In retro-rl |
|---|---|---|
| **State** `s` | The true, complete description of the world (Markov) | The emulator's RAM — the agent never sees this directly |
| **Observation** `o` | What the agent actually receives | The `(84, 84, 4)` frame stack |
| **Action** `a` | The agent's decision | One of 9 movement combos (`Discrete(9)`) |
| **Reward** `r` | Scalar signal for a single step | Shaped reward after clip, one number per decision |
| **Return** `G` | Discounted sum of *future* rewards from a step | `Σ γ^k r_{t+k}` — what we maximize |
| **Transition** | One `(o, a, r, o', done)` tuple | One row in the rollout buffer |
| **Episode** | One play-through from reset to terminal/truncation | One life (with `end_on_life_lost=True`) |
| **Trajectory** | The ordered sequence of transitions in an episode | `τ = (o₀,a₀,r₀, o₁,a₁,r₁, ...)` |
| **Rollout** | A fixed-length batch of transitions collected for one update | 1024 transitions (8 envs × 128 steps) — may span several episodes |

Because Airstriker is a POMDP, the literature's `V(s)` is really `V(o)` here —
the value of the *observation*, since pixels are all the network has. We write
`V(s)` loosely throughout, following convention.

### 1.2 The stable-retro library

`stable-retro` is a fork of OpenAI Gym's Retro library that emulates classic
consoles (Genesis, NES, SNES, etc.) and exposes them as Gymnasium environments.

```python
import retro
env = retro.make("Airstriker-Genesis-v0", state="Level1", scenario="scenario")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

Raw output:
- **Observation:** `(224, 320, 3)` uint8 RGB frame — raw Genesis display
- **Action space:** `MultiBinary(12)` — 12 Genesis buttons as independent bits (B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z)
- **Reward:** native game score delta (raw integer, can be large / sparse)
- **Info dict:** RAM-derived variables from the integration's `data.json` — `score`, `lives`, etc.

### 1.3 ROM check and gym bridge

[`env/retro_env.py`](../src/retro_rl/env/retro_env.py) — `make_retro_env(cfg, seed, record_dir, render_mode)`

Before building the env, `_check_rom_imported(game)` calls
`retro.data.get_romfile_path(game)`. For Airstriker this is a no-op (ROM ships
with `stable-retro`). For user-supplied games it gives an actionable error
message that includes the exact import command to run.

`_to_gymnasium(env)` then wraps the result in a `shimmy.GymV21CompatibilityV0`
shim if needed — stable-retro >= 0.9.2 returns a gymnasium-native env, but the
shim guards against version drift.

The public function `make_retro_env` calls `apply_wrappers(env, cfg)` which
composes the full wrapper stack (§1.5).

### 1.4 Action space problem: MultiBinary vs Discrete

**Why `MultiBinary(12)` is bad for PPO.** SB3's PPO models a `MultiBinary(k)`
space as **k independent Bernoulli heads** — one sigmoid output per button.
The deterministic policy fires button `i` only when `P(button_i = 1) > 0.5`.

For Airstriker's fire button, the reward for firing is **sparse and delayed**:
kill an enemy → score → shaped reward. The reward for survival is **dense and
immediate** (per-step survival bonus). The policy converges to
`P(fire=1) ≈ 0.14` — enough for the stochastic policy to occasionally fire,
but below 0.5 so the **deterministic eval policy never fires at all**.

This was observed empirically in v2: stochastic rollout return ≈ 1174, but
deterministic eval frozen at -35.75 for 700K steps.

**Fix: `DiscreteActionWrapper`** (in
[`env/wrappers.py`](../src/retro_rl/env/wrappers.py)):

Maps `Discrete(N)` → `MultiBinary(12)` via a config-driven combo table. v9
uses 9 combos — no-op + 8 movement directions, all with fire bit = 0 (firing
is handled by `AutoFireWrapper`). The PPO head is now a **categorical
distribution**; the deterministic policy uses `argmax` over 9 choices directly.
No probability threshold for individual buttons.

```python
class DiscreteActionWrapper(gym.ActionWrapper):
    def action(self, action) -> np.ndarray:
        return self._combos[int(action)].copy()
```

### 1.5 The wrapper stack

Applied by `apply_wrappers(env, cfg)` in innermost-first order:

```
raw retro env  (MultiBinary(12), 224×320×3 uint8)
   └─ [1] AutoFireWrapper       # override fire bit at frame level
      └─ [2] DiscreteActionWrapper  # Discrete(9) → MultiBinary(12)
         └─ [3] ActionRepeat     # repeat action k frames, max-pool last 2
            └─ [4] StickyAction  # optional: repeat prev action with prob p
               └─ [5] EndOnLifeLost  # terminate on life decrement
                  └─ [6] RewardShapingWrapper  # additive shaping terms
                     └─ [7] GrayscaleResize    # 224×320×3 → 84×84×1
                        └─ [8] FrameStack      # (84,84,4) temporal context
                           └─ [9] TimeLimit    # max_episode_steps cutoff
```

#### [1] AutoFireWrapper

```python
class AutoFireWrapper(gym.Wrapper):
    def step(self, action):
        action = np.array(action, dtype=np.int8, copy=True)
        action[self._fire_idx] = 1 if (self._counter % self._period) == 0 else 0
        self._counter += 1
        return self.env.step(action)
```

**Why it exists:** Airstriker fires only on the **rising edge** of button B —
one bullet per press, not per held frame. v3/v4 training failed because all
combos had `B=1`, so the agent held B continuously and emitted one bullet per
life.

This wrapper sits **innermost** (wraps the raw env), so it sees every emulator
frame, including the ones inside an `ActionRepeat` skip loop. The fire cadence
is therefore **independent of action_repeat**.

The fire bit emitted by the policy is completely ignored. The agent's action
space becomes **movement-only** — where to fly. Firing happens at the
configured cadence (v9: `period=12`, ~5 Hz).

#### [2] DiscreteActionWrapper

Translates `Discrete(N)` integer → `MultiBinary(12)` button vector via
`self._combos[action]`. The combo table is config-driven in `env.yaml`:

```yaml
action_combos:
  - [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # no-op
  - [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # UP
  - [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # DOWN
  # ... 6 more directional combos
```

#### [3] ActionRepeat (frame-skip)

```python
class ActionRepeat(gym.Wrapper):
    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # buffer last 2 frames
            total_reward += float(reward)
            if terminated or truncated: break
        max_frame = self._obs_buf.max(axis=0)  # element-wise max
        return max_frame, total_reward, terminated, truncated, info
```

**Why action repeat matters for RL:**

- The policy makes one decision every `k` emulator frames instead of every
  frame. At `action_repeat=8` and Genesis 60 FPS: the agent decides 7.5 times
  per second — much more aligned with human reaction speed.
- **Reduces the effective time horizon:** the agent sees 1/k as many
  observations per episode, making credit assignment easier (shorter sequences
  of decisions to credit).
- **Accumulates reward** across skipped frames so the agent sees the full
  consequence of its decision.
- **Max-pools the last 2 frames** to suppress Genesis sprite flickering —
  some sprites only appear on alternating frames due to hardware sprite limits.

**v9 choice — `action_repeat=8`:** Halved from v8's 4. Eliminated the
"shaking" 2-3-frame LEFT↔RIGHT oscillation visible in v8 replay, where logits
flipped every other frame because each single-frame decision was too cheap.
At AR=8 each decision commits for ~133ms.

#### [4] StickyAction (optional)

With probability `p`, repeat the **previous** action instead of the new one.
Adds stochasticity to the environment's dynamics, making the env non-deterministic
even from the same save-state. Used to get non-zero `std_return` during eval.
v9 uses `p=0` (disabled).

#### [5] EndOnLifeLost

```python
class EndOnLifeLost(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get(self._key)
        if lives is not None and self._prev_lives is not None and lives < self._prev_lives:
            terminated = True
        self._prev_lives = lives
        return obs, reward, terminated, truncated, info
```

**Why early termination helps credit assignment:** Without this, the agent
experiences its 3 lives in one episode. The death of life 1 → death penalty →
life 2 begins. The agent must credit its actions in life 1 to the death
penalty it receives, across the gap of life 2's gameplay. This is extremely
hard: the credit has to propagate backward through an entire extra life.

With `EndOnLifeLost=True`, each life is its own episode. The death penalty
arrives at the end of the episode where the fatal decision was made → clean,
tight credit assignment.

The inner env keeps running (not reset); SB3's vec env calls `reset()` at the
next episode boundary. The life counter continues across the underlying episode.

#### [6] RewardShapingWrapper

Wraps [`env/reward_shaping.py`](../src/retro_rl/env/reward_shaping.py). Adds
configurable shaping terms on top of the native game reward.

**Why reward shaping is necessary:**

The native Airstriker reward is the raw score delta — purely from killing
enemies. This signal is:
- **Sparse:** early in training the agent dies before ever hitting anything.
  Zero kills = zero positive gradient = no learning direction.
- **Delayed:** the connection between "dodge correctly and stay alive" and
  "eventually kill enemies" is many steps long.

Without shaping, the agent receives the same reward whether it died on step 1
or step 10,000 — the gradient cannot distinguish good survival from bad.

**The shaping function** (`shape_reward` in reward_shaping.py):

```python
r = 0.0
r += cfg.score_delta * (score - prev_score)   # kill reward
r += cfg.x_progress * max(0, dx)              # forward progress (disabled for vertical scroll)
r += cfg.life_loss * (prev_lives - lives)     # life lost penalty
r += cfg.stage_clear                          # if stage clear flag fired
if terminated:
    r += cfg.death                             # final death penalty
else:
    r += cfg.survival_bonus                   # per-step: alive = good
r = clip(r, cfg.clip[0], cfg.clip[1])
```

**The `ShapingState` tracks deltas across steps:**

```python
@dataclass
class ShapingState:
    prev_score: int | None = None
    prev_x: int | None = None
    prev_lives: int | None = None
    cumulative_shaped: float = 0.0
```

The score *delta* (`score - prev_score`) is what matters, not the absolute
score. Between steps: if the agent killed an enemy, the score jumps → positive
reward. If not, delta = 0 → no reward.

**Survival bonus rationale:** A small per-step positive reward for staying
alive. This creates a **dense gradient signal** even when no enemies are killed.
The agent immediately learns that dying is bad (bonus disappears, death penalty
hits) vs. surviving (bonus accumulates).

**Clip:** All shaping is clipped to `[lo, hi]`. Without clipping, a single
massive score event could swamp all other gradients and destabilize learning.
The clip keeps the reward in a bounded range that the value function can learn
to predict reliably.

**Missing info key behavior:** If the integration's `data.json` doesn't expose
a key (e.g. `stage_clear` for Airstriker), the function logs one warning and
contributes zero — it never crashes.

**v9 reward config (from `configs/env_v9.yaml`):**

```yaml
reward:
  score_delta: 1.0
  x_progress: 0.0      # disabled — vertical scroll game
  life_loss: -20.0
  death: -30.0
  stage_clear: 200.0
  survival_bonus: 0.03
  clip: [-50.0, 10.0]
```

#### [7] GrayscaleResize

```python
class GrayscaleResize(gym.ObservationWrapper):
    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self._size, interpolation=cv2.INTER_AREA)
        return resized[:, :, None].astype(np.uint8)
```

**`(224, 320, 3)` → `(84, 84, 1)`**

Two reasons:
1. **Color is mostly irrelevant.** Airstriker's relevant information — ship
   position, bullet positions, enemy patterns — is encoded in shape and
   motion, not color. Grayscale removes 2/3 of input channels with minimal
   information loss.
2. **84×84 is the Atari/DQN standard.** Smaller than 224×320 by 32×, reducing
   CNN computation dramatically while retaining enough spatial detail for
   policy decisions.

`cv2.INTER_AREA` downsampling is chosen over bilinear/nearest: it correctly
anti-aliases by averaging pixel values in each target cell, preserving edge
clarity at small scales.

The output is kept as `uint8` (0–255) to save memory. Normalization to float
[0, 1] happens inside the CNN's `forward` method — the rollout buffer stores
uint8, reducing memory by 4×.

#### [8] FrameStack

```python
class FrameStack(gym.Wrapper):
    def reset(self, ...):
        obs, info = self.env.reset(...)
        for _ in range(self._n):
            self._frames.append(obs)     # fill buffer with n copies of first frame
        return self._stack(), info

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1)   # (H, W, C*n)
```

**Why frame stacking restores the Markov property:**

A single 84×84 grayscale frame is **not Markov** — it has no velocity
information. You cannot determine from one frame:
- Is that bullet moving toward me or away?
- Is an enemy moving left or right?

With `frame_stack=4`, the agent receives 4 consecutive frames stacked along
the channel axis → shape `(84, 84, 4)`. The network can compute
differences across the 4 frames, recovering velocity and direction. This turns
the observation into a **good approximation of the Markov state**.

On reset, the buffer is filled with the first frame repeated 4 times — this
is a standard initialization choice (alternatives: zeros, but they look like
glitchy frames and hurt early learning).

SB3 later applies `VecTransposeImage` to convert `(H, W, C)` → `(C, H, W)`
channels-first format for PyTorch convolutions.

#### [9] TimeLimit

`gym.wrappers.TimeLimit(env, max_episode_steps=4500)` — truncates episodes
at 4500 outer steps (= 4500 × `action_repeat` emulator frames). This is a
**truncation**, not a termination: the episode ends due to time limit, not
because the agent died. The distinction matters for advantage estimation
(see §3.4).

### 1.5.1 Timescales — the units that confuse everyone

The pipeline operates at several nested timescales. "Step" means different
things at different layers, which is a frequent source of confusion. Here is
the exact unit ladder for v9:

| Timescale | Duration | Relation | What happens |
|---|---|---|---|
| **Emulator frame** | 1/60 s ≈ 16.7 ms | base unit | One Genesis frame rendered; `AutoFireWrapper` decides the fire bit here |
| **Decision step** (outer step) | 8 frames ≈ 133 ms | `= action_repeat` frames | One policy decision; `ActionRepeat` holds the action, sums reward, max-pools the last 2 frames |
| **Fire pulse** | 12 frames ≈ 200 ms | `= auto_fire.period` frames | One bullet fired (~5 Hz), *independent* of decision step |
| **Rollout segment** | 128 decision steps | `= n_steps` | One env's contribution to a rollout buffer |
| **Episode** | ≤ 4500 decision steps | `≤ max_episode_steps` | One life, ends on death or time-limit truncation |
| **Training iteration** | 1024 decision steps | `= n_envs × n_steps` | One rollout collection + one PPO update |

Key facts to internalize:

- **"num_timesteps" in SB3 = decision steps**, summed across all 8 envs. So
  4M `total_timesteps` = 4M decision steps = 32M emulator frames ≈ 148 hours of
  game time compressed into ~21 h wall-clock.
- **The fire pulse (12 frames) and decision step (8 frames) are deliberately
  decoupled.** `AutoFireWrapper` wraps the raw env, so it counts emulator
  frames regardless of how `ActionRepeat` groups them. The agent never controls
  firing; it only controls movement at the 7.5 Hz decision rate.
- **Two reward components combine per decision step.** `ActionRepeat` sums the
  *native* game reward over the 8 emulator frames. `RewardShapingWrapper` sits
  *outside* `ActionRepeat`, so it fires once per decision step: it reads the
  info dict from the final emulator frame, computes the shaped term from
  deltas accumulated over the whole skip (e.g. total score gained, lives lost),
  applies the `[-50, 10]` clip to *that shaped term*, and adds it to the summed
  native reward. So the clip bounds the shaped component per decision step, not
  per emulator frame.

### 1.6 Public entrypoints

[`env/__init__.py`](../src/retro_rl/env/__init__.py) exports:

- `make_env(cfg, seed, render_mode)` — single wrapped env (used in eval,
  scripts)
- `make_env_fn(cfg, seed, rank)` — closure factory for `SubprocVecEnv` (used
  in training; `rank` offsets the seed so each worker has a distinct RNG state)

---

## Milestone 2 — Models + Agents

This milestone defines **what the agent sees** (feature extractors) and **what
it decides** (actor-critic policy heads).

### 2.1 The Actor-Critic architecture

PPO uses an **actor-critic** design: a single shared visual backbone feeds two
separate heads:

```
observation (4, 84, 84) uint8
       │
  Feature Extractor (CNN backbone)        ← shared representation
       │
       ├──► Policy head (actor)           → π(a|s): action distribution
       │     Linear → softmax over actions
       │     Sampled during training, argmax at eval
       │
       └──► Value head (critic)           → V(s): expected return scalar
             Linear → scalar
             Used only for training (advantage estimation)
             Discarded at deployment
```

**Why sharing the backbone?** Both heads benefit from the same visual
features. The value function needs to understand the same scene structure as
the policy. Sharing weights reduces parameters and speeds learning by
allowing gradients from both heads to improve the shared representation.

**Why separate heads?** The actor and critic solve different problems — one
outputs a distribution, the other a scalar. Separate MLP heads after the shared
features give each head its own representational capacity without cross-task
interference.

### 2.2 RetroCNN — the Nature-CNN baseline

[`models/cnn.py`](../src/retro_rl/models/cnn.py):

```
(4, 84, 84) uint8
   │  /255
Conv2d(4→32, k=8, s=4) → ReLU     → (32, 20, 20)
Conv2d(32→64, k=4, s=2) → ReLU    → (64, 9, 9)
Conv2d(64→64, k=3, s=1) → ReLU    → (64, 7, 7)
Flatten                             → (3136,)
Linear(3136 → features_dim) → ReLU → (512,)
```

Large strided convolutions: `stride=4` first layer halves twice over, giving
fast spatial compression. This network was introduced in the DQN paper (Mnih
et al. 2015) for Atari — solid baseline, ~1.8M parameters at features_dim=512.

**Limitation:** The large strides reduce the effective receptive field's
spatial precision. On Airstriker — where many small bullets are spread across
the frame — the network may miss fine-grained positional patterns.

### 2.3 ImpalaCNN — the v9 upgrade

[`models/impala.py`](../src/retro_rl/models/impala.py):

**The residual block:**

```python
class _ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        x = torch.relu(x)
        x = self.conv1(x)    # Conv 3x3, s1, p1  (C → C)
        x = torch.relu(x)
        x = self.conv2(x)    # Conv 3x3, s1, p1  (C → C)
        return x + residual  # identity skip: pure add, no activation
```

Three structural properties that matter for RL training:

**(a) Pre-activation order** (`ReLU → Conv`, not `Conv → ReLU`). The skip
is `F(x) + x` with a **pure identity** — no activation on the skip path.
Gradient consequence:

```
∂L/∂x = ∂L/∂(out) · (∂F/∂x + 1)
```

The `+1` is a **gradient highway**. Even if the convolutional path saturates
and `∂F/∂x → 0`, the gradient still flows unattenuated backward through the
skip. This makes 15-conv stacks trainable without vanishing gradients.

**(b) Channel-preserving skips.** Both convs in the block are `C → C`, so
`F(x)` and `x` have the same shape — the skip is a plain addition with no
projection. No extra parameters on the skip path.

**(c) No BatchNorm.** RL minibatches are small, correlated (same policy
generating them), and non-stationary (policy keeps changing which states it
visits). BN's running statistics become unreliable in this regime and
frequently hurt. The near-identity initialization (`F≈0` at init) is a
benign starting point without BN's help.

**The conv sequence:**

```python
class _ConvSequence(nn.Module):
    def forward(self, x):
        x = self.conv(x)   # Conv 3x3, s1, p1: changes channel count, keeps H×W
        x = self.pool(x)   # MaxPool k=3, s=2, p=1: halves H×W
        x = self.res1(x)   # residual refinement
        x = self.res2(x)   # residual refinement
        return x
```

One job per component: the **stack conv** changes channel width; the
**maxpool** halves resolution; the **two residual blocks** add nonlinear
depth at that resolution. All downsampling via maxpool (not strided convs)
preserves finer spatial detail.

**Spatial flow:**

| After | Channels | H×W | Notes |
|---|---|---|---|
| Input | 4 | 84×84 | 4 grayscale frames |
| Stack 1 (4→16) | 16 | 42×42 | maxpool /2 |
| Stack 2 (16→32) | 32 | 21×21 | maxpool /2 |
| Stack 3 (32→32) | 32 | 11×11 | maxpool /2 |
| ReLU + Flatten | — | 3872 | 32×11×11 |
| Linear(3872→256) + ReLU | — | 256 | ~991K params |

**Receptive field (why depth matters):**
Each output cell in the 11×11 final map has a theoretical receptive field of
**141 pixels** — larger than the 84-pixel frame. Every cell can integrate
information from anywhere on screen. This lets the policy relate the ship's
position to threats in any corner of the field. The Nature-CNN's large-stride
convs plateau here; the IMPALA backbone's residual depth is why v9 significantly
outperformed v5-v8.

**Parameter budget:**
- Conv backbone: ~97.7K params
- Linear projection (3872→256): ~991K params (**88% of the network**)
- Actor + critic heads: ~42K params
- **Total: 1,131,098 params**

The "deep ResNet" carries very few parameters; the depth buys representation
quality (nonlinearity, gradient flow), not parameter count. The FC projection
dominates.

### 2.4 Feature extractor registry

[`models/policies.py`](../src/retro_rl/models/policies.py):

```python
FEATURE_EXTRACTORS = {
    "nature_cnn": RetroCNN,
    "impala": ImpalaCNN,
}

def policy_kwargs(features_dim=512, features_extractor="nature_cnn") -> dict:
    extractor_cls = FEATURE_EXTRACTORS[features_extractor]
    return {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": {"features_dim": features_dim},
        # normalize_images=False because both extractors do their own /255
        # inside forward(); leaving it True would double-normalize.
        "normalize_images": False,
    }
```

The registry is a **flat dict** — the "second variant" trigger that CLAUDE.md
specifies (earn the abstraction when the second concrete thing exists). The
validator in `TrainConfig` uses the registry as the source of valid names, so a
typo in the YAML fails at config load time.

**`normalize_images=False` is load-bearing.** SB3's default behavior for image
observation spaces is to divide the input by 255 *inside the policy* before
handing it to the feature extractor. But both `RetroCNN` and `ImpalaCNN` already
do `observations.float() / 255.0` in their own `forward()`. Leaving SB3's
default (`True`) would normalize twice — feeding the network values in
`[0, 1/255]` instead of `[0, 1]`, crushing the input scale and crippling
learning. Setting it `False` hands the extractor the raw uint8 tensor so its
own `/255` is the only normalization.

**What this helper does *not* set: the MLP head architecture.** The doc above
omits `net_arch`, so SB3 falls back to its **default for actor-critic policies**:
two 64-unit hidden layers for each of the policy head (`pi`) and value head
(`vf`), i.e. `dict(pi=[64, 64], vf=[64, 64])`, with **Tanh** activations (SB3's
default `activation_fn`). So the heads in the §2.1 diagram are SB3 defaults, not
something we configure. The contrast worth internalizing: the **backbone uses
ReLU** (our extractor code, standard for conv stacks — unbounded positive
activations), while the **heads use Tanh** (SB3 default for actor-critic MLPs —
bounded `[-1, 1]` activations keep the pre-softmax logits and the value output
in a stable range, which empirically stabilizes policy-gradient updates).

### 2.5 PPO factory

[`agents/ppo.py`](../src/retro_rl/agents/ppo.py) — `build_ppo(vec_env, cfg)`:

```python
return PPO(
    policy="CnnPolicy",
    env=vec_env,
    learning_rate=linear_schedule(ppo_cfg.learning_rate),  # callable: f(progress) → lr
    clip_range=linear_schedule(ppo_cfg.clip_range),        # callable: f(progress) → clip
    policy_kwargs=build_policy_kwargs(features_dim, features_extractor),
    ...
)
```

**Linear schedules:** SB3 accepts callables `f(progress_remaining: float) → float`
where `progress_remaining` goes from 1.0 (start) to 0.0 (end). Multiplying by
the initial value gives a linear ramp from `initial` to 0.

```python
def linear_schedule(initial_value):
    def _fn(progress_remaining):
        return float(progress_remaining * initial_value)
    return _fn
```

This anneals both learning rate (standard for deep RL) and clip range (prevents
large updates late in training when the policy is already good).

### 2.6 Agent Protocol

[`agents/base.py`](../src/retro_rl/agents/base.py):

```python
class Agent(Protocol):
    def predict(self, obs, deterministic=False) -> tuple[np.ndarray, Any]: ...
    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path, ...) -> "Agent": ...
```

A `typing.Protocol` (structural subtyping) rather than an ABC. Both SB3's PPO
and the `RandomAgent` conform without being forced into a class hierarchy. This
is intentional: we don't own SB3's PPO class, so we can't make it inherit from
our ABC. Protocols match on duck-typing.

### 2.7 Random agent baseline

[`agents/random_agent.py`](../src/retro_rl/agents/random_agent.py): Samples
uniformly from the action space with a fixed seed. Useful as a performance
baseline (if a trained agent can't beat random, something is very wrong) and
as a cheap sanity-check for the wrapper stack (verify reward flows without
expensive training). Saved/loaded as a JSON sidecar (no torch weights needed).

---

## Milestone 3 — Training Pipeline

This milestone is the heart of the RL system. It wires the env and agent into
the **PPO learning loop**, manages parallel workers, and records checkpoints.

### 3.1 The PPO learning loop — overview

Proximal Policy Optimization (Schulman et al. 2017) is an **on-policy** RL
algorithm. Each iteration:

```
1. Rollout collection  ─► 1024 transitions (8 envs × 128 steps each)
2. Advantage estimation (GAE)
3. Optimization: 4 epochs over 1024 transitions in minibatches of 256
4. Discard data, goto 1
```

#### What "on-policy" means — precisely

"On-policy" has a specific mathematical meaning that goes deeper than "use
fresh data." The policy gradient theorem (§3.4) gives the gradient of the
objective as an **expectation under the current policy**:

```
∇_θ J(θ) = E_{τ ~ π_θ}[ ∇_θ log π_θ(a|s) · Ψ_t ]
```

The subscript `τ ~ π_θ` is the load-bearing part: the expectation is over
trajectories **generated by** `π_θ`. To compute an unbiased estimate of this
gradient from a finite sample of transitions, those transitions must have been
**drawn from the current policy's behavior** — the same distribution the
expectation is over.

If you use transitions collected under some other (older) policy `π_old`, the
empirical average no longer approximates the true expectation: you are
estimating the wrong integral. The resulting gradient is **biased** — it points
in the wrong direction. Over many updates, biased gradients send the policy
toward local optima that look good under the old distribution but are poor
under the policy you actually end up with.

**Concretely:** On-policy means the rollout buffer must be discarded after each
PPO update. The data is only valid for one policy, and that policy just
changed. Using it again after the weights shift — even once — violates the
on-policy assumption.

#### How on-policy training works mechanically during a single iteration

```
Step 1 — Freeze the policy weights: θ_old = θ

Step 2 — Rollout collection (on-policy):
  For each of 8 parallel envs, run 128 decision steps:
    obs_t → actor(θ_old) → sample action a_t ~ π_θ_old(·|obs_t)
    store: (obs_t, a_t, r_t, log π_θ_old(a_t|obs_t), V_θ_old(obs_t))
  Buffer now holds 1024 transitions, all generated by π_θ_old.

Step 3 — GAE over the buffer:
  Compute advantages Â_t and value targets R_t using the stored V_θ_old(s).
  These are valid because V was estimated by the same frozen weights.

Step 4 — Optimization (4 epochs × 4 minibatches):
  For each minibatch of 256 transitions:
    Forward-pass with current θ (which starts = θ_old, then drifts slightly):
      → log π_θ(a_t|obs_t)   [fresh]
      → V_θ(obs_t)           [fresh]
    Importance ratio: r_t = exp(log π_θ - log π_θ_old)
    Clip r_t to [1-ε, 1+ε] to bound the drift (§3.4).
    Loss = -L^CLIP + 0.5·L^VF - ent_bonus
    Gradient step: θ ← θ - α·∇L

Step 5 — Discard entire buffer. θ_old := θ. Goto Step 2.
```

Two subtleties worth internalizing:

- **The importance ratio is what lets PPO do 4 epochs.** Strictly on-policy
  would mean exactly one gradient step per rollout (after step 1, θ ≠ θ_old
  and the data is technically stale). PPO extends this by tracking *how much*
  the policy has drifted via the ratio `r_t = π_θ / π_θ_old` and clipping
  it to ensure the drift stays within the trust region. The 4 epochs are a
  controlled violation of the on-policy constraint — acceptable as long as
  the clip keeps `r_t ∈ [0.9, 1.1]`. Once you exceed that range, the
  gradient is zeroed by the clip, preventing further exploitation of stale
  data.

- **The buffer stores `log π_θ_old` at collection time precisely because of
  the importance ratio.** Without the stored log-probs, you could not compute
  `r_t` during optimization — you would have no denominator. This is the
  concrete memory footprint of the on-policy requirement.

#### The rollout cycle illustrated — what the 8 envs actually do

```
  Env 1 │←── rollout 1 (128 steps) ──────────────────────│←── rollout 2 (128 steps) ──────...
        [─────────────────────── ep1 ────────────────────│──────────── ep1 cont. ─────────
        0                                               127 128                           255
                                                             ↑ continue from where left off

  Env 3 │←── rollout 1 (128 steps) ──────────────────────│←── rollout 2 (128 steps) ──────...
        [─────── ep1 ──── ✕ │ ep2 ────────────────────── │── ep2 cont. ── ✕ │ ep3 ─────────
        0                74  75                         127 128           172  173         255
                          └─ auto-reset,                     └─ auto-reset,
                             keep filling window                keep filling window

  ...all 8 envs run in parallel (separate subprocesses)...

  After all 8 envs finish their 128 steps:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Pool → 8 envs × 128 steps = 1024 transitions (one rollout)     │
  │  GAE  → compute advantages Â_t and value targets R_t            │
  │  Opt  → 4 epochs × 4 minibatches of 256 → θ updated            │
  │  Done → buffer discarded; all 8 envs continue from step 128     │
  └──────────────────────────────────────────────────────────────────┘

  Rollout 2 begins: each env picks up exactly where it left off.
  (Env 1 at ep1-step 128;  Env 3 at ep2-step 53;  etc.)
```

The 128-step window is a **data collection budget**, not an episode boundary.
Episodes that end mid-window auto-reset and keep filling the same window.
The optimize step fires once per completed window across all 8 envs — never
mid-episode, never mid-window.

#### Why stale data causes problems — the distributional shift argument

Suppose you collect 1024 transitions under policy `π_0` (which tends to move
UP-RIGHT) and then compute:

```
Â_t = Q_π0(s, a) - V_π0(s)
```

These advantages measure action quality **relative to what `π_0` does in `s`
on average**. After one gradient step the policy is now `π_1` (which tends to
move UP-LEFT instead). If you reuse those same transitions:

1. The states `s_t` in the buffer are distributed as `d^{π_0}(s)` — the state
   visitation distribution of `π_0`. But `π_1` visits a different set of
   states `d^{π_1}(s)`. The gradient update pushes `π_1` to be good at states
   it may never actually visit.

2. `V_{π_0}(s)` used in GAE no longer approximates `V_{π_1}(s)`. The baseline
   subtraction that reduces variance now injects bias instead.

3. `Â_t` was computed as "how much better than `π_0`'s average." But you are
   now training `π_1`. An action that was above average for `π_0` may be below
   average for `π_1`. You are reinforcing the wrong things.

These errors compound multiplicatively across updates. In practice, agents
trained with severely stale data collapse: the policy drives itself into a
corner of state space that was visited under the old policy but is a dead-end
under the new one.

#### On-policy vs off-policy — concrete comparison

| Property | On-policy (PPO) | Off-policy (DQN, SAC) |
|---|---|---|
| **Data source** | Current policy only | Any past policy (replay buffer) |
| **Buffer size** | Tiny (1024 transitions, discarded each iteration) | Large (10⁵–10⁶ transitions, kept for the entire run) |
| **Sample efficiency** | Low — each transition used ≤4 epochs then thrown away | High — each transition used many times across many updates |
| **Bias of gradient** | Low (data matches the distribution in the gradient formula) | Potentially high without correction (data came from different policies) |
| **Stability mechanisms** | Clip (ε), entropy bonus, GAE | Target network, replay buffer, Bellman consistency |
| **Target network need** | No — value target is bootstrapped from the policy's own V | Yes — without a frozen target network, the Bellman target is a moving source of bias (chasing a moving target causes divergence) |
| **Wall-clock efficiency** | Can achieve high throughput with parallel envs | Can train on past experience without new env interaction |

**Target networks — why off-policy needs them but on-policy doesn't.** In
DQN, the Q-function update is:

```
Q(s, a) ← r + γ · max_{a'} Q(s', a')
```

Both the prediction (`Q(s,a)`, left side) and the target (`γ·max Q(s', a')`,
right side) use the same network weights. As the network updates, the target
moves too — you are chasing a moving reference, which is a well-known source of
instability and divergence in nonlinear function approximators. The fix: freeze
a copy of Q (the **target network** `Q_target`) and update it slowly (every
10K steps). The prediction chases the frozen target.

PPO does not face this problem because the value target `R_t = Â_t + V(s_t)`
is **computed once at rollout time** from the frozen `θ_old` and held fixed
through all 4 optimization epochs. The "moving target" problem doesn't arise
because the value targets in the buffer don't change during optimization — they
are snapshots from the frozen weights. This is a direct consequence of being
on-policy: the data was collected under a specific frozen policy, so the value
estimates from that policy are consistent targets for that iteration.

#### Side-by-side: PPO rollout buffer vs DQN replay buffer

```
PPO (on-policy) — rollout buffer
─────────────────────────────────────────────────────────────────────
  Lifetime:  one iteration (128 steps × 8 envs = 1024 transitions)
  Contents:  transitions from π_current only
  Usage:     4 epochs × 4 minibatches → then completely overwritten

  ┌─────────────────────────────────┐
  │  1024 fresh transitions (π_now) │  ← written this iteration
  └─────────────────────────────────┘
        ↓ GAE + 4 epochs of updates ↓
  ┌─────────────────────────────────┐
  │         [overwritten]           │  ← next iteration's fresh data
  └─────────────────────────────────┘

  Update cadence: once every 1024 env steps (one policy update per rollout)

DQN (off-policy) — replay buffer
─────────────────────────────────────────────────────────────────────
  Lifetime:  entire training run (kept and grown continuously)
  Contents:  transitions from ALL past policies π_0, π_1, ..., π_now
  Usage:     random minibatch of 32 sampled at every 4 env steps

  ┌──────┬──────┬──────┬────── ... ──────┬──────┬──────┐
  │ π_0  │ π_1  │ π_2  │      ...        │π_498 │π_now │
  │ data │ data │ data │                 │ data │ data │
  └──────┴──────┴──────┴────── ... ──────┴──────┴──────┘
    oldest (may be evicted at capacity ~1M)      newest
                 ↑
    at each update: sample 32 random transitions from anywhere in here
    → these came from many different past policies (that's "off-policy")
    → target network Q_target frozen for 10K steps to stabilize the target

  Update cadence: once every 4 env steps (many updates per env interaction)
```

**The core tradeoff in one line:**
- PPO throws data away to keep gradients unbiased; pays in sample efficiency.
- DQN reuses all historical data for efficiency; pays with a target network and
  the risk that old transitions are stale relative to the current Q-function.

**Sample efficiency tradeoff — why on-policy is expensive.** In this project:
- 4M total decision steps at batch_size=1024 ≈ **3906 PPO iterations**
- Each iteration discards 1024 transitions after 4 epochs (~4096 gradient
  samples total per batch, i.e. each transition is used ~4 times)
- A DQN run could replay each transition **thousands of times** from its buffer

Off-policy algorithms are therefore dramatically more sample-efficient: SAC
often achieves competitive performance with 5–20× fewer env interactions.
On-policy's advantage is **stability and simplicity** — no replay buffer, no
target network, no Bellman error divergence risk — which matters when
hyperparameter sensitivity and debugging cost are the bottleneck, as they are
in visual RL where each training run takes hours.

#### The rollout buffer — what one iteration physically stores

Each PPO iteration fills a fixed-size buffer of shape `(n_steps, n_envs, ...)` =
`(128, 8, ...)`, flattened to **1024 transitions**. Per transition, SB3 stores:

| Field | Shape (per transition) | dtype | Recorded when | Used for |
|---|---|---|---|---|
| `observations` | `(4, 84, 84)` | uint8 | rollout (forward pass input) | re-forward during optimization |
| `actions` | `()` scalar | int64 | rollout (sampled) | recompute `log π_θ(a)` |
| `rewards` | `()` | float32 | rollout (env step) | GAE |
| `episode_starts` | `()` | bool | rollout | GAE terminal handling |
| `values` `V(s_t)` | `()` | float32 | rollout (critic forward) | GAE (the `V` in `δ_t`) |
| `log_probs` `log π_old` | `()` | float32 | rollout (actor forward) | importance ratio denominator |
| `advantages` `Â_t` | `()` | float32 | **after** rollout (GAE) | `L^CLIP` weighting |
| `returns` `R_t` | `()` | float32 | **after** rollout (GAE) | `L^VF` regression target |

Two practical points:

- **Observations dominate memory.** `1024 × 4 × 84 × 84` uint8 ≈ **29 MB** per
  iteration; everything else is a handful of float32 scalars (~24 KB total).
  Keeping obs in uint8 (normalizing to float only inside the CNN forward) is a
  4× memory win that matters at this buffer size.
- **`values` and `log_probs` are the "old policy" snapshot.** They are recorded
  *once* during rollout with the data-collecting weights `θ_old` and then held
  fixed through all 4 optimization epochs. The optimization recomputes fresh
  `V_θ(s)` and `log π_θ(a)` each minibatch and compares against these frozen
  references — that comparison *is* the importance ratio (§3.4) and the value
  target. This is the concrete mechanism behind "on-policy": the buffer is only
  valid while `θ` stays close to `θ_old`, which the clip enforces.

### 3.2 VecEnv construction

[`training/trainer.py`](../src/retro_rl/training/trainer.py) — `_build_vec_env(cfg)`:

```python
env_fns = [make_env_fn(cfg.env, seed=cfg.seed, rank=i) for i in range(cfg.n_envs)]
venv = VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))
if cfg.normalize_reward:
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg.ppo.gamma)
```

**`SubprocVecEnv`** (always, even at `n_envs=1`)**: Runs each env in a
separate subprocess. Critical constraint: stable-retro enforces **one emulator
per process**. If the train env and eval env were in the same process, the
second `retro.make()` call would raise `RuntimeError("Cannot create multiple
emulator instances per process")`. By putting train envs in subprocesses, the
main process is free to run the eval env.

**`VecMonitor`** sits *inside* `VecNormalize`: it logs `ep_rew_mean` using the
raw (unnormalized) reward. This is intentional — `rollout/ep_rew_mean` in
TensorBoard reflects the actual game score, comparable across runs.

**`VecNormalize`** (v9 addition) divides rewards by a running standard
deviation estimate, keeping return variance at ~1. The value function then
needs to predict values in range ~[-5, 5] instead of [-2000, 7000]. This
made the value head converge dramatically: EV went from ~0.4 (v8 LR=1e-4
only) to ~0.9, value_loss from ~800 to ~0.1.

**Why `norm_obs=False`:** Images are normalized to [0, 1] inside the CNN's
`forward`. Applying VecNormalize to images would give the train env a different
pixel scale than the eval env (which doesn't use VecNormalize) — mismatched
input distributions would silently hurt generalization.

**`gamma` threading:** VecNormalize's running variance estimate of returns
uses the discount factor: `σ²(G) = σ²(r + γG')`. If `gamma` mismatches
PPO's gamma, the scale estimate is wrong. The config threads `cfg.ppo.gamma`
to both.

**Resume safety:** VecNormalize running stats (mean, variance, count) are
saved as a `.pkl` sidecar alongside every checkpoint. On `resume_from`:

```python
venv = VecNormalize.load(str(vecnormalize_stats), venv)
venv.training = True   # continue updating the running stats
venv.norm_reward = True
```

Without this, a crash-resume would start the reward scale fresh, causing the
value head to mis-estimate values for the first ~200K steps.

### 3.3 The training call

```python
model.learn(
    total_timesteps=cfg.total_timesteps,
    callback=callbacks,
    tb_log_name=cfg.run_name,
    log_interval=cfg.log_interval,
    reset_num_timesteps=resume_from is None,
)
```

SB3's `PPO.learn()` runs the entire rollout-update loop internally. `callback`
is a `CallbackList` that fires hooks at each step, rollout end, and training end.
`reset_num_timesteps=False` on resume preserves the step counter so schedules
(LR, clip_range, ent_coef) continue from the right point in the annealing curve.

### 3.4 The PPO update — deep theory

This section explains exactly what happens during the "optimization" phase
(step 3 in §3.1) and why PPO is designed the way it is.

#### Foundation: policy gradients

We maximize the expected cumulative discounted reward:

```
J(θ) = E_π[ Σ_t γ^t r_t ]
```

The **policy gradient theorem** (Sutton et al. 1999) gives the gradient:

```
∇_θ J(θ) = E_π[ ∇_θ log π_θ(a_t | s_t) · Ψ_t ]
```

where `Ψ_t` measures how good action `a_t` was. The gradient tells us: *"in
the direction of actions that were better than expected."*

Intuition: `log π(a|s)` is the log-probability of the action taken. Its
gradient points in the direction that makes the action more likely.
Multiplying by `Ψ_t` (positive if the action was good) makes good actions
more likely and bad actions less likely.

**Problem 1 — High variance:** Using `Ψ_t = R_t` (full return from step t)
involves the entire random future trajectory. Small changes in the policy
create large swings in `R_t`. The gradient is unbiased but extremely noisy →
requires enormous sample sizes to learn reliably.

**Problem 2 — Data staleness:** After one gradient step, the policy changes.
The data collected under the old policy `π_old` is no longer valid for the
new policy `π`. Using stale data violates the on-policy assumption and leads
to biased (incorrect) gradient estimates.

#### Fix 1: Advantage estimation (reduces variance)

Instead of the full return `R_t`, use the **advantage**:

```
A(s, a) = Q(s, a) - V(s)
```

"How much better is action `a` than the policy's average at state `s`?" This
is **baseline subtraction** — subtracting `V(s)` doesn't change the expected
gradient (unbiased) but removes the "what was my average return from this
state" component, dramatically reducing variance. Actions are reinforced
relative to what was expected, not by their absolute return.

Estimating `A` requires computing `Q(s, a)` and `V(s)`. GAE handles this
(§3.5). The critic network provides `V(s)`.

#### Fix 2: Importance sampling ratio

PPO reuses each rollout for `n_epochs=4` gradient updates. After the first
update, the policy has drifted from the data-collecting policy `π_old`. PPO
accounts for this with the **importance sampling ratio**:

```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

- `r=1`: policy unchanged; gradient weights action equally
- `r>1`: action became more likely under the new policy
- `r<1`: action became less likely

This is why each rollout transition stores `log π_θ_old(a_t|s_t)` — the
denominator. At optimization time: `r = exp(log π_θ - log π_θ_old)`.

#### Fix 3: Clipped surrogate objective

A naive objective `E[r_t · Â_t]` would push `r_t` arbitrarily far when
advantages are large — destructive updates that invalidate the on-policy
assumption. PPO clips it:

```
L^CLIP(θ) = E[ min( r_t · Â_t ,  clip(r_t, 1-ε, 1+ε) · Â_t ) ]     ε = 0.1 (v9)
```

The `min` makes it **pessimistic**:

| Scenario | Effect |
|---|---|
| `Â_t > 0` (good action) | Increase `π(a)`, but clip at `1+ε` → no benefit pushing `r` past 1.1 |
| `Â_t < 0` (bad action) | Decrease `π(a)`, but clip at `1-ε` → no benefit pushing `r` below 0.9 |

The clip creates a **soft trust region**: updates are bounded so the new policy
doesn't stray too far from the data-generating policy, keeping the importance
sampling approximation valid. This is the key insight of PPO — it approximates
TRPO's hard KL constraint with a cheap, differentiable clip.

**Worked clip example.** Take one transition with a strongly positive advantage,
`Â_t = +4.834` (the `Â₀` from the GAE example above), and `ε = 0.1`. Watch what
the objective does as the policy drifts during the 4 optimization epochs:

| Update state | `r_t` | `r_t·Â_t` | `clip(r_t,0.9,1.1)·Â_t` | `L^CLIP = min(...)` | Gradient? |
|---|---|---|---|---|---|
| Before any update | 1.00 | 4.834 | 4.834 | **4.834** | yes — push `r` up |
| Action got more likely | 1.05 | 5.076 | 5.076 | **5.076** | yes — still inside band |
| At the clip edge | 1.10 | 5.317 | 5.317 | **5.317** | yes — at the cap |
| Pushed past the cap | 1.30 | 6.284 | 5.317 | **5.317** | **no** — clipped branch wins, `∂/∂r = 0` |

Once `r_t` exceeds `1+ε = 1.1`, the `min` selects the clipped term `1.1·Â_t`,
which is **constant in `r_t`** — its gradient is zero. The optimizer gets no
further reward for making this already-favored action even more likely. That is
the trust region: a good action is reinforced, but only up to a 10% probability-
ratio change per rollout, after which the gradient flatlines and the update
stops. The symmetric thing happens for `Â_t < 0` at the `1−ε = 0.9` floor.

**Why `min` and not just clip?** Note the table only shows the `Â>0` side. The
`min` is what makes the objective *pessimistic* and is subtle: for `Â<0`, an
*un*clipped large `r` (the policy accidentally made a bad action much more
likely) is **not** clipped away — the `min` keeps the larger-magnitude negative
term so the objective still penalizes it. The clip only removes the *incentive
to over-optimize in the favorable direction*; it never hides a mistake that
needs correcting. This asymmetry is the entire safety property of PPO.

**The `clip_fraction` diagnostic** in TensorBoard is the fraction of transitions
where the clip was active (`|r−1| > ε`) in a given update. Healthy PPO sits
around 0.1–0.3; near 0 means updates are too timid (raise LR or ε), near 0.5+
means the policy is trying to move violently every update (lower LR).

#### The full PPO loss

```
L(θ) = L^CLIP(θ)  -  c₁ · L^VF(θ)  +  c₂ · H[π_θ](s)
```

- `L^CLIP`: policy loss (maximize: good actions become more likely)
- `L^VF = MSE(V_θ(s), V_target)`: value loss — critic fits the GAE returns
  (`c₁ = vf_coef = 0.5`)
- `H[π_θ]`: entropy bonus — prevents premature action distribution collapse
  (`c₂ = ent_coef`, annealed 0.02 → 0.001 in v9)

SB3 minimizes `-L`, so signs flip in code.

**Entropy annealing rationale:** Early in training, high entropy = exploratory
policy that tries many actions and discovers good strategies. Late in training,
high entropy competes with convergence — the policy needs to commit to good
actions. Linear annealing balances this. In v5 with constant ent_coef=0.02,
the entropy term dominated late-run gradients and dragged the policy back toward
uniform (observed: `approx_kl → 0`, eval return de-committed from 244 to 214).

### 3.5 Generalized Advantage Estimation (GAE)

After rollout collection, advantages must be estimated for all transitions.
This is a bias-variance tradeoff problem.

**TD error (one-step):**

```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

This is the temporal difference error — "the value prediction was off by this
much." It uses the critic `V` to estimate `Q(s,a) ≈ r_t + γV(s_{t+1})`.

**Pure one-step (`Â = δ_t`):** Low variance (δ_t is a single sample), but
**high bias** — the estimate leans entirely on the imperfect critic V. If V
is wrong (which it is early in training), the advantage estimate is wrong.

**Monte Carlo (`Â = Σ γ^l r_{t+l} - V(s_t)`):** Unbiased (uses actual
observed rewards, not bootstrapped V), but **high variance** — requires
waiting for the full episode return, which has enormous sample-to-sample
variation.

**GAE (`λ=0.95`)** is an exponentially-weighted average of all n-step estimators:

```
Â_t^GAE = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}
```

Implemented via backward recursion over the rollout buffer:

```
Â_T = δ_T                          (last step, or terminal)
Â_t = δ_t + (γλ) · Â_{t+1}
```

`λ` is the bias-variance knob:
- `λ=0`: `Â = δ_t` — one-step TD (low variance, high bias)
- `λ=1`: `Â = Σ γ^l r - V(s_t)` — Monte Carlo (high variance, low bias)
- `λ=0.95` (v9): mostly-unbiased, variance-controlled blend

**Terminal state handling:** At a true death (`terminated=True`), there is no
next state to bootstrap from: `δ_t = r_t - V(s_t)` (no γV(s')).
At a **truncation** (`truncated=True`, hit TimeLimit), bootstrapping is still
valid: the episode ended by time limit, not death, so `V(s_{t+1})` is a
legitimate future value estimate. Getting this distinction wrong introduces
systematic bias into advantage estimates.

**Why GAE is the deep reason VecNormalize unblocked v9:**
Every `δ_t` contains `V(s)` and `V(s')`. If the critic is poor, the
advantages are noisy/biased. In v8, raw returns made the value regression
ill-conditioned (EV ~0.22, value_loss in the hundreds) → noisy V → noisy δ_t
→ noisy/biased advantages → slow, unstable policy learning. VecNormalize fixed
the return scale → V learned well → δ_t became trustworthy → advantages
became good → the policy could exploit the IMPALA features. The chain is:
**value quality → advantage quality → policy improvement**.

**Advantage normalization:** After GAE, SB3 normalizes advantages to mean 0 /
std 1 per minibatch (`normalize_advantage=true`). This is an additional
variance-control step: it prevents minibatches with unusually high-return
episodes from dominating the gradient.

**The value target:** `R_t = Â_t + V(s_t)` — the GAE return is used as the
regression target for the value head (`L^VF = MSE(V_θ(s_t), R_t)`). GAE and
the value head are coupled: the critic's predictions feed into GAE, and GAE's
returns train the critic.

#### Worked GAE example — 4-step backward recursion

Formulas are easy to nod along to and hard to actually understand. Here is the
full computation over a tiny 4-step rollout with concrete numbers. Use
`γ = 0.99`, `λ = 0.95`, so `γλ = 0.9405`.

Suppose the agent took 4 decision steps and then the episode **truncated**
(hit the time limit — did *not* die), with these recorded values:

| t | reward `r_t` | critic `V(s_t)` | next value `V(s_{t+1})` |
|---|---|---|---|
| 0 | 1.0 | 5.0 | 5.5 |
| 1 | 1.0 | 5.5 | 6.0 |
| 2 | 1.0 | 6.0 | 6.2 |
| 3 | 1.0 | 6.2 | 6.5 ← bootstrap (truncation, not death) |

**Step 1 — compute the one-step TD errors** `δ_t = r_t + γV(s_{t+1}) − V(s_t)`:

```
δ₀ = 1.0 + 0.99·5.5 − 5.0 = 1.0 + 5.445 − 5.0 = 1.445
δ₁ = 1.0 + 0.99·6.0 − 5.5 = 1.0 + 5.940 − 5.5 = 1.440
δ₂ = 1.0 + 0.99·6.2 − 6.0 = 1.0 + 6.138 − 6.0 = 1.138
δ₃ = 1.0 + 0.99·6.5 − 6.2 = 1.0 + 6.435 − 6.2 = 1.235
```

Each `δ_t` says "this step's reward + discounted next-value was `δ` higher than
the critic predicted" — a local surprise signal.

**Step 2 — accumulate advantages backward** `Â_t = δ_t + γλ·Â_{t+1}`,
starting from the last step (`Â₄ = 0` past the bootstrap):

```
Â₃ = δ₃ + 0.9405·0     = 1.235
Â₂ = δ₂ + 0.9405·Â₃    = 1.138 + 0.9405·1.235 = 1.138 + 1.162 = 2.300
Â₁ = δ₁ + 0.9405·Â₂    = 1.440 + 0.9405·2.300 = 1.440 + 2.163 = 3.603
Â₀ = δ₀ + 0.9405·Â₁    = 1.445 + 0.9405·3.603 = 1.445 + 3.389 = 4.834
```

**Step 3 — value targets** `R_t = Â_t + V(s_t)`:

```
R₀ = 4.834 + 5.0 = 9.834
R₁ = 3.603 + 5.5 = 9.103
R₂ = 2.300 + 6.0 = 8.300
R₃ = 1.235 + 6.2 = 7.435
```

What to take away:

- **Advantages grow toward the start** (4.834 at t=0 vs 1.235 at t=3) because
  earlier steps "see" more discounted future surprise — the recursion folds all
  later `δ`s into `Â₀`. This is GAE blending the n-step estimators.
- **Each `Â_t` mixes near and far signal**, weighted by `(γλ)^l`. With
  `γλ = 0.94`, a surprise 10 steps away contributes `0.94^10 ≈ 0.54×` — still
  meaningful; at `λ=0` it would contribute 0× (pure one-step), at `λ=1` it
  would contribute `0.99^10 ≈ 0.90×` (near-Monte-Carlo).
- **The `R_t` values are what the critic regresses toward.** If the critic were
  perfect, every `δ_t` would be ≈ 0 and every `Â_t` would be ≈ 0 — there would
  be no surprise and no gradient. Advantages are literally a measure of how
  *wrong* the current value function is.

**The terminal-vs-truncation branch in action.** The example bootstrapped at
t=3 (`V(s₄) = 6.5`) because the episode *truncated*. Had the agent **died** at
t=3 instead, there would be no future, so:

```
δ₃ = r₃ − V(s₃) = 1.0 − 6.2 = −5.2     (no +γV(s₄) term)
Â₃ = −5.2
```

A death turns the last advantage sharply negative (the critic expected 6.2 of
future value; the agent got nothing), correctly punishing whatever action led
there. Bootstrapping through a death by mistake would instead credit the agent
with 6.5 of phantom future value — a silent, systematic bias. This is why the
`terminated` vs `truncated` distinction from the env layer (§1.5 [9]) propagates
all the way into the math here.

### 3.6 Callbacks

[`training/callbacks.py`](../src/retro_rl/training/callbacks.py) — three SB3
`BaseCallback` subclasses:

#### PeriodicCheckpointCallback

```python
def _on_step(self) -> bool:
    if self.num_timesteps < self._next_save:
        return True
    self.manager.save(self.model, self.num_timesteps, eval_return=None)
    self._next_save = ((self.num_timesteps // self.every_steps) + 1) * self.every_steps
    return True
```

Fires every `every_steps` env steps. Calls `manager.save(eval_return=None)` —
contributes to last-K rotation but not to the "best" tracker.

#### EvalAndVideoCallback

Fires every `every_steps` env steps. Core logic:

1. **Build eval env lazily** on first firing (avoids retro's one-emulator-per-
   process constraint during initialization).
2. **Run `n_episodes` deterministic rollouts** with `deterministic=True` in
   `model.predict`. Uses `lstm_states` + `episode_start` for compatibility with
   RecurrentPPO (plain PPO ignores both args).
3. **Log TB scalars:** `eval/mean_return`, `eval/std_return`, `eval/mean_length`.
4. **Update best checkpoint** via `manager.save(eval_return=mean_return)`.
5. **Record video** of episode 0: calls `env.render()` after each step when
   `render_mode='rgb_array'`.

Deterministic vs stochastic eval: at inference, `model.predict(obs, deterministic=True)`
uses `argmax(logits)` instead of sampling. This removes the luck component —
the same policy always picks the same action from the same observation. For
reproducible evaluation metrics, deterministic is standard.

#### EntCoefLinearSchedule

```python
def _on_rollout_end(self) -> None:
    steps = int(self.model.num_timesteps)
    progress = min(1.0, max(0.0, steps / self.total_timesteps))
    self.model.ent_coef = self.initial + progress * (self.final - self.initial)
    self.logger.record("train/ent_coef", float(self.model.ent_coef))
```

Fires on `_on_rollout_end` (just before the gradient update pass), so PPO
sees the annealed value for the entire upcoming optimization epoch. Mutates
`model.ent_coef` directly — SB3 reads this attribute afresh at each update.

Logs to TB as `train/ent_coef` so the schedule is auditable alongside
`approx_kl` and `entropy_loss`.

### 3.7 CheckpointManager

[`training/checkpoint.py`](../src/retro_rl/training/checkpoint.py):

**Files produced per save:**

```
outputs/checkpoints/<run_name>/
  step-250000.zip         ← SB3 model weights + optimizer state
  step-250000.json        ← sidecar: run_name, step, eval_return, eval_length, timestamp
  step-250000.pkl         ← VecNormalize running stats (if normalize_reward=True)
  best.zip                ← copy of the highest-eval-return checkpoint seen so far
  best.json               ← sidecar for best
  best.pkl                ← VecNormalize stats for the best checkpoint
  config_snapshot.json    ← full TrainConfig at training start (reproducibility)
```

**Atomic write (tmp → rename):**

```python
def _atomic_save_zip(model, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    model.save(str(tmp))
    os.replace(tmp, path)       # atomic on POSIX
```

Readers (the backend API) never see a half-written zip because `os.replace`
is atomic at the filesystem level. If the process crashes mid-write, the tmp
file is left behind but the existing `step-N.zip` is unaffected.

**Pruning policy:** `_prune()` keeps a step checkpoint if it belongs to *any*
of:
- The `keep_last_k` most recent (resume safety — you can resume from any of
  the last K)
- The `keep_top_k` by `eval_return` (dashboards can replay best checkpoints)
- The `keep_top_k` by `eval_length` (best survival, regardless of score)

`best.zip` is never touched by pruning.

**Resume recovery:** On `__init__`, scans `best.json` and restores
`_best_return`. If the training process crashes and is restarted, the manager
picks up the prior best without re-running all evaluations.

### 3.8 Config snapshot

At the start of `train()`:

```python
config_snapshot_path.write_text(
    json.dumps(cfg.model_dump(mode="json"), indent=2, default=str)
)
```

The entire `TrainConfig` (all hyperparameters, env config, paths) is serialized
to `config_snapshot.json` in the run directory. Every checkpoint sidecar records
its path. This means any checkpoint can be exactly reproduced from:
1. The checkpoint `.zip` (weights)
2. The `config_snapshot.json` (hyperparameters)
3. The random seed (also in the snapshot)

### 3.9 Reading the TensorBoard scalars — a diagnostic glossary

Training health is read entirely from the TB scalars SB3 and our callbacks
log. Most of these are referenced throughout this doc and the decisions log;
here is the single place that defines each one, its formula, healthy range,
and what an unhealthy value means. **This table is the operator's instrument
panel** — knowing how to read it is most of what separates "debugging RL" from
"randomly changing hyperparameters."

| Scalar | What it is | Healthy | Unhealthy → likely cause |
|---|---|---|---|
| `rollout/ep_rew_mean` | Mean **raw** (un-normalized) episode reward across the 8 train envs. Logged by `VecMonitor` *inside* `VecNormalize`. | Rising, then plateauing | Flat from step 0 → reward not flowing (wrapper/shaping bug); collapsing → policy de-committing |
| `rollout/ep_len_mean` | Mean episode length (decision steps). Proxy for survival. | Rising (agent survives longer) | Stuck low → agent dies instantly; equal to `max_episode_steps` → agent learned to stall/hide |
| `eval/mean_return` | **The headline metric.** Mean return of N deterministic eval episodes on a bare env. | Monotone-ish rise | Diverges from `ep_rew_mean` → train/eval mismatch (e.g. stochastic-only policy, the v2 bug) |
| `eval/std_return` | Spread across eval episodes. | >0 if env is stochastic | Exactly 0 for all evals → deterministic save-state + no env noise (expected for Airstriker; not a bug) |
| `eval/mean_length` | Mean eval episode length. | Tracks survival | — |
| `train/explained_variance` (EV) | `1 − Var(R_t − V(s_t)) / Var(R_t)`. How much of the return variance the critic explains. **1 = perfect critic, 0 = no better than predicting the mean, <0 = worse than the mean.** | → 0.8–0.99 | Pinned at 0 → value head not fitting (return scale too large — the v8→v9 VecNormalize fix); negative → critic diverging (lower LR) |
| `train/value_loss` (`L^VF`) | `MSE(V_θ(s), R_t)`. Critic regression error. | Small, stable (v9: ~0.1) | Hundreds and rising → ill-conditioned value targets (the symptom VecNormalize cured: ~800 → ~0.1) |
| `train/approx_kl` | `E[log π_old − log π_new]`, an estimate of how far the policy moved this update. | ~0.01–0.03 | → 0 → policy frozen (advantages vanished and/or entropy annealed away — the v5/v7 late-run collapse); large (>0.1) → updates too aggressive |
| `train/clip_fraction` | Fraction of transitions where the PPO clip was active (`|r−1| > ε`). | ~0.1–0.3 | ~0 → updates too timid (raise LR or ε); >0.5 → policy lurching every update (lower LR) |
| `train/policy_gradient_loss` (`−L^CLIP`) | The clipped surrogate objective (negated, since SB3 minimizes). | Small-magnitude, noisy around 0 | Steadily growing magnitude → instability |
| `train/entropy_loss` (`−H[π]`) | Negative policy entropy. For `Discrete(9)`, max entropy is `ln(9) ≈ 2.197` (uniform); 0 means a one-hot deterministic policy. | Starts near `−2.2` (exploratory), rises toward 0 as the policy commits | Stuck near `−2.2` late → policy never commits (ent_coef too high); crashes to 0 early → premature collapse (ent_coef too low) |
| `train/ent_coef` | The current entropy coefficient `c₂`, logged by our `EntCoefLinearSchedule`. | Linear ramp 0.02 → 0.001 over 4M steps | Audited against the de-commitment failure mode — if the policy freezes while this is already tiny, the schedule annealed too fast |
| `train/learning_rate` | Current LR (linear-annealed by our schedule). | Linear ramp 1e-4 → 0 | — |

**How these chain together (the v9 causal story in one read):**
`value_loss` ↓ and EV ↑ (good critic) → trustworthy `δ_t` → trustworthy
advantages → `approx_kl` and `clip_fraction` stay in healthy bands (real,
bounded policy updates) → `eval/mean_return` climbs. When the chain breaks, it
almost always breaks at the **critic** first (EV / value_loss), which is exactly
why the v9 breakthrough was a reward-*scale* fix (VecNormalize), not a policy
change. Read the panel left-to-right in that causal order when triaging.

---

## Milestone 4 — Evaluation

Evaluation answers: "how good is this checkpoint, objectively and
reproducibly?" It is deliberately separate from training.

### 4.1 Why separate evaluation

During training, PPO measures performance via `rollout/ep_rew_mean` — the
mean reward of the stochastic training episodes across all `n_envs` workers.
This is a **biased** proxy for true performance because:

1. **Stochastic policy:** training uses sampled actions, not argmax. The
   stochastic policy tends to score higher than the deterministic one early in
   training (random exploration occasionally gets lucky) and lower late in
   training (once the policy is confident, argmax is better than sampling).
2. **VecNormalize:** the training env uses normalized rewards. Comparing
   `ep_rew_mean` across runs with and without VecNormalize is meaningless.
3. **Seed dependency:** each training env has a different seed. The mean
   across 8 workers approximates the expected return but has high variance.

The eval env is a **fresh, non-normalized, single env** with a fixed seed.
Deterministic policy. Raw rewards. Same seed every time. This gives a
reproducible number for comparing checkpoints and runs.

### 4.2 The evaluator

[`evaluation/evaluator.py`](../src/retro_rl/evaluation/evaluator.py) —
`evaluate(agent, env, n_episodes, deterministic=True)`:

```python
for ep_i in range(n_episodes):
    obs, _ = env.reset()
    ep_return, ep_length, ep_deaths, ep_stage_cleared = 0.0, 0, 0, False
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        ep_return += float(reward)
        ep_length += 1
        if terminated:
            ep_deaths += 1         # death, not truncation
        if info.get(stage_clear_key):
            ep_stage_cleared = True
        if record_video and ep_i == 0:
            frames.append(np.asarray(env.render()))
    episode_results.append(EpisodeResult(...))
return compute_metrics(episode_results), frames
```

**Death vs truncation distinction:** `terminated=True` means the environment
ended the episode (game over). `truncated=True` means the TimeLimit wrapper
cut the episode. These are fundamentally different:
- A death is a genuine failure state.
- A truncation means the episode could have continued — the agent survived.

Counting only `terminated` as a "death" gives a correct deaths-per-episode
metric. Counting truncations would penalize long-surviving agents.

**Stage-clear detection:** At each step, `info.get("stage_clear")` is checked.
If it's truthy at any point, the episode is flagged. This is an OR across the
episode — once the stage is cleared, it stays cleared.

**Video recording:** Only episode 0 of each eval cycle is recorded (cost
control). `env.render()` returns an RGB ndarray when built with
`render_mode='rgb_array'`. Frames are collected in a list and written to MP4
by `utils/video.py:write_mp4`.

### 4.3 Metrics

[`evaluation/metrics.py`](../src/retro_rl/evaluation/metrics.py):

```python
@dataclass(frozen=True)
class EvalMetrics:
    n_episodes: int
    mean_return: float        # primary performance metric
    std_return: float         # indicates determinism / variance (0 for fixed-seed Airstriker)
    min_return: float
    max_return: float
    mean_length: float        # proxy for survival time
    std_length: float
    stage_clear_rate: float   # 0.0–1.0, fraction of episodes that cleared
    mean_deaths: float        # deaths per episode
```

`frozen=True` makes `EvalMetrics` immutable — it's a measurement result, not
mutable state. Mutation would be a bug.

All metrics are computed by `compute_metrics(episodes: list[EpisodeResult])` —
a pure function over a list, trivially testable and side-effect free.

### 4.4 CLI evaluation

[`scripts/evaluate.py`](../scripts/evaluate.py):

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/ppo_airstriker_v9/best.zip \
  --config configs/ppo_v9.yaml \
  --episodes 20 \
  --seed 42 \
  --output-dir outputs/eval/v9
```

Produces:
- `metrics.json` — full `EvalMetrics` as JSON
- `episode_0.mp4` — video of the first episode (unless `--no-video`)

---

## Milestone 5 — Backend (FastAPI)

The backend serves training artifacts over HTTP. The frontend, scripts, and
any external tools communicate exclusively through this REST API — never by
importing backend code directly.

### 5.1 Architecture

[`backend/api.py`](../src/retro_rl/backend/api.py) — `create_app(...)`:

Three singleton registries are created at app startup and stored on
`app.state`:

| Object | Responsibility |
|---|---|
| `CheckpointResolver` | Scans `outputs/checkpoints/` on-demand, lists runs and checkpoints, reads JSON sidecars |
| `AgentRegistry` | LRU cache of loaded PPO models (cap=4 ≈ 80MB RAM) |
| `EpisodeRegistry` | Thread-safe map of `episode_id → EpisodeRuntime` |

Routes use `Depends(get_resolver)` etc. to pull these from `app.state` — this
lets tests inject mocks via `app.dependency_overrides` without a real filesystem.

CORS middleware pre-configured for `localhost:8501` (Streamlit's port).

All routes are **sync** — the hot path is CPU-bound (PPO inference, env.step,
PNG encode). FastAPI runs sync routes in a threadpool automatically, which is
the right behavior for CPU work.

### 5.2 Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness: status, version, uptime |
| `GET` | `/checkpoints` | All checkpoints across all runs (with eval metrics from sidecars) |
| `GET` | `/runs` | Per-run summary: best return, latest step, checkpoint count |
| `GET` | `/runs/{name}/metrics` | All TB scalar series for a run (lazy EventAccumulator) |
| `POST` | `/episodes` | Start a rollout: load agent + build env, return episode_id |
| `GET` | `/episodes/{id}/state` | Snapshot of current episode (step, return, done flag) |
| `GET` | `/episodes/{id}/frame` | Advance one env step, return PNG image |
| `DELETE` | `/episodes/{id}` | Close env, free resources |

### 5.3 Key design decisions

**`CheckpointResolver`** reads sidecar JSONs (not the `.zip` files) to enumerate
checkpoints. This is fast — no model loading for catalog operations.

**`AgentRegistry`** uses an LRU cache: `PPO.load(path)` is expensive (~0.5s,
loads all weights). Caching keeps the last 4 loaded agents in memory. Cache
miss evicts the LRU agent.

**`EpisodeRuntime`** manages one rollout: builds the env (from the checkpoint's
`config_snapshot.json`), runs `env.reset()`, and then steps one frame at a
time on each `GET /episodes/{id}/frame` call. Frames are PNG-encoded via PIL.
A per-instance lock ensures thread safety when two frontend requests hit the
same episode concurrently.

**TB metrics** are parsed lazily: `GET /runs/{name}/metrics` creates an
`EventAccumulator`, calls `Reload()`, and returns all scalar series as JSON.
This is the only route that imports tensorboard — kept out of the hot path.

### 5.4 Episode lifecycle

```
POST /episodes  →  201 Created { episode_id: "..." }
   ↓
GET /episodes/{id}/frame  (repeated) →  200 image/png
   ↓
GET /episodes/{id}/state  →  { step: N, return: X, done: true/false }
   ↓
DELETE /episodes/{id}  →  204 No Content
```

The frontend polls `GET /frame` at the configured FPS. Once `done=true`
(from `/state`), it stops polling and calls `DELETE` to clean up the env.

---

## Milestone 6 — Frontend (Streamlit)

The frontend is a Streamlit multi-page app that visualizes training progress
and lets users watch the trained agent play — all through the backend API.

### 6.1 HTTP-only seam

**Rule: frontend never imports backend code.** All contact funnels through
[`frontend/components/api_client.py`](../frontend/components/api_client.py):

```python
BASE_URL = os.environ.get("RETRO_RL_BACKEND_URL", "http://localhost:8000")

@st.cache_data(ttl=10)
def list_runs() -> list[dict]:
    return requests.get(f"{BASE_URL}/runs").json()["runs"]

def list_checkpoints() -> list[dict]:
    ...  # uncached (staleness = correctness here)

def start_episode(checkpoint_id, seed, deterministic, max_steps) -> dict:
    ...  # never cached (stateful, each call creates a new resource)
```

**`st.cache_data` TTL strategy:** Catalog endpoints (health 5s, runs/ckpts 10s,
metrics 15s) are cached — they change slowly (only when training produces a
new checkpoint). Episode endpoints are uncached — they are stateful resources
where staleness = incorrect state.

### 6.2 Pages

**`app.py` (landing):** Backend health probe, run/checkpoint counts, summary
table. `render_sidebar()` is exported for all pages — runs the health probe
and shows the backend URL setting.

**`pages/1_Training.py`:** Run picker dropdown, summary metrics (best return,
peak episode length), and auto-arranged 2-column plot panels for eval metrics
(`eval/mean_return`, `eval/mean_length`) and rollout metrics (`rollout/ep_rew_mean`,
`train/value_loss`, `train/entropy_loss`, etc.). Charts use plotly dark theme
with transparent background.

**`pages/2_Play.py`:** Checkpoint picker, episode start/stop/pause controls,
and a frame streaming loop:

```python
while st.session_state.playing and not done:
    frame_png = api_client.get_frame(episode_id)
    placeholder.image(frame_png, use_column_width=True)
    state = api_client.get_state(episode_id)
    done = state["done"]
    time.sleep(1.0 / fps)
```

Streamlit's rerun model doesn't suit per-frame updates, so the page runs an
explicit `while` loop within a single rerun. Stop button toggles a session
flag; the next Streamlit rerun calls `DELETE /episodes/{id}`.

**`pages/3_Compare.py`:** Multi-run overlay on a chosen scalar (e.g.
`eval/mean_return` across v5, v7, v8, v9). Peak/final summary table. Useful
for ablation analysis — e.g. showing how VecNormalize (v9) vs constant LR (v8)
changed the return curve.

### 6.3 Launching

```bash
# Terminal 1 — backend
python scripts/serve.py --checkpoint-root outputs/checkpoints \
                        --tensorboard-root outputs/tensorboard

# Terminal 2 — frontend
streamlit run frontend/app.py
# → http://localhost:8501
```

Frontend detects the backend URL from `RETRO_RL_BACKEND_URL` env var,
defaulting to `http://localhost:8000`.

---

## Cross-cutting: data flow summary

```
stable-retro emulator (Airstriker-Genesis-v0)
    │
    │  raw RGB frame (224×320×3), MultiBinary(12) action, native reward
    │
    ▼
AutoFireWrapper → tap B every 12 frames (independent of policy)
DiscreteActionWrapper → Discrete(9) combos (movement only)
ActionRepeat(8) → hold action 8 emulator frames, sum rewards, max-pool frames
EndOnLifeLost → terminate episode on first life decrement
RewardShapingWrapper → add score_delta, life_loss, survival_bonus; clip [-50,10]
GrayscaleResize → 224×320×3 → 84×84×1 uint8
FrameStack(4) → (84,84,4) uint8  ← Markov approximation
TimeLimit(4500) → truncate at 4500 outer steps
    │
    │  shaped observation: (84,84,4) uint8
    │
    ▼
SubprocVecEnv (8 workers, each in a subprocess)
VecMonitor (logs raw episode stats to TB: ep_rew_mean, ep_len_mean)
VecNormalize (norm_reward=True: divides rewards by running std → ~unit variance)
    │
    │  batch of obs (8×84×84×4), normalized rewards
    │
    ▼
SB3 PPO rollout buffer (1024 transitions = 8 envs × 128 steps)
    │
    ▼
GAE (γ=0.99, λ=0.95) → advantages Â_t + returns R_t
Normalize advantages (mean=0, std=1 per minibatch)
    │
    ▼
4 epochs × (1024/256=4 minibatches):
    IMPALA CNN forward → 256-d features
    Actor head → 9 logits → Categorical → log π_θ(a|s)
    Critic head → scalar V(s)
    Importance ratio r_t = exp(log π_θ - log π_θ_old)
    L^CLIP = E[min(r_t Â_t, clip(r_t,0.9,1.1)·Â_t)]
    L^VF = MSE(V_θ(s), R_t)
    L^H = entropy bonus (ent_coef annealed 0.02→0.001)
    total loss = -L^CLIP + 0.5·L^VF - L^H
    Adam update (lr=1e-4, max_grad_norm=0.5)
    │
    ▼
Every 250K steps: CheckpointManager.save(model) → step-N.zip + step-N.json + step-N.pkl
Every 100K steps: EvalAndVideoCallback fires:
    - 5 deterministic episodes on bare eval env (no VecNormalize, fixed seed)
    - log eval/mean_return, eval/std_return, eval/mean_length to TensorBoard
    - if new best: write best.zip + best.json + best.pkl
    - write eval-step-N.mp4
    │
    ▼
FastAPI backend reads checkpoints/ and tensorboard/ → serves /runs, /checkpoints,
  /runs/{name}/metrics, /episodes
    │
    ▼
Streamlit frontend pulls from backend API → Training, Play, Compare pages
```

---

## Appendix: version history and what each version fixed

| Version | Key change | Eval return (best) |
|---|---|---|
| v1 | Baseline PPO, MultiBinary(12), score_delta only | ~-10 (pinned at clip floor) |
| v2 | Reward shaping v2: survival_bonus, end_on_life_lost=False, wider clip | ~1174 (stochastic only) |
| v3 | DiscreteActionWrapper, Discrete(9) always-fire combos | — |
| v4 | Diagnosed: held-B fires only once → zero kills | ~0 |
| v5 | AutoFireWrapper (period=4), score_delta restored | ~244 |
| v6 | EntCoefLinearSchedule (0.02→0.001) | ~275 |
| v7 | AutoFire period 4→24 (2.5 Hz) — stop saturating bullet array | ~1248 |
| v8 | Period=18, survival_bonus 0.01→0.03, 4M steps | ~4271 |
| v9 | IMPALA ResNet + action_repeat=8 + VecNormalize + LR=1e-4 | **7168 @ 2.7M** (best — clears the game) |
| v10 | RecurrentPPO + LSTM(256) + frame_stack=1 | 1993 @ 4M (retired — cold-start blindness) |
| v11 | RecurrentPPO + LSTM(256) + frame_stack=4 + n_epochs=2 | 5325 @ 5.7M (+25% vs v8, −26% vs v9) |
| v12 | Temporal-attention extractor + frame_stack=8 (**standard PPO**) | **7609 @ 5.3M** (new best, +6.2% vs v9; 6M steps, lr=5e-5) |

---

*For the detailed IMPALA ResNet architecture and PPO math in isolation, see
[docs/v9_procedure_pipeline.md](v9_procedure_pipeline.md).*

---

## Appendix: v10 — RecurrentPPO + LSTM (discontinued)

**Result: did not work well. Run stopped at 2.8M / 4M steps.**

v10 replaced PPO with `sb3_contrib.RecurrentPPO` (LSTM hidden=256) and dropped
`frame_stack` from 4 to 1, on the hypothesis that the LSTM hidden state should
own all temporal memory. At 2.8M steps: `eval/mean_return=1993` vs v9's 7028,
`eval/mean_length=555` vs v9's 4500, `clip_fraction=0.363` stuck elevated since
1.2M steps.

Three failure modes were identified:

1. **`frame_stack=1` cold-start blindness.** At episode reset `h=0, c=0` — the
   LSTM has zero temporal context for the first several decision steps. A single
   84×84 grayscale frame contains no velocity information; the agent flies blind
   on every new life. This was the primary driver of the survival collapse.

2. **LSTM re-execution drift (`clip_fraction=0.363`).** RecurrentPPO re-runs the
   LSTM from stored initial states during the 4 optimization epochs. Each pass
   with updated weights produces slightly different hidden states, widening the
   importance ratio `r_t = π_θ/π_θ_old` beyond the [0.9, 1.1] clip band. The
   policy oscillated rather than converging.

3. **Stochastic/deterministic gap.** `rollout/ep_rew_mean=3311` vs
   `eval/mean_return=1993` — the argmax deterministic policy was significantly
   worse than the sampled stochastic policy, indicating the committed actions
   were wrong ones.

**Lesson:** `frame_stack=1 + LSTM` requires more samples than a 4M step budget
allows for a fast-paced game like Airstriker. Frame stacking and LSTM operate at
different timescales (fast motion vs slow spawn patterns) and are complementary,
not redundant. See **v11** for the corrected approach:
`frame_stack=4 + LSTM + n_epochs=2 + lr=5e-5`.

### A.1 Why LSTM in RL at all

Standard PPO with frame stacking approximates the Markov property by
concatenating k recent frames:

```
obs = [frame_{t-3}, frame_{t-2}, frame_{t-1}, frame_t]  →  shape (4, 84, 84)
```

This is a **fixed-length, handcrafted memory**. It handles motion and velocity
(4 frames gives direction of movement) but fails at anything requiring
longer-range memory:

- an enemy that appeared 2 seconds ago and will reappear
- a spawn pattern that repeats every N frames
- threat history that exceeds the stack depth

Frame stacking also has a structural limitation: the designer decides at
design time how far back the agent can "see." If the relevant signal is 10
frames back but the stack is 4, information is lost by construction.

LSTM provides **learned, adaptive, variable-length memory** — the network
decides what to remember and for how long, rather than the designer
hardcoding it.

### A.2 The LSTM cell — the math

An LSTM cell takes three inputs at each step:

- `x_t` — current input (CNN features, shape `[features_dim]`)
- `h_{t-1}` — previous hidden state `[lstm_size]` — the network's "output," passed forward
- `c_{t-1}` — previous cell state `[lstm_size]` — the long-term memory "conveyor belt"

Four learned gates (each a linear layer + nonlinearity):

```
Forget gate:  f_t = σ( W_f · [h_{t-1}, x_t] + b_f )    ← what to erase from cell state
Input gate:   i_t = σ( W_i · [h_{t-1}, x_t] + b_i )    ← how much new info to write
Candidate:    g_t = tanh( W_g · [h_{t-1}, x_t] + b_g ) ← what new values to write
Output gate:  o_t = σ( W_o · [h_{t-1}, x_t] + b_o )    ← what to expose from cell state
```

All four are computed in one matrix multiply: `W` has shape
`[4·lstm_size, features_dim + lstm_size]`.

State updates:

```
Cell state:   c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
                    ───────────────     ─────────────
                    old memory          new memory
                    scaled by forget    scaled by input gate

Hidden state: h_t = o_t ⊙ tanh(c_t)
```

`⊙` = element-wise multiply. `σ` = sigmoid ∈ [0,1]. `tanh` ∈ [-1,1].

**Why LSTM solves vanishing gradients vs vanilla RNN:**

Vanilla RNN: `h_t = tanh(W_h · h_{t-1} + W_x · x_t)`

Backward gradient: `∂h_t/∂h_{t-1} = W_h · diag(1 − tanh²(h_t))`

If `||W_h|| < 1` or tanh saturates → gradient shrinks at every step →
vanishes after ~10 steps.

LSTM cell state gradient: `∂c_t/∂c_{t-1} = f_t` (element-wise, no matrix
multiply on the backward pass). When the forget gate `f_t ≈ 1` (network
decides "keep this memory"), gradients flow through `c` with **no decay** —
the cell state is a gradient highway. The network learns to open the forget
gate when it needs to preserve a signal across many steps.

### A.3 Architecture in v10

```
obs (1, 84, 84) uint8         ← frame_stack=1 (single frame, not 4)
    ↓  /255
IMPALA CNN  →  256-d features ← same backbone as v9
    ↓
LSTM(input=256, hidden=256)
    takes: (features_t,  h_{t-1},  c_{t-1})
    emits: (h_t,  c_t)
    ↓
h_t (256-d)
    ├─► Actor MLP [64, 64] Tanh → 9 logits → Categorical → action
    └─► Critic MLP [64, 64] Tanh → scalar V(s)
```

**Why frame_stack=1 with LSTM:** With frame_stack=4, the LSTM input already
contains temporal signal and partially re-learns what stacking encodes. With
frame_stack=1, the LSTM is the **sole source of temporal memory** — it must
infer velocity, motion direction, and spawn patterns entirely from the
sequence of single frames. Harder to train, but tests whether the LSTM
actually learns useful memory rather than duplicating the stack.

**Parameter count:** LSTM weight matrix `W` has shape
`[4·256, 256+256] = [1024, 512]` → ~524K parameters for the LSTM alone,
on top of the CNN (~97K conv + ~991K FC) and heads (~42K). v10 total ≈ 1.66M
parameters vs v9's 1.13M.

### A.4 How RecurrentPPO changes the training loop

Standard PPO treats every transition in the buffer **independently** — the
CNN processes each observation without knowing what came before. This allows
random shuffling of minibatches and parallel independent forward passes.

RecurrentPPO **cannot** do this. The LSTM at step t needs `h_{t-1}` from
step t-1. The forward pass is **order-dependent and stateful**. This changes
four things fundamentally.

#### Hidden state threading during rollout collection

During collection, `(h, c)` must be threaded step-to-step and zeroed at
episode boundaries:

```python
lstm_states = (zeros(n_layers, n_envs, lstm_size),
               zeros(n_layers, n_envs, lstm_size))

for step in range(n_steps):   # 128 steps
    features = cnn(obs)
    action, value, log_prob, lstm_states = policy.forward(
        features, lstm_states, episode_starts[step]
    )
    # episode boundary masking — applied inside forward():
    #   h = h * (1 - episode_starts)   ← zero on death, fresh start for ep2
    #   c = c * (1 - episode_starts)

    buffer.store(..., lstm_states, episode_starts[step])  # ← stored per step
    obs, reward, done = env.step(action)
    episode_starts[step+1] = done
```

Carrying hidden state from a dead episode into a new one would poison the
memory with irrelevant context — the zero-masking at `episode_starts` is
load-bearing.

#### Sequence-based minibatches (not random shuffle)

Standard PPO: shuffle 1024 transitions randomly → 4 minibatches of 256.

RecurrentPPO: must maintain temporal order within each env's trajectory.

```
1024 transitions = 8 envs × 128 steps
→ 8 sequences of length 128 (one per env)
→ seqs_per_minibatch = batch_size / seq_len = 256 / 128 = 2
→ 4 minibatches per epoch  (sequences shuffled, steps within each kept ordered)
```

Each minibatch uses the stored `(h, c)` at the start of its sequences as
initial LSTM state — loaded directly from the buffer, not recomputed.

#### Truncated Backpropagation Through Time (TBPTT)

The full 128-step LSTM sequence is used for the forward pass, but the
backward pass runs through the **entire sequence** (`seq_len = n_steps = 128`
in this config). The stored initial `(h_start, c_start)` is **detached**
from the computation graph — gradients do not flow beyond the sequence
boundary into previous rollouts.

```
Forward:   step 0 → step 1 → ... → step 127  (h, c thread through all steps)
Backward:  gradients flow back through all 128 steps, then stop at h_start
                                                         (detached — TBPTT boundary)
```

#### Why v10's clip_fraction is elevated (0.35)

During optimization the LSTM is re-run from stored initial states. After one
gradient update, changed LSTM weights produce different hidden states for the
same sequence → the effective policy distribution shifts further from
`π_θ_old` than in standard PPO even for the same parameter-change magnitude.
Result: `r_t = π_θ / π_θ_old` drifts further from 1.0 → more transitions
hit the clip → higher `clip_fraction`. This is expected behavior, not a bug.
If it climbs above 0.5, the right intervention is lowering LR or reducing
`n_epochs` from 4 to 2.

### A.5 Data size flow — full diagram

```
═══════════════════════════════════════════════════════════════════════════════
  PHASE 1 — ROLLOUT COLLECTION  (one decision step, all 8 envs in parallel)
═══════════════════════════════════════════════════════════════════════════════

  SubprocVecEnv (8 workers)
  ┌──────────────────────────────────────────────────────┐
  │  obs_t    (8, 1, 84, 84)  uint8   ← 1 frame, 8 envs │
  │  h_{t-1}  (1, 8, 256)    float32 ← LSTM hidden state │
  │  c_{t-1}  (1, 8, 256)    float32 ← LSTM cell state   │
  └──────────────────┬───────────────────────────────────┘
                     │
                     ▼ /255 → float32
              ┌──────────────┐
              │  IMPALA CNN  │  processes all 8 envs in parallel (batch=8)
              ├──────────────┤
              │ (8, 1,84,84) │
              │      ↓ Stack 1: Conv(1→16,3×3) + MaxPool(/2) + 2 ResBlocks
              │ (8,16,42,42) │
              │      ↓ Stack 2: Conv(16→32,3×3) + MaxPool(/2) + 2 ResBlocks
              │ (8,32,21,21) │
              │      ↓ Stack 3: Conv(32→32,3×3) + MaxPool(/2) + 2 ResBlocks
              │ (8,32,11,11) │
              │      ↓ ReLU + Flatten
              │  (8, 3872)   │
              │      ↓ Linear(3872→256) + ReLU
              │  (8, 256)    │  ← features_t
              └──────┬───────┘
                     │
                     ▼ reshape to (seq=1, batch=8, input=256)
              ┌──────────────┐
              │     LSTM     │  one step forward
              ├──────────────┤
              │  input:      │  (1, 8, 256)    ← features_t
              │  h_{t-1}:    │  (1, 8, 256)    ← carried from previous step
              │  c_{t-1}:    │  (1, 8, 256)      (zeroed at episode_starts)
              │              │
              │  output h_t: │  (1, 8, 256)  →  squeeze  →  (8, 256)
              │  output c_t: │  (1, 8, 256)
              └──────┬───────┘
                     │  h_t  (8, 256)
               ┌─────┴──────┐
               ▼            ▼
        ┌──────────┐  ┌──────────┐
        │  Actor   │  │  Critic  │
        │  MLP     │  │  MLP     │
        ├──────────┤  ├──────────┤
        │ (8,256)  │  │ (8,256)  │
        │ Linear   │  │ Linear   │
        │ (8, 64)  │  │ (8, 64)  │
        │ Tanh     │  │ Tanh     │
        │ (8, 64)  │  │ (8, 64)  │
        │ Linear   │  │ Linear   │
        │ (8, 64)  │  │ (8, 64)  │
        │ Tanh     │  │ Tanh     │
        │ Linear   │  │ Linear   │
        │  (8, 9)  │  │  (8, 1)  │
        │ logits   │  │  V(s_t)  │
        └────┬─────┘  └────┬─────┘
             │ Categorical  │ squeeze
             ▼              ▼
         action (8,)    value (8,)
         log_prob (8,)

  Store into rollout buffer at slot t:
    obs[t]            ← (8, 1, 84, 84)  uint8
    actions[t]        ← (8,)            int64
    rewards[t]        ← (8,)            float32
    episode_starts[t] ← (8,)            float32  (1.0 if just reset)
    values[t]         ← (8,)            float32
    log_probs[t]      ← (8,)            float32
    hidden[t]         ← (1, 8, 256)     float32  ← h_t  ★ NEW vs standard PPO
    cell[t]           ← (1, 8, 256)     float32  ← c_t  ★ NEW vs standard PPO

  Repeat for t = 0 → 127  (128 steps total)


═══════════════════════════════════════════════════════════════════════════════
  PHASE 2 — FULL BUFFER  (after 128 steps × 8 envs)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │  Field             Shape                dtype    Size           │
  ├─────────────────────────────────────────────────────────────────┤
  │  observations   (128, 8, 1, 84, 84)    uint8    ~29 MB         │
  │  actions        (128, 8)               int64    ~  8 KB        │
  │  rewards        (128, 8)               float32  ~  4 KB        │
  │  episode_starts (128, 8)               float32  ~  4 KB        │
  │  values         (128, 8)               float32  ~  4 KB        │
  │  log_probs      (128, 8)               float32  ~  4 KB        │
  │  hidden_states  (128, 1, 8, 256)       float32  ~  1 MB  ★     │
  │  cell_states    (128, 1, 8, 256)       float32  ~  1 MB  ★     │
  ├─────────────────────────────────────────────────────────────────┤
  │  Total ≈ 31 MB   (vs ~29 MB for standard PPO)                   │
  └─────────────────────────────────────────────────────────────────┘
                     │
                     ▼  GAE backward pass (uses values + rewards + episode_starts)
  ┌─────────────────────────────────────────────────────────────────┐
  │  advantages     (128, 8)               float32  ← Â_t          │
  │  returns        (128, 8)               float32  ← R_t=Â_t+V_t  │
  └─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
  PHASE 3 — SEQUENCE CONSTRUCTION  (reshape buffer for optimization)
═══════════════════════════════════════════════════════════════════════════════

  Buffer layout:  (128 steps, 8 envs)
  Viewed as:       8 sequences × 128 steps each
                   (each env's full trajectory = one sequence)

  seq_len            = n_steps              = 128
  n_sequences        = n_envs               = 8
  seqs_per_minibatch = batch_size / seq_len = 256 / 128 = 2
  n_minibatches      = n_sequences / seqs_per_minibatch = 8 / 2 = 4

  Per epoch: shuffle the 8 sequences, split into 4 minibatches of 2:

  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
  │  Minibatch 1    │  Minibatch 2    │  Minibatch 3    │  Minibatch 4    │
  │  env3 + env7    │  env1 + env5    │  env0 + env6    │  env2 + env4    │
  │  2×128=256 rows │  2×128=256 rows │  2×128=256 rows │  2×128=256 rows │
  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
  Total per epoch: 4 × 256 = 1024 transitions ✓


═══════════════════════════════════════════════════════════════════════════════
  PHASE 4 — OPTIMIZATION FORWARD PASS  (one minibatch: 2 sequences × 128 steps)
═══════════════════════════════════════════════════════════════════════════════

  Load from buffer:
    obs            (128, 2, 1, 84, 84)  uint8
    h_start        (  1, 2,    256)     float32  ← h at step 0 of each sequence
    c_start        (  1, 2,    256)     float32  ← c at step 0 of each sequence
    episode_starts (128, 2)             float32
    actions_old    (128, 2)             int64
    log_probs_old  (128, 2)             float32
    advantages     (128, 2)             float32
    returns        (128, 2)             float32

                     │
                     ▼ reshape: merge seq+batch for CNN  (128×2 = 256 frames at once)
              ┌──────────────┐
              │  IMPALA CNN  │  batch = seq_len × seqs_per_mb = 256
              ├──────────────┤
              │ (256,1,84,84)│  → same conv stack as Phase 1
              │      ↓
              │  (256, 256)  │  ← features for all 256 frames
              └──────┬───────┘
                     │
                     ▼ reshape back to (seq_len, batch, features) = (128, 2, 256)
              ┌──────────────────────────────────────────────┐
              │                  LSTM                        │
              ├──────────────────────────────────────────────┤
              │  input:   (128, 2, 256)  ← all 128 steps     │
              │  h_start: (  1, 2, 256)  ← detached (TBPTT)  │
              │  c_start: (  1, 2, 256)  ← detached (TBPTT)  │
              │  mask:    (128, 2)       ← episode_starts     │
              │                                               │
              │  internal loop: for t in 0..127:             │
              │    h = h * (1 − mask[t])  ← zero on reset    │
              │    h, c = cell(input[t], h, c)               │
              │                                               │
              │  output:  (128, 2, 256)  ← h_t all steps     │
              │  h_T,c_T: (  1, 2, 256)  ← final state       │
              └──────┬───────────────────────────────────────┘
                     │  (128, 2, 256)
                     ▼ reshape → (256, 256)  merge seq+batch for MLP
               ┌─────┴──────┐
               ▼            ▼
        ┌──────────┐  ┌──────────┐
        │  Actor   │  │  Critic  │
        │  MLP     │  │  MLP     │
        │ (256,256)│  │ (256,256)│
        │    ↓     │  │    ↓     │
        │ (256, 9) │  │ (256, 1) │
        │  logits  │  │ V_θ(s)   │
        └────┬─────┘  └────┬─────┘
             │              │ squeeze
             ▼              ▼
       log_π_θ  (256,)   V_θ   (256,)
             │
             ▼
  r_t   = exp(log_π_θ − log_π_old)   (256,)  ← importance ratio
  Â     = advantages.flatten()        (256,)
  R     = returns.flatten()           (256,)

  L^CLIP = mean[ min(r_t·Â,  clip(r_t, 0.9, 1.1)·Â) ]   scalar
  L^VF   = mean[ (V_θ − R)² ]                             scalar
  L^H    = −mean[ entropy(logits) ]                        scalar
  Loss   = −L^CLIP + 0.5·L^VF + ent_coef·L^H             scalar
             │
             ▼  backward() — BPTT through all 128 LSTM steps
  Gradients flow through:
    Linear heads  ←  LSTM weights (W_f,W_i,W_g,W_o, 128 steps)  ←  CNN
             │
             ▼  optimizer.step()  →  θ updated


═══════════════════════════════════════════════════════════════════════════════
  SUMMARY — counts per rollout iteration
═══════════════════════════════════════════════════════════════════════════════

  Rollout:              128 steps × 8 envs  =  1,024 transitions collected
  Epochs:               4
  Minibatches per epoch:                       4  (2 sequences each)
  Gradient steps total:                        16
  Transitions seen per epoch:               1,024
  Transitions seen total:                   4,096  (each used 4×, then discarded)

  CNN forward passes:
    collection:      128 steps × 8 envs  =  1,024  (batch=8,   one step at a time)
    optimization:    4 epochs × 4 mb     =     16  (batch=256, full seq in parallel)
    total CNN calls: 1,024 + 16×256 frames =  5,120 frames per iteration

  LSTM forward passes:
    collection:      128 individual steps          (seq=1,   batch=8)
    optimization:    4 epochs × 4 mb × 128 steps  (seq=128, batch=2) ← 16 full unrolls
```

### A.6 Standard PPO vs RecurrentPPO — what changes

| Aspect | Standard PPO (v9) | RecurrentPPO (v10) |
|---|---|---|
| **Policy memory** | Frame stack (k=4 frames, fixed) | LSTM hidden state (learned, adaptive) |
| **Forward pass** | Independent per observation | Sequential, stateful — h,c threaded |
| **Buffer extras** | None | Per-step (h,c) for each env — ~2 MB |
| **Minibatch structure** | Random shuffle of 1024 transitions | Ordered sequences of length 128 |
| **Gradient flow** | Each obs independently | BPTT through 128 LSTM steps |
| **Episode boundary** | No action needed | Zero h,c at episode_starts |
| **Training cost** | O(batch_size) per update | O(batch_size × seq_len) per update |
| **clip_fraction** | Typically 0.10–0.25 | Elevated ~0.35 (LSTM re-execution drift) |
| **eval/std_return** | 0 (deterministic fixed-seed) | > 0 (sticky-action eval fix) |

---

## Appendix: v12 — Temporal Attention (transformer over the frame window)

**Status: complete. Best eval return 7609 @ 5.3M — new numerical best, +6.2% over
v9 (see B.7). The hypothesis held: attention over the frame window beats pure
IMPALA feature learning, at the cost of ~2× the sample budget — and B.2.8 explains
*why* that sample-cost tax is the expected price of the weaker inductive bias.**

v10 and v11 asked whether a *recurrent* memory (LSTM) could beat the stateless
v9 baseline. Both fell short, and the post-mortem (A.1–A.6 above, plus the v11
result in the version table) pinned the cause on one structural defect: the
LSTM's hidden state is **zero at every episode reset** (`h=0, c=0`), so the
policy is briefly blind at the start of every life and — worse — *deterministic*
eval episodes that never recover from that cold start die early, producing the
bimodal eval curve that capped v11 at 5325 vs v9's 7168.

v12 keeps the *goal* of v10/v11 — give the policy temporal context beyond a few
frames — but changes the *mechanism* from recurrence to **self-attention over a
fixed window of frames**. A transformer that attends over the last `K` frames
has **no hidden state to initialize**, so the cold-start defect disappears by
construction. And because the whole thing fits inside a feature extractor that
consumes one (stacked) observation at a time, it runs on **ordinary PPO** — none
of the RecurrentPPO machinery in A.4–A.5 is needed.

This appendix is a from-first-principles treatment of the transformer, then the
specific architecture and how it slots into the existing pipeline.

### B.1 Why attention, not recurrence — the cold-start argument

Recall the three memory mechanisms this project has used:

| Mechanism | Memory type | Reach | Cold start |
|---|---|---|---|
| Frame stack (v1–v9) | fixed, handcrafted | exactly `k` frames | none — window is filled with the first frame at reset |
| LSTM (v10/v11) | learned, recurrent | unbounded in principle | **severe** — `h,c=0`, context must rebuild step-by-step |
| Attention (v12) | learned, windowed | exactly `K` frames, *content-addressed* | none — window is filled with the first frame at reset |

The LSTM compresses all history into a fixed-size state vector `(h, c)` that is
updated *recurrently*: `h_t = f(h_{t-1}, x_t)`. The strength is unbounded reach;
the weakness is that the state must be **built up sequentially** and is **wrong
at `t=0`** (it is zero). For a fast game where an episode is a single life, the
agent pays the cold-start tax on *every* episode. Critically, `h=0` is an
out-of-distribution starting point the LSTM was rarely trained from (training
envs run continuously with warm state), so early hidden states `h₁, h₂, …` are
also off — the network is not merely uninformed, it is in a part of hidden-state
space it has barely visited. The transformer sidesteps this entirely: the
FrameStack buffer resets to `[x₁, x₁, …, x₁]` — K copies of the real first
observation — which the network encounters at the start of *every* episode and
trains on constantly. As the episode progresses the buffer fills naturally
(`[x₁,x₁,...,x₂] → [x₁,...,x₂,x₃] → …`), and the attention is reading real,
in-distribution frames from step 0 onward, with no recovery required.

Attention takes the opposite stance: keep the last `K` frames *explicitly*
(exactly as frame stacking already does), and let the network learn, **at every
single step**, which of those `K` frames are relevant to the current decision —
by content, not by recency. There is no state to carry between steps, so there
is nothing to be wrong at reset. The window is always full (the `FrameStack`
wrapper pads it with the first frame, §1.5 [8]). This is the same property that
makes frame stacking robust, but with a *learned, content-addressed* read over a
**longer** window (v12 uses `K=8` vs v9's 4) instead of a fixed convolution over
a short one.

The tradeoff for giving up the LSTM's unbounded reach: the window is finite, so
patterns longer than `K` decision steps (~1.06 s at `K=8`, `action_repeat=8`)
are invisible. The bet is that Airstriker's relevant temporal structure —
bullet motion, near-term spawn timing — lives inside that window, the same bet
that made frame stacking work for v9.

### B.2 Transformer fundamentals — the mathematics of self-attention

This section builds the transformer encoder from scratch. Everything v12 uses is
here; nothing else is needed. It is written to stand alone as a transformer
tutorial — read B.2.0–B.2.9 even if you skip the rest.

#### B.2.0 Origins and the encoder-only design

The transformer was introduced in *"Attention Is All You Need"* (Vaswani et al.,
2017) for machine translation. The original is an **encoder-decoder**: an encoder
maps the source sentence to a set of contextual token vectors, and a decoder
generates the target one token at a time, attending both to its own past outputs
(*self-attention*) and to the encoder's outputs (*cross-attention*). Three
families descend from it, and the distinction matters for understanding v12:

| Family | Example | Attention | Used for |
|---|---|---|---|
| **Encoder-only** | BERT, ViT, **v12** | full (bidirectional) self-attention | understanding a fixed input → one representation |
| **Decoder-only** | GPT, Llama | *causal* (masked) self-attention | autoregressive generation |
| **Encoder-decoder** | original Transformer, T5 | encoder self-attn + decoder cross-attn | sequence-to-sequence |

v12 is **encoder-only**, the same family as the **Vision Transformer (ViT)**.
The analogy is exact and worth holding onto:

- **ViT** splits an image into a grid of patches, linearly embeds each patch into
  a token, adds positional embeddings, and runs a transformer encoder; it reads a
  classification token (or pools) for the final representation.
- **v12** splits the *observation* into its `K=8` frames, embeds each frame into a
  token *with a shared CNN* (a richer embedder than ViT's linear patch projection,
  because a frame has spatial structure worth convolving), adds positional
  embeddings, runs a transformer encoder, and reads the last token.

So v12 is "ViT over time instead of over space": the sequence axis is the
**temporal** frame axis, not spatial patches. This framing also tells you what to
expect — ViT's well-known property is that it **needs more data than a CNN** to
reach the same accuracy, then surpasses it at scale. B.2.8 makes that precise and
ties it directly to v12's 2× sample cost.

**Causal vs full attention.** A *causal mask* forbids token `i` from attending to
any token `j > i` (the future), which is mandatory for autoregressive generation
(GPT must not peek ahead). v12 uses **full, non-causal** attention: every frame
may attend to every other. This is allowed because v12 is not generating a
sequence — it reads one observation and emits one feature vector. (B.3 notes the
subtlety that since we only *read* the last token, causal and full attention give
the identical result there anyway.)

#### B.2.1 Tokens

A transformer operates on a **sequence of tokens**, each a `d`-dimensional
vector. Stack them as the rows of a matrix:

```
X ∈ ℝ^{K×d}     K tokens, each of width d ("d_model")
```

In v12 a *token is one frame*: the per-frame CNN (B.3) turns each of the `K=8`
stacked frames into a `d=256`-dimensional vector. So `X` is the `8×256` matrix of
per-frame embeddings. The transformer's job is to mix information *across the 8
frames* so the final per-frame representation is aware of the others.

#### B.2.2 Queries, keys, values

Self-attention is a differentiable, content-based lookup. Each token emits three
projections of itself, via learned weight matrices `W_Q, W_K, W_V ∈ ℝ^{d×d_k}`:

```
Q = X W_Q     (K × d_k)    query  — "what is this token looking for?"
K = X W_K     (K × d_k)    key    — "what does this token offer as a label?"
V = X W_V     (K × d_v)    value  — "what content does this token carry?"
```

The mental model: token `i`'s **query** `q_i` is compared against every token's
**key** `k_j`; the better they match, the more of token `j`'s **value** `v_j`
gets mixed into token `i`'s output. This is a soft dictionary lookup where the
match is a learned dot product instead of an exact key equality.

**Why three separate projections, and why the asymmetry matters.** A natural
question: why not just use the token embedding `x_i` directly for all three roles?
Two reasons.

1. **Query and key play different roles, so they need different projections.**
   "What frame 7 is looking for" (its query) is a different question from "what
   label frame 3 advertises" (its key). Tying them (`W_Q = W_K`) would force the
   score matrix `S = XW_Q W_Kᵀ Xᵀ` to be **symmetric** (`S_{ij}=S_{ji}`), i.e.
   "`i` attends to `j`" exactly as much as "`j` attends to `i`." Temporal relations
   are *directional* — the current frame should attend strongly to the previous
   frame to read velocity, but the previous frame need not attend back. Separate
   `W_Q, W_K` let the learned bilinear form `q_i·k_j = x_iᵀ(W_Q W_Kᵀ)x_j` be
   **asymmetric**, which is what directional relations require.

2. **The value is a separate "what to retrieve" channel.** `W_V` decouples *how
   tokens are matched* (Q·K) from *what information flows once matched* (V). A
   frame might be matched on its bullet positions but contribute its motion
   features to the output — the matching content and the retrieved content need
   not be the same, so they get their own subspace.

**The kernel-smoothing view (optional, for intuition).** Attention is exactly a
**Nadaraya–Watson kernel regression**: the output `Out_i = Σ_j A_{ij} v_j` is a
kernel-weighted average of the values, with kernel weight `A_{ij} ∝
exp(q_i·k_j/√d_k)` — an (unnormalized) Gaussian-like similarity kernel in the
projected space. Self-attention is "smoothing each token toward the values of the
tokens it is most similar to, where similarity is learned." This is why attention
is sometimes called a *soft, content-addressable* memory: it is associative
recall, differentiable end-to-end.

#### B.2.3 Scaled dot-product attention — the core formula

```
                    ⎛ Q Kᵀ ⎞
Attention(Q,K,V) = softmax⎜ ──── ⎟ V                    (K × d_v)
                    ⎝ √d_k ⎠
```

Read it in three stages:

**(1) Scores.** `S = Q Kᵀ ∈ ℝ^{K×K}`. Entry `S_{ij} = q_i · k_j` is the raw
compatibility of query `i` with key `j` — a single dot product, large when the
two vectors point the same way.

**(2) Scale, then softmax over each row.**

```
A_{ij} = exp(S_{ij}/√d_k) / Σ_{j'} exp(S_{ij'}/√d_k)        Σ_j A_{ij} = 1
```

Each row of `A` is a probability distribution: `A_{ij}` is "how much token `i`
attends to token `j`." These are the **attention weights**.

**(3) Weighted sum of values.** `Out_i = Σ_j A_{ij} v_j`. Each token's output is
a convex combination of all tokens' value vectors, weighted by relevance.

**Why divide by `√d_k`?** The dot product `q_i·k_j = Σ_{m=1}^{d_k} q_{im}k_{jm}`
sums `d_k` product terms. Make the standard assumption that the components are
independent, mean-0, unit-variance. Then:

```
E[q_i·k_j]   = Σ_m E[q_{im}] E[k_{jm}]                 = 0
Var(q_i·k_j) = Σ_m Var(q_{im} k_{jm})
             = Σ_m E[q_{im}²] E[k_{jm}²]   (indep, mean 0)
             = Σ_m 1·1  =  d_k     ⇒  std = √d_k
```

So the raw scores have standard deviation `√d_k`. Left unscaled, the scores grow
with `d_k`, pushing softmax into its **saturated** regime where one weight ≈ 1 and
the rest ≈ 0. In that regime the softmax Jacobian is nearly zero, so **gradients
vanish** and the layer can't learn which tokens to attend to. Dividing by `√d_k`
renormalizes the score variance back to ~1, keeping softmax in its sensitive,
high-gradient range. This is the single most important numerical detail in the
attention mechanism.

**Softmax temperature view.** `softmax(S/τ)` with temperature `τ` interpolates
between a uniform average (`τ→∞`, all weights equal) and a hard argmax (`τ→0`, one
weight = 1). The `√d_k` divisor is precisely a temperature that holds the
*effective sharpness* constant as the head dimension changes — without it, wider
heads would silently run "colder" (sharper, near-argmax) and stop learning.

**A worked micro-example (`K=3`, `d_k=2`).** Suppose after projection token 1's
query and the three keys are:

```
q₁ = [1, 0]      k₁ = [1, 0]     k₂ = [0, 1]     k₃ = [1, 1]
```

Step 1 — scores (dot products):  `q₁·k₁ = 1`, `q₁·k₂ = 0`, `q₁·k₃ = 1`.
Step 2 — scale by `√d_k = √2 ≈ 1.414`:  `[0.707, 0, 0.707]`.
Step 3 — softmax:  `exp([0.707, 0, 0.707]) = [2.03, 1.00, 2.03]`, sum `= 5.06`, so

```
A₁ = [0.401, 0.198, 0.401]
```

Token 1 splits its attention ~40/20/40: it attends most to tokens 1 and 3 (whose
keys align with its query `[1,0]`) and least to token 2 (orthogonal key `[0,1]`).
Its output is the convex combination `0.401·v₁ + 0.198·v₂ + 0.401·v₃`. Repeating
this for `q₂, q₃` fills the `3×3` weight matrix `A`, one softmax-normalized row per
query. In v12 this is an `8×8` matrix per head — one row per frame, each row a
distribution over which of the 8 frames that frame pulls information from.

#### B.2.4 Multi-head attention

A single attention function can only express one notion of "relevance." We want
several in parallel — e.g. one head that tracks the immediately preceding frame
(velocity), another that looks ~6 frames back (a spawn that is about to repeat).
**Multi-head attention** runs `H` independent attention functions in subspaces of
width `d_k = d/H`, then concatenates and projects:

```
head_h = Attention(X W_Q^h, X W_K^h, X W_V^h)        each (K × d_k)
MHA(X) = Concat(head_1, …, head_H) · W_O             (K × d)
```

with `W_O ∈ ℝ^{d×d}`. v12 uses `d=256`, `H=4`, so each head works in `d_k=64`
dimensions. Each head has its own `W_Q^h, W_K^h, W_V^h`, so the four heads learn
four different "what is relevant" relations over the same 8 frames.

**Why split into subspaces instead of one big head?** Two reasons, one
expressive, one statistical:

- **A single softmax is unimodal-ish — it tends to commit.** One attention head
  produces *one* `K×K` weight matrix; because of the softmax it struggles to
  simultaneously "attend strongly to frame 7 *and* to frame 2 for *different*
  reasons." Splitting into `H` heads gives `H` independent weight matrices, so the
  layer can track several relations at once — e.g. head A locks onto the
  immediately previous frame (instantaneous velocity), head B onto ~6 frames back
  (a spawn cycle about to repeat), head C onto the current frame itself
  (pass-through). The outputs are concatenated, so the heads are *additive
  channels of relevance*, not competing for one budget.

- **Cost is held constant.** Because `d_k = d/H`, the total work `H·(K·d_k²) =
  K·d·d_k` and parameter count are (to first order) independent of `H`: you are
  *reshaping* the same `d`-dimensional computation into `H` parallel `d/H`-wide
  ones, not adding compute. The trade is rank-vs-count: each head's `QKᵀ` score
  map is at most rank `d_k=64`, so very fine-grained single-relation matching is
  slightly weaker, but you get four relations instead of one — empirically a good
  trade. `W_O` then linearly recombines the concatenated head outputs back into
  the `d`-dimensional residual stream.

#### B.2.5 Position: attention is order-blind by default

Self-attention as defined is **permutation-equivariant**: permute the input
tokens and the outputs permute identically, because `Out_i = Σ_j A_{ij} v_j` has
no term that depends on the *indices* `i, j` — only on the token *contents*.
That is fatal for a temporal sequence: frame order (was the bullet here *then*
there, or there *then* here?) carries the velocity signal.

The fix is a **positional encoding** added to the token embeddings before
attention:

```
X' = X + P        P ∈ ℝ^{K×d}, row P_k encodes "this is frame k"
```

Two common choices: fixed **sinusoidal** encodings (original Transformer, good
for extrapolating to unseen lengths) or **learned** positional embeddings (a
trainable parameter, one `d`-vector per position).

**Sinusoidal encoding (for reference).** The original Transformer defines, for
position `pos` and embedding dimension index `i ∈ [0, d)`:

```
PE(pos, 2i)   = sin( pos / 10000^{2i/d} )
PE(pos, 2i+1) = cos( pos / 10000^{2i/d} )
```

Each dimension is a sinusoid whose wavelength grows geometrically from `2π` (high
`i`-frequency, fine position) to `~10000·2π` (low frequency, coarse position) — a
"binary clock in continuous form." Its elegant property: for any fixed offset `k`,
`PE(pos+k)` is a **linear function** of `PE(pos)` (a rotation by a matrix that
depends only on `k`, from the angle-addition identities). That means a downstream
linear map can express "attend `k` positions back" *independent of absolute
position* — relative offsets become linearly decodable. It also extrapolates to
sequence lengths unseen in training, since the formula is defined for all `pos`.

**v12 uses learned embeddings** — a trainable parameter, one `d`-vector per slot.
The window length `K=8` is fixed, so there is nothing to extrapolate to, and a
learnable per-slot bias is the simplest thing that works; the network discovers
whatever positional code is useful for reading frame order:

```python
self.pos_emb = nn.Parameter(torch.zeros(1, K, d))   # learned, K×d
tokens = tokens + self.pos_emb                       # inject order
```

(A note on the modern alternative: large LLMs increasingly use **rotary position
embeddings (RoPE)**, which rotate Q and K by position-dependent angles so that the
*dot product* `q_i·k_j` depends only on the relative offset `i−j`. v12 doesn't need
it — at `K=8` a learned table is trivial — but it's the same underlying goal as the
sinusoidal linearity property above: make relative position cheap to read.)

#### B.2.6 The transformer encoder layer

One encoder layer wraps multi-head attention and a position-wise feed-forward
network (FFN), each inside a **residual connection** with **LayerNorm**. v12 uses
the **pre-LN** ("norm-first") variant:

```
z   = x + MHA( LayerNorm(x) )            ← attention sublayer + residual
out = z + FFN( LayerNorm(z) )            ← feed-forward sublayer + residual

FFN(u) = W₂ · GELU(W₁ u + b₁) + b₂        W₁ ∈ ℝ^{d_ff×d}, W₂ ∈ ℝ^{d×d_ff}
```

Each piece, and why it is there:

- **Residual connections (`x + …`).** Same gradient-highway logic as the IMPALA
  residual block (§2.3a): the `+x` gives `∂out/∂x` an additive `+1` term, so
  gradients flow to early layers even if a sublayer saturates. This is what makes
  deep transformer stacks trainable.

- **LayerNorm**, not BatchNorm. LayerNorm normalizes across the `d` features of
  *each token independently*: `LN(x) = γ ⊙ (x−μ)/σ + β` with `μ, σ` computed over
  that token's own `d` components. It uses **no batch statistics**, so — exactly
  as argued for avoiding BatchNorm in the IMPALA backbone (§2.3c) — it is stable
  under RL's small, correlated, non-stationary minibatches.

- **Pre-LN vs post-LN.** Original Transformers put LayerNorm *after* the residual
  add (`LN(x + sublayer(x))`); pre-LN puts it *inside* (`x + sublayer(LN(x))`).
  Pre-LN keeps a clean, un-normalized residual path from input to output, which
  makes gradients better-behaved and removes the need for learning-rate warmup.
  It is the modern default and what v12 selects via `norm_first=True`. (This is
  also why the readout in B.3 can safely take a raw token off the residual
  stream.)

- **The FFN** is a 2-layer MLP applied to **each token independently** (same
  weights for all positions). Attention *mixes information across tokens*; the
  FFN then *transforms each token's mixed representation* nonlinearly. v12 uses
  `d_ff = 2d = 512` and a **GELU** activation. Roughly: attention = "communication
  between frames," FFN = "computation within a frame." The FFN is also where most
  of a transformer's parameters and arguably its stored "knowledge" live — it is a
  per-token nonlinear feature transform, the wider `d_ff` giving it room to compute.

- **GELU** (Gaussian Error Linear Unit) is `GELU(x) = x·Φ(x)`, where `Φ` is the
  standard-normal CDF — i.e. it scales each input by the probability that a
  standard Gaussian is below it. Unlike ReLU's hard gate at 0, GELU is **smooth
  and non-monotonic** near the origin (it dips slightly negative for small
  negative `x`), which gives cleaner gradients and is the de-facto default in
  transformers (BERT, GPT). A common tanh approximation:
  `0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])`.

**The residual-stream view (a useful modern mental model).** Pre-LN transformers
are clearest seen as a **residual stream**: a `d`-dimensional vector per token that
flows straight from input to output, and each sublayer *reads* a normalized copy
of it, *computes* an update, and *adds* that update back. Attention writes
"information gathered from other frames"; the FFN writes "nonlinear features of
this frame." Nothing overwrites the stream — sublayers only add to it — which is
exactly why gradients flow cleanly (the `+1` from each residual) and why stacking
more layers degrades gracefully rather than catastrophically. The final readout
(B.3) just takes one token off this stream.

**A note on normalization variants.** LayerNorm subtracts the mean and divides by
the standard deviation of each token's `d` features, then applies a learned scale
`γ` and shift `β`. Many recent LLMs replace it with **RMSNorm**
(`x / √(mean(x²)+ε) · γ` — no mean-centering, no bias), which is cheaper and works
about as well. v12 uses standard LayerNorm because that is what PyTorch's
`nn.TransformerEncoderLayer` provides; at this scale the difference is immaterial.

#### B.2.7 Computational cost

For a length-`K`, width-`d` sequence, one attention layer costs:

```
projections (Q,K,V,O):   O(K · d²)
scores QKᵀ and A·V:      O(K² · d)
```

At v12's `K=8, d=256`: the `K²d` term is `64·256 ≈ 16K` multiply-adds per layer —
negligible — and the `Kd²` projection term dominates at `~524K`. The transformer
is **cheap relative to the per-frame CNN** (B.3); the real compute cost of v12 is
encoding `K=8` frames through the CNN, not the attention. Note also that, unlike
the LSTM's inherently **sequential** `O(K)` recurrence, attention is a couple of
**parallel** matrix multiplies — there is no step-by-step unroll.

**The `O(K²)` wall (why it doesn't bite us, but matters in general).** The score
matrix `S = QKᵀ` is `K×K`, so attention's compute and memory scale **quadratically
in sequence length**. At `K=8` this is a rounding error, but it is *the* defining
scaling problem of transformers: doubling the context length quadruples attention
cost. This is why long-context LLMs invest in **efficient-attention** variants —
FlashAttention (an exact, IO-aware kernel that avoids materializing the full `K×K`
matrix), sparse/local attention (each token attends only to a neighborhood),
linear attention, etc. v12 needs none of this: the window is deliberately short
(`K=8` ≈ 1.06 s), chosen to cover Airstriker's relevant temporal structure, not to
push context length.

#### B.2.8 Inductive bias and why transformers are data-hungry

This subsection is the conceptual payoff — it explains v12's central empirical
result (the ~2× sample cost, B.7) from first principles.

An **inductive bias** is a built-in assumption that constrains *which* functions a
model can easily represent, before it sees any data. Strong, *correct* biases let
a model generalize from little data (the hypothesis space is small and well-aimed);
weak biases give a larger hypothesis space — a higher ceiling — but require more
data to pin down the right function. This is the classic bias–variance lever cast
in architectural terms.

Contrast the three architectures this project has used by their built-in biases:

| Architecture | Built-in biases | Consequence |
|---|---|---|
| **CNN** (v9 IMPALA) | **locality** (a kernel sees only a neighborhood), **translation equivariance + weight sharing** (the same filter slides everywhere), spatial **hierarchy** | strong, well-matched priors for pixels → learns from *less* data; ceiling capped by the priors |
| **LSTM** (v10/v11) | **sequential recency / Markovian recurrence** (state summarizes the past, updated step-by-step) | a temporal prior, but couples memory to a state that must be built up — hence the cold-start defect |
| **Transformer** (v12) | **almost none** — self-attention is permutation-equivariant and *fully connected* across tokens from layer 1; the *only* injected structure is the positional embedding | maximally flexible (any token can read any token) → higher ceiling, but must **learn** the relational structure that the CNN gets for free → needs *more* data |

The CNN, in particular, gets temporal structure *for free*: v9's first conv layer
mixes the 4 stacked frames by position, hard-wiring "compare adjacent frames"
(which *is* velocity) into the architecture. v12's attention layers start
**permutation-equivariant** — they don't even know frame order until the
positional embedding is learned — and must *discover* from the reward signal which
frames to compare and how. That discovery is exactly what costs the extra samples.

This is the **Vision Transformer story** (B.2.0) repeating in miniature: ViT
underperforms CNNs on small datasets and overtakes them only at large scale,
precisely because it trades a CNN's hard-wired spatial priors for flexibility that
must be paid for in data. v12 reproduces it: it needed ~5.3M steps to reach
(and just exceed) what v9's IMPALA reached at ~2.7M. The **+6.2% ceiling gain is
the flexibility payoff**; the **2× sample cost is the weak-bias tax**. Both are
the textbook prediction — v12 is a clean confirmation, not a surprise.

The practical corollary (already in B.7): when integrating a new game, start with
the high-bias, sample-efficient CNN; reach for attention only when you have the
step budget to amortize its data appetite and need the extra ceiling.

#### B.2.9 Training dynamics — LR sensitivity, warmup, and the smoke plateau

Transformers are notoriously **sensitive to the learning-rate schedule**. The
original Transformer required a hand-shaped **warmup**: ramp the LR up linearly for
the first few thousand steps, then decay it (`∝ 1/√step`). The reason traces to
post-LN's gradient scaling, which can be large and ill-conditioned early in
training. **Pre-LN** (B.2.6, what v12 uses) substantially fixes this — it keeps the
residual path un-normalized and makes early gradients well-behaved, which is why
pre-LN transformers can often train *without* warmup. But residual sensitivity
remains, and attention has its own slow-start dynamic:

- At initialization the `W_Q, W_K` projections are small/random, so all scores
  `q_i·k_j ≈ 0`, so **every attention row is ≈ uniform** — each token just averages
  all frames. In that regime attention carries almost no signal; the heads have not
  yet *differentiated* into the distinct relations of B.2.4. The network must first
  push the query/key projections away from this uniform fixed point before the
  temporal read becomes useful.

This is visible in v12's smoke (B.7's preamble): at `lr=1e-4` the return
**plateaued from ~200K–300K steps, then broke out sharply** — the signature of
heads sitting near-uniform, then differentiating once the projections grew enough
to make scores informative. Dropping to **`lr=5e-5`** for the full run gave a
smoother early trajectory (less thrash while the heads specialize) at the cost of
slightly slower wall-clock progress — a deliberate trade given the 6M budget. The
plateau was *not* a sign the architecture couldn't learn; it was the
representation-learning phase that B.2.8 predicts a low-bias model must pay up
front. The late-run collapse-and-recover at 5.9M (B.7) is a separate phenomenon —
the entropy schedule reaching its floor — not a transformer-specific dynamic.

### B.3 The v12 architecture — `TemporalAttentionExtractor`

[`models/attention.py`](../src/retro_rl/models/attention.py). The extractor is a
drop-in for `RetroCNN`/`ImpalaCNN` — same `(observation_space, features_dim)`
constructor — so it plugs into the existing registry and PPO factory unchanged
(§2.4, §2.5).

```
obs (B, K=8, 84, 84) uint8        ← the channel axis IS the frame sequence
   │                                  (grayscale ⇒ 1 channel/frame ⇒ C = K)
   │  /255, reshape → (B·K, 1, 84, 84)        tokenize: one frame per token
   ▼
┌─────────────────────────────────────────────┐
│  Per-frame CNN  (Nature-CNN, SHARED weights) │
│   Conv(1→32, k8 s4) → ReLU   → (32, 20, 20)  │
│   Conv(32→64,k4 s2) → ReLU   → (64,  9,  9)  │
│   Conv(64→64,k3 s1) → ReLU   → (64,  7,  7)  │
│   Flatten                     → (3136,)      │
│   Linear(3136 → 256) → ReLU   → (256,)       │
└─────────────────────────────────────────────┘
   │  reshape → (B, K, 256)        K token embeddings
   │  + learned positional embedding  (1, K, 256)
   ▼
┌─────────────────────────────────────────────┐
│  TransformerEncoder  (2 layers, pre-LN)      │
│    per layer:  MHA(4 heads, d=256)           │
│                + FFN(256→512→256, GELU)      │
│                2× LayerNorm, 2× residual     │
│  full (non-causal) self-attention over K=8   │
└─────────────────────────────────────────────┘
   │  z ∈ (B, K, 256)
   ▼  take the LAST token:  z[:, -1, :]
features (B, 256)   →   Actor MLP [64,64] Tanh → 9 logits
                    →   Critic MLP [64,64] Tanh → V(s)
```

Design decisions, each with its rationale:

- **A token = a frame.** The channel axis of the stacked observation `(K,84,84)`
  *is* the sequence (grayscale ⇒ one channel per frame, §1.5 [7–8]). We reshape
  `(B,K,84,84) → (B·K,1,84,84)`, run the CNN once on that flattened batch, and
  reshape back to `(B,K,256)`. The CNN weights are **shared** across the `K`
  frames (it is the same function applied to each), exactly like a token
  embedding shared across positions in NLP.

- **Per-frame encoder = Nature-CNN, not IMPALA.** The temporal modeling now lives
  in the attention layers, so the per-frame encoder need not be deep. Using the
  lighter Nature-CNN keeps the cost of `K=8` forward passes in the ballpark of
  v9's single IMPALA pass on a 4-channel input. (`features_dim=256` so the conv
  flatten dim is the same 3136 as §2.2.)

- **`d_model = features_dim = 256`.** Attention runs in the extractor's output
  width; there is no separate token-dimension knob. `n_heads=4` must divide it
  (256/4 = 64 per head). `n_heads`, `n_layers=2`, and `d_ff=512` are **baked into
  the extractor**, not exposed as config — they are used in exactly one place;
  promote to YAML only if a smoke shows they need tuning.

- **Last-token readout.** We return `z[:, -1, :]` — the *most recent* frame's
  representation, after it has attended over all `K` frames. Why the last token,
  and why full (non-causal) attention is fine: we only ever read position `K−1`,
  and under a causal mask position `K−1` already attends to all positions `≤K−1`
  (i.e. everything). So causal and full attention give the **same** output for
  the last token — we use full attention because it is simpler, and read the last
  token because it represents "now, informed by the recent past," which is
  exactly the feature the policy needs.

- **Cold-start: eliminated.** At episode reset the `FrameStack` wrapper fills all
  `K` slots with the first frame, so the window is always populated. There is no
  zero-state — the worst case is a *mildly degenerate* window (all 8 slots equal)
  for the first step, which resolves to genuine history within `K` steps. Contrast
  the LSTM's `h,c=0`, which is an *out-of-distribution* state the network must
  actively recover from.

**Parameter budget (approximate):**

| Component | Params | Note |
|---|---|---|
| Per-frame CNN (shared) | ~0.88 M | the `3136→256` FC dominates (§2.2 pattern) |
| Positional embedding | ~2 K | `K×d = 8×256` |
| Transformer (2 layers) | ~1.05 M | per layer ≈ 0.53 M (MHA ≈ 0.26 M + FFN ≈ 0.26 M) |
| Actor + critic heads | ~42 K | SB3 default `[64,64]` MLPs |
| **Total** | **~1.97 M** | vs v9 1.13 M, v10 1.66 M |

The transformer is where the added parameters go — but, per B.2.7, not where the
added *compute* goes (that is the `K`-fold CNN).

### B.4 Integration with standard PPO — the simplicity win

The defining practical advantage of v12 over v10/v11: it requires **zero changes
to the training loop**. Everything in A.4–A.5 — hidden-state threading, ordered
sequence minibatches, TBPTT, the per-step `(h,c)` buffer fields — **does not
exist** for v12. The reason is structural: a transformer-over-the-window is a
*stateless function of one observation*. It reads the `K` frames that the
`FrameStack` wrapper already packs into a single observation tensor, so from
PPO's point of view it is just another CNN feature extractor.

Concretely, compared to the standard-PPO pipeline of Milestone 3, v12 changes
**nothing**:

- **Rollout buffer:** identical to §3.1 (no `(h,c)` fields). Observations are the
  same `(K,84,84)` uint8 tensors, just with `K=8` instead of 4.
- **Minibatching:** the random shuffle of 1024 transitions is **restored** (A.4
  had to give this up for ordered sequences). Each transition is independent
  again.
- **Forward pass:** independent per observation — the attention happens *inside*
  the extractor over that observation's own 8 frames, never across buffer rows.
- **Trainer / callbacks / checkpoint / eval / backend:** untouched. The
  `algorithm="ppo"` branch in [`trainer.py`](../src/retro_rl/training/trainer.py)
  already wires `build_ppo`; `predict()` returns `(action, None)` (no recurrent
  state), so eval and the FastAPI backend are unchanged.

The entire wiring is **one registry line** (§2.4):

```python
FEATURE_EXTRACTORS = {
    "nature_cnn": RetroCNN,
    "impala":     ImpalaCNN,
    "temporal_attn": TemporalAttentionExtractor,   # ← v12
}
```

plus a config overlay (`ppo_v12.yaml` extends `ppo_v9.yaml`, swapping
`features_extractor: impala → temporal_attn` and `frame_stack: 4 → 8`; everything
else — VecNormalize, LR=1e-4, ent-coef schedule, 4M steps — is inherited from the
best run for a clean head-to-head).

### B.5 Shape / data flow — one observation through the extractor

```
═══════════════════════════════════════════════════════════════════════════════
  v12 feature extractor — forward pass for a batch of B observations
═══════════════════════════════════════════════════════════════════════════════

  input   obs   (B, 8, 84, 84)  uint8        ← 8-frame stack, channel axis = time
            │  x = obs.float()/255
            │  reshape (B,8,84,84) → (B·8, 1, 84, 84)     each frame = one token
            ▼
  ┌────────────────────────────┐
  │  Per-frame CNN (shared)     │  batch = B·8 single-channel frames
  │   (B·8, 1, 84, 84)          │
  │     ↓ Conv(1→32,k8,s4)+ReLU │
  │   (B·8, 32, 20, 20)         │
  │     ↓ Conv(32→64,k4,s2)+ReLU│
  │   (B·8, 64,  9,  9)         │
  │     ↓ Conv(64→64,k3,s1)+ReLU│
  │   (B·8, 64,  7,  7)         │
  │     ↓ Flatten               │
  │   (B·8, 3136)               │
  │     ↓ Linear(3136→256)+ReLU │
  │   (B·8, 256)                │  ← one 256-d embedding per frame
  └─────────────┬───────────────┘
                │  reshape → (B, 8, 256)            sequence of 8 tokens
                │  + pos_emb (1, 8, 256)            inject frame order
                ▼
  ┌────────────────────────────────────────────────────────┐
  │  Transformer encoder, layer ℓ = 1,2   (pre-LN)          │
  │                                                          │
  │   u = LayerNorm(x)                  (B, 8, 256)          │
  │   Q,K,V = u W_Q, u W_K, u W_V       (B, 8, 64) ×4 heads  │
  │   A = softmax(Q Kᵀ / √64)           (B, 4, 8, 8)         │ ← per-head 8×8
  │   head = A · V                      (B, 8, 64) ×4        │   attention map
  │   x = x + Concat(heads) W_O         (B, 8, 256)          │ ← residual
  │   x = x + FFN(LayerNorm(x))         (B, 8, 256)          │ ← residual, GELU
  └─────────────┬────────────────────────────────────────────┘
                │  z = x   (B, 8, 256)
                ▼  take last token   z[:, -1, :]
            (B, 256)   ← "current frame, informed by the prior 7"
                │
          ┌─────┴─────┐
          ▼           ▼
       Actor        Critic        (SB3-default [64,64] Tanh MLP heads, §2.4)
      9 logits      V(s)

  Note: the 8×8 attention matrix A is per-head and per-observation — fully
  parallel across the B batch and the 4 heads. No step-by-step unroll, no
  state carried between observations (contrast A.5's LSTM).
```

### B.6 Frame stack vs LSTM vs attention — what changes

| Aspect | Frame stack (v9) | LSTM (v10/v11) | Attention (v12) |
|---|---|---|---|
| **Memory** | fixed conv over `k` frames | learned recurrent state | learned content-read over `K` frames |
| **Reach** | exactly `k` (=4) | unbounded (in principle) | exactly `K` (=8), explicit |
| **Cross-frame mixing** | first conv layer, by position | recurrence, sequential | self-attention, by content |
| **Cold start** | none | **severe** (`h,c=0`) | none (window pre-filled) |
| **Algorithm** | standard PPO | RecurrentPPO (A.4) | **standard PPO** |
| **Rollout buffer** | obs only | obs + per-step `(h,c)` | obs only |
| **Minibatches** | random shuffle | ordered sequences | random shuffle |
| **Forward pass** | independent per obs | sequential, stateful | independent per obs |
| **Parallelism** | full | limited (sequential unroll) | full (parallel matmuls) |
| **Added params** | 0 | ~0.5 M (LSTM) | ~0.8 M (transformer) |
| **Added compute** | ~0 | LSTM unrolls | `K`-fold CNN passes |
| **Eval-time risk** | none | bimodal (cold-start dropouts) | none by construction |

**The thesis of v12 in one line:** keep the LSTM's *learned, adaptive* temporal
read, but realize it over an *explicit, always-present* window instead of a
recurrent state — buying back the stateless simplicity (and cold-start immunity)
of frame stacking while extending the window and making the read content-based.

### B.7 v12 — Final result

v12 ran for 6M steps at `lr=5e-5` (re-tuned from the original `1e-4` after two
500K smokes revealed a transformer optimization cold-start: attention heads took
~300K steps to stabilize, producing a plateau that broke out sharply thereafter).

| Checkpoint | eval_return | eval_length |
|---|---|---|
| 3.2M | 6,298 | 1,484 |
| 3.3M | 7,458 | 1,529 |
| 5.1M | 6,868 | **4,500** (max steps — survived full episode) |
| 5.2M | 7,317 | 1,470 |
| **5.3M** | **7,609** | 1,548 — **best** |
| 5.6M | 7,097 | 1,451 |
| 5.8M | 6,636 | 1,421 |
| 5.9M | 2,212 | 463 — **policy collapse** |
| 6.0M | 7,517 | 1,479 — recovery |

**Best: 7,609 @ 5.3M — new numerical best, +6.2% (+441 points) over v9's 7,168 @ 2.7M.**

The attention hypothesis is validated: a fixed-window transformer over the frame
sequence provides real benefit over pure IMPALA feature learning, **once given
sufficient steps**. The key trade-offs vs v9:

- **Sample cost**: v12 needs ~5.3M steps to peak vs v9's 2.7M — almost exactly
  2×. Wall-clock: ~65-70h vs ~30-35h.
- **Eval variance**: v12's late-run curve is noisier. The 5.9M crash to 2,212
  (followed by immediate recovery to 7,517 at 6M) is likely the `ent_coef`
  schedule reaching its floor (`0.001` at 6M), reducing the entropy regulariser
  and exposing occasional policy-mode collapse. v9's late-run was also noisy but
  had no comparable single-interval crash.
- **Compute efficiency**: v9 reaches within 6% of v12's peak at half the steps.
  For a new game integration, start with v9's architecture (IMPALA + VecNormalize
  + `lr=1e-4`); extend to v12 only if you have the step budget and need the
  marginal gain.

The v10→v11→v12 sequence cleanly isolates the recurrence vs attention
trade-off: LSTM cold-start is structural and cannot be overcome with more steps
(v11, 6M, 5,325); attention eliminates the cold-start and, at the same budget,
exceeds v9. **Experiment closed.** The best checkpoint is
`outputs/checkpoints/ppo_airstriker_v12/best.zip`.
