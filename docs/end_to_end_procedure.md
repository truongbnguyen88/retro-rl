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
algorithm. "On-policy" means: collect data with the current policy, update
the policy, throw away the data, repeat. Each iteration:

```
1. Rollout collection  ─► 1024 transitions (8 envs × 128 steps each)
2. Advantage estimation (GAE)
3. Optimization: 4 epochs over 1024 transitions in minibatches of 256
4. Discard data, goto 1
```

Why on-policy? On-policy methods are generally more stable for actor-critic
policies because the gradient estimates use data from the same distribution
the policy currently defines. Off-policy methods (DQN, SAC) reuse old data
but require extra machinery (replay buffers, target networks) to stay stable.

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
| v9 | IMPALA ResNet + action_repeat=8 + VecNormalize + LR=1e-4 | **7168 @ 2.7M** |
| v10 | RecurrentPPO + LSTM(256) + frame_stack=1 (in progress) | — |

---

*For the detailed IMPALA ResNet architecture and PPO math in isolation, see
[docs/v9_procedure_pipeline.md](v9_procedure_pipeline.md).*
