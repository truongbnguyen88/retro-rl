# v9 Procedure & Pipeline — IMPALA ResNet + PPO

A from-scratch walkthrough of how the v9 agent is built and trained: what goes
in, what comes out, the IMPALA ResNet backbone in detail, and the PPO learning
loop. All shapes and parameter counts are taken from the actual v9 network
(`Discrete(9)` actions, `(4, 84, 84)` observations, 256-d features,
**1,131,098** total params).

---

## 0. The one distinction that trips everyone up

**"IMPALA" names two different things** in Espeholt et al. 2018:

1. a distributed actor–learner **algorithm** (with V-trace off-policy correction);
2. a deep residual conv **network** (the vision backbone).

In v9 we use **only the network** and pair it with **PPO** as the learning
algorithm. There is **no V-trace and no distributed actor–learner** here.

> **IMPALA ResNet + PPO = IMPALA's eyes, PPO's brain.**
> The network maps pixels → features; PPO maps experience → weight updates.
> They are orthogonal — SB3 treats the feature extractor as a swappable module,
> so the same PPO machinery ran the Nature-CNN in v5–v8 and the IMPALA backbone
> in v9 with a one-line config change.

---

## 1. Inputs — what the policy actually sees

The raw Genesis frame (224×320×3 RGB) is transformed by the env wrapper stack
before it reaches the network:

| Stage | Effect |
|---|---|
| Grayscale + resize | 224×320×3 → 84×84×1 |
| Frame stack ×4 | 4 consecutive frames → **(4, 84, 84)** — encodes *motion* (bullet/enemy velocity & direction) from otherwise-static images |
| `action_repeat = 8` | one policy decision is held for 8 emulator frames; reward summed across them |
| `VecTransposeImage` (SB3) | → channels-first **(B, 4, 84, 84)**, `uint8` |

**Input = a `(4, 84, 84)` uint8 tensor.** It is cast to float and divided by
255 *inside* the feature extractor's `forward` (keeps the rollout buffer in
uint8 → ~4× memory saving).

**I/O subtlety — the agent does not fire.** The 9 actions are *movement-only*
combos (no-op + 8 directions). `AutoFireWrapper` taps the B button at the raw
emulator-frame level (period 12, ~5 Hz) independently of the policy. The agent
learns *where to fly*, not *when to shoot*.

---

## 2. Forward pass — input → action

```
(4,84,84) uint8
   │  /255
   ▼
IMPALA CNN backbone  ──────────────────────────────────┐
   3 conv stacks, channels 16 → 32 → 32                 │
   each stack:  Conv3x3 → MaxPool(/2) → ResBlock → ResBlock
   spatial:  84 → 42 → 21 → 11                          │
   → 32 × 11 × 11 = 3872  ──► Flatten ──► Linear(3872,256) ──► ReLU
   ▼
256-d feature vector  (shared representation)
   │
   ├──► policy MLP: 256→64→64 (Tanh) ──► action_net Linear(64→9) ──► 9 logits
   │                                         → softmax → Categorical π(a|s)
   │                                         → sample (train) / argmax (eval)
   │
   └──► value  MLP: 256→64→64 (Tanh) ──► value_net  Linear(64→1) ──► V(s) scalar
```

**Outputs, two heads off the shared 256-d features:**

- **Actor** → 9 logits → categorical distribution over actions. This *is* the
  policy. Sampled during training, argmax at deterministic eval.
- **Critic** → scalar `V(s)`, the expected discounted return. Used only for
  training (advantage estimation); discarded at deploy.

SB3 keeps the conv backbone shared but gives actor and critic **separate**
256→64→64 MLP heads.

---

## 3. The IMPALA ResNet backbone in detail

The v9 perception upgrade. Bottom-up: residual block → conv stack → full
backbone. The body has **15 `Conv2d` layers + 1 `Linear`**.

### 3.1 The residual block (used ×2 per stack, 6 total)

```
        x ─────────────────────────────┐  (identity skip — pure, no activation)
        │                               │
      ReLU                              │
        │                               │
   Conv 3×3, s1, p1  (C→C)              │
        │                               │
      ReLU                              │
        │                               │
   Conv 3×3, s1, p1  (C→C)              │
        │                               │
        └────────────►  ( + ) ◄─────────┘
                          │
                       out = F(x) + x      (channels & H×W unchanged)
```

Three structural choices and why they matter:

**(a) Pre-activation ordering** (`ReLU → Conv`, not `Conv → ReLU`; He et al.
2016 "identity mappings"). The block output is `F(x) + x` with a **pure
identity skip** — no ReLU, no scaling on it. Gradient consequence:

```
∂L/∂x = ∂L/∂(out) · (∂F/∂x + 1)
```

That **`+1`** is a gradient highway: even if `∂F/∂x` is tiny (saturated convs),
the loss still propagates undiminished to earlier layers. This is what makes a
15-conv stack trainable without vanishing gradients. A post-activation
`relu(F(x)+x)` would gate the skip and lose the clean identity.

**(b) Channel-preserving → the skip is a plain add.** Both convs are
`channels → channels`, so `F(x)` and `x` share shape — no 1×1 projection on the
skip (hence the block takes a single `channels` arg). The channel-changing conv
lives in the stack, outside the residual blocks.

**(c) No BatchNorm.** Canonical ResNet interleaves BN; the IMPALA RL variant
drops it. On-policy RL minibatches are small and the data distribution is
correlated and non-stationary (the policy keeps changing which states it
visits), so BN's running statistics are unreliable and often hurt. With `F≈0`
at init the block is near-transparent — a benign starting point.

### 3.2 The conv sequence (one stack)

```
Conv3x3(in→out, stride 1, pad 1)   # changes channel count; preserves H,W
MaxPool(k=3, stride 2, pad 1)      # halves H,W
ResidualBlock(out)                 # refine
ResidualBlock(out)                 # refine
```

One job per component: the **stack-conv** changes channel width, the
**maxpool** halves spatial resolution (bounds compute as channels grow), the
**two residual blocks** add nonlinear refinement at that resolution. The body
uses **no stride>1 in any conv** — all downsampling is via maxpool, which keeps
finer spatial detail than the Nature-CNN's strided convs (stride 4 then 2).

### 3.3 Spatial + channel flow (exact)

MaxPool output: `floor((H + 2·1 − 3)/2) + 1 = floor((H−1)/2) + 1`.

| After | Channels | H×W | Tensor |
|---|---|---|---|
| input | 4 | 84×84 | 4·84·84 |
| stack 1 | 16 | **42×42** | 16·42·42 |
| stack 2 | 32 | **21×21** | 32·21·21 |
| stack 3 | 32 | **11×11** | 32·11·11 = **3872** |
| flatten → linear | — | — | **256** |

### 3.4 Parameter budget (where the weights live)

Conv params per layer = `(3·3·in + 1)·out`:

| Block | params |
|---|---|
| Stack 1 (4→16): conv 592 + 2 res ×4640 | ~9.9K |
| Stack 2 (16→32): conv 4640 + 2 res ×18.5K | ~41.6K |
| Stack 3 (32→32): conv 9248 + 2 res ×18.5K | ~46.2K |
| **Conv backbone total** | **~97.7K** |
| **Linear(3872→256)** | **~991K** |
| Heads (2× 256→64→64 + 9-logit + value) | ~42K |
| **Model total** | **1,131,098** |

The striking part: the **convolutional body is only ~98K params**, but the
**flatten→Linear projection is ~991K (~88% of the network)**. The "deep ResNet"
carries little weight; the depth buys *representation quality* (nonlinearity,
gradient flow), not parameter count. To shrink the model, the lever is the FC
projection (e.g. global-average-pool the 11×11 instead of flattening), not the
conv stacks.

### 3.5 Receptive field — why depth matters here

Tracking RF `r` and jump `j` forward (`conv3x3 s1: r += 2j`;
`maxpool s2: r += 2j, j ×= 2`):

| After | RF (pixels) | jump |
|---|---|---|
| stack 1 | 21 | 2 |
| stack 2 | 61 | 4 |
| stack 3 | **141** | 8 |

The final theoretical RF (**141 px**) **exceeds the 84-px frame** — each of the
11×11 output cells can integrate information from *anywhere on screen*. That is
the payoff of residual depth: the policy can relate the ship's position to
threats anywhere in the playfield (a bullet entering top-right while dodging
bottom-left). The shallower Nature-CNN has a smaller effective RF and plateaus
exactly in this "many small fast objects spread across the frame" regime —
which motivated the swap in v9.

---

## 4. Full architecture sketch

### 4.1 Backbone (dataflow)

```
INPUT  observation  (4, 84, 84)  uint8
   │
   │  x = x.float() / 255.0            ← normalize inside forward()
   ▼
┌──────────────────────── ConvSequence 1  (4 → 16) ────────────────────────┐
│   Conv 3×3, s1, p1   (4→16)          (16, 84, 84)                         │
│   MaxPool 3×3, s2, p1                (16, 42, 42)   ← spatial /2          │
│   ResidualBlock(16)                  (16, 42, 42)                         │
│   ResidualBlock(16)                  (16, 42, 42)                         │
└──────────────────────────────────────────────────────────────────────────┘
   ▼                                    (16, 42, 42)
┌──────────────────────── ConvSequence 2  (16 → 32) ───────────────────────┐
│   Conv 3×3, s1, p1   (16→32)         (32, 42, 42)                         │
│   MaxPool 3×3, s2, p1                (32, 21, 21)   ← spatial /2          │
│   ResidualBlock(32)                  (32, 21, 21)                         │
│   ResidualBlock(32)                  (32, 21, 21)                         │
└──────────────────────────────────────────────────────────────────────────┘
   ▼                                    (32, 21, 21)
┌──────────────────────── ConvSequence 3  (32 → 32) ───────────────────────┐
│   Conv 3×3, s1, p1   (32→32)         (32, 21, 21)                         │
│   MaxPool 3×3, s2, p1                (32, 11, 11)   ← spatial /2          │
│   ResidualBlock(32)                  (32, 11, 11)                         │
│   ResidualBlock(32)                  (32, 11, 11)                         │
└──────────────────────────────────────────────────────────────────────────┘
   ▼                                    (32, 11, 11)
  ReLU                                  (32, 11, 11)
   │
  Flatten                               (3872,)        ← 32·11·11 = 3872
   │
  Linear 3872 → 256                     (256,)         ← ~991K params (88% of net)
   │
  ReLU
   ▼
FEATURES  (256,)  ─────────────  shared representation
```

### 4.2 Actor–critic heads (PPO)

```
                         FEATURES (256,)
                          │            │
              ┌───────────┘            └───────────┐
              ▼                                     ▼
        POLICY head                            VALUE head
   Linear 256→64, Tanh                    Linear 256→64, Tanh
   Linear  64→64, Tanh                    Linear  64→64, Tanh
   Linear  64→9   (action_net)            Linear  64→1   (value_net)
              │                                     │
        9 logits → softmax                       V(s) scalar
        → Categorical π(a|s)                    (critic, train-only)
              │
        sample (train) / argmax (eval)
              ▼
        action a ∈ {0..8}  (movement combo)
```

### 4.3 Layer / shape / param table

```
 #   Component                in→out ch    spatial      params
 ──  ───────────────────────  ──────────   ─────────    ───────
     INPUT                     4            84×84        —
 1   Conv (stack1)             4→16         84×84        592
     MaxPool                   16           84→42        —
 2   Conv  res1.conv1          16→16        42×42        2,320
 3   Conv  res1.conv2          16→16        42×42        2,320
 4   Conv  res2.conv1          16→16        42×42        2,320
 5   Conv  res2.conv2          16→16        42×42        2,320
 6   Conv (stack2)             16→32        42×42        4,640
     MaxPool                   32           42→21        —
 7   Conv  res1.conv1          32→32        21×21        9,248
 8   Conv  res1.conv2          32→32        21×21        9,248
 9   Conv  res2.conv1          32→32        21×21        9,248
10   Conv  res2.conv2          32→32        21×21        9,248
11   Conv (stack3)             32→32        21×21        9,248
     MaxPool                   32           21→11        —
12   Conv  res1.conv1          32→32        11×11        9,248
13   Conv  res1.conv2          32→32        11×11        9,248
14   Conv  res2.conv1          32→32        11×11        9,248
15   Conv  res2.conv2          32→32        11×11        9,248
     ReLU + Flatten            —            →3872        —
 ─   Linear                    3872→256     —            991,488
 ─   ───────────────────────────────────────────────────────────
     Conv backbone                                       ~97.7K
     + FC projection                                     ~991K
     + actor/critic heads                                ~42K
     TOTAL                                               1,131,098
```

Visual takeaways: **spatial resolution halves at each stack** (84→42→21→11) via
the maxpools while **channels stay narrow** (16→32→32); the **identity skips**
are the gradient highways that keep all 15 convs trainable; the **single Linear
projection holds ~88% of the weights** despite the "deep ResNet" label.

---

## 5. How it's trained — the PPO loop

PPO is **on-policy**: collect a batch with the *current* weights, update, throw
the batch away, repeat.

**1. Rollout collection.** 8 parallel envs (`n_envs=8`) each run `n_steps=128`
→ **1024 transitions** per iteration. Each transition stores
`(obs, action, reward, V(s), log π(a|s), done)`. The `V(s)` and `log π` recorded
here are the "old policy" reference.

**2. Reward normalization (the v9 fix).** `VecNormalize(norm_reward=True)`
divides rewards by a running std so returns sit at ~unit variance, making the
value regression well-conditioned (EV 0.47→0.98, value_loss ~0.02 vs v8's
~hundreds). Eval runs on a bare env reporting raw returns, so the metric stays
comparable.

**3. Advantage estimation (GAE).** Using recorded values and `γ=0.99, λ=0.95`:

- `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`  (TD error)
- `Â_t = Σ (γλ)^k δ_{t+k}`  (exponentially-weighted; bootstrapped at truncation)
- `Â` answers "was this action better or worse than the critic expected?"

**4. Optimization.** For `n_epochs=4`, shuffle the 1024 transitions into
minibatches of 256 and minimize:

```
L = L_policy  +  vf_coef · L_value  −  ent_coef · H(π)
```

- **Clipped policy loss:** with ratio `r = π_new(a|s) / π_old(a|s)`,
  `L_policy = −E[ min( r·Â , clip(r, 1−ε, 1+ε)·Â ) ]`, `ε = clip_range = 0.1`.
  The clip is the trust region — it forbids moving the policy too far from the
  data-collecting policy, which is what makes PPO stable.
- **Value loss:** `vf_coef = 0.5 × MSE(V_pred, returns)` — trains the critic.
- **Entropy bonus:** `−ent_coef · H(π)` rewards a less-peaked distribution →
  exploration. v9 anneals `ent_coef` 0.02 → 0.001 over 4M steps (this schedule
  annealed too fast and froze the policy at ~3.1M — a known v10 lever).

**5. Backprop is end-to-end.** Gradients from the combined loss flow through the
heads **and the shared IMPALA conv backbone** — the visual features are
*learned*, shaped by what helps predict value and improve the policy.
`learning_rate = 1e-4`, `max_grad_norm = 0.5`.

**6. Repeat** with updated weights on fresh data.

---

## 6. PPO internals (deep dive)

Section 5 is the operational summary. This section unpacks three things people
most often ask about: what `obs` actually is, *why* the PPO update is shaped the
way it is, and how GAE estimates advantages.

### 6.1 What `obs` is — observation vs. state

`obs` is the **observation**: the actual tensor fed into the network — in v9 the
**`(4, 84, 84)` uint8 stack** (4 grayscale 84×84 frames), stored in the rollout
buffer as uint8.

The terminology matters. Airstriker is a **POMDP** (partially observed): the
true Markov state lives in the emulator's RAM (positions, velocities, spawn
timers), but the agent never sees it — it only sees **pixels**. A single frame
isn't Markov: from one image you can't tell whether a bullet is moving up or
down. **Frame-stacking 4** is the fix — the stack encodes motion (Δposition
across frames), making `obs` a *good-enough approximation* of the Markov state.

So in the transition tuple `(obs, action, reward, V(s), log π(a|s), done)`,
`obs_t` is "the 4-frame image the network looked at when it chose `action_t`."
Everything else is computed from or alongside it.

> Strictly it is `o_t` (observation), not `s_t` (state). The literature writes
> `V(s)` loosely; here `V` is really `V(o_t)` — the value of the *observation*,
> since that is all the network has.

### 6.2 How the PPO update is built

**Foundation — policy gradients.** We maximize `J(θ) = E_π[ Σ_t γ^t r_t ]`. The
policy gradient theorem gives

```
∇_θ J(θ) = E[ ∇_θ log π_θ(a_t | s_t) · Ψ_t ]
```

where `Ψ_t` measures "how good was `a_t`." The simplest choice (REINFORCE) sets
`Ψ_t = R_t`, the full return. Two fatal problems:

1. **High variance** — `R_t` depends on the entire random future → noisy gradient.
2. **Destructive steps** — one large step can collapse the policy, and the data
   was collected under the *old* policy, so it is no longer valid for the new one.

PPO fixes (1) with **advantages + GAE** (§6.3) and (2) with the **clipped
surrogate objective**.

**The importance-sampling ratio.** PPO reuses each batch for several gradient
epochs (`n_epochs=4`). After the first update, the optimized policy `π_θ` has
drifted from the data-collecting policy `π_θ_old`, so it weights by the ratio

```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

`r=1` → unchanged; `r>1` → action made more likely; `r<1` → less likely. This is
why each transition stores `log π_θ_old(a_t|s_t)` — the fixed denominator
(`r = exp(log π_θ − log π_θ_old)`).

**The clipped surrogate.** A naive `E[ r_t · Â_t ]` would let one update push
`r_t` arbitrarily far when `Â_t` is large. PPO clips it:

```
L^CLIP(θ) = E[ min( r_t · Â_t ,  clip(r_t, 1−ε, 1+ε) · Â_t ) ]      ε = clip_range = 0.1
```

The `min` makes it **pessimistic**:

- **`Â_t > 0` (good action):** raise `r_t`, but the clip caps the benefit at
  `r = 1+ε` (1.1 in v9) → no incentive to overshoot.
- **`Â_t < 0` (bad action):** lower `r_t`, but the clip floors the benefit at
  `r = 1−ε` (0.9) → no further gain from pushing past it.

Net effect: a **first-order, cheap trust region** — it approximates TRPO's hard
KL constraint with a clip, keeping updates small enough that the on-policy data
stays approximately valid across the 4 epochs.

**The full loss.**

```
L(θ) = L^CLIP(θ)  −  c1 · L^VF(θ)  +  c2 · H[π_θ](s)
```

- `L^VF = MSE(V_θ(s), V_target)` — value head regresses toward the GAE returns,
  `c1 = vf_coef = 0.5`.
- `H[π_θ]` — entropy bonus, keeps the action distribution from collapsing too
  early, `c2 = ent_coef` (annealed 0.02→0.001). SB3 minimizes `−L`, so signs
  flip in code.

**The `approx_kl` diagnostic** in the logs is `E[log π_old − log π_new]`, an
estimate of how far the policy moved this iteration. When it → 0 (v9 at ~3.1M)
the policy has stopped updating — usually because advantages shrank and/or the
entropy term annealed away. That is the late-run freeze noted in §5.

### 6.3 Generalized Advantage Estimation (GAE)

**The advantage.** `A(s, a) = Q(s, a) − V(s)` — "how much better is action `a`
than the policy's *average* behavior at `s`?" Using `A` instead of the raw
return `R` in the policy gradient is **baseline subtraction**: it doesn't change
the gradient's expectation (unbiased) but **dramatically reduces variance** —
actions are reinforced relative to a learned reference `V(s)`, not by their
noisy absolute return.

**The estimation problem — bias vs. variance.** We estimate `A` from sampled
rewards and the imperfect critic `V`. With the one-step TD error

```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)
```

- **one-step (`Â = δ_t`):** low variance, but **biased** (leans entirely on `V`);
- **Monte Carlo (`Â = Σ γ^l r_{t+l} − V(s_t)`):** unbiased, but **high variance**;
- **n-step** estimators interpolate between them.

**What GAE does.** It is an exponentially-weighted average of *all* n-step
estimators, decayed by `λ`:

```
Â_t^GAE = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}
```

computed by a backward recursion over the rollout:

```
Â_t = δ_t + (γλ) · Â_{t+1}        (Â = 0 past a terminal state)
```

`λ` is the bias–variance knob:

| `λ` | Â reduces to | character |
|---|---|---|
| 0 | `δ_t` (one-step TD) | low variance, high bias |
| 1 | `Σ γ^l r − V(s)` (Monte Carlo) | high variance, low bias |
| **0.95** (v9) | weighted blend | mostly-unbiased, variance-controlled |

Terminal handling matters: at a true death `δ_t = r_t − V(s_t)` (no bootstrap);
at a time-limit **truncation** it bootstraps with `V(s_{t+1})`. Getting this
wrong silently biases learning.

**How it helps training:**

1. **Lower-variance, stable gradients** → faster, more reliable improvement.
2. **Sharper credit assignment** → advantages say *which actions beat
   expectation*, not just *which trajectories scored high*.
3. **Faster credit propagation** → bootstrapping through `V` spreads reward
   backward without waiting for full episode returns.
4. **It defines the value target:** `R_t = Â_t + V(s_t)` is what the critic
   regresses toward (`L^VF`) — GAE and the value head are coupled.

**Why this is the deep reason VecNormalize unblocked v9.** Every `δ_t` contains
`V(s)` and `V(s')`, so GAE quality depends entirely on the critic. In v8, raw
returns made the value regression ill-conditioned (EV ~0.22, value_loss in the
hundreds) → noisy `V` → noisy `δ_t` → **noisy/biased advantages** → slow,
unstable policy learning. v9's reward normalization made `V` well-conditioned
(EV 0.47→0.98, value_loss ~0.02), which made the advantages trustworthy, which
let the policy exploit the better IMPALA features. The chain is
**value → advantages → policy**; VecNorm fixed the first link. (PPO also
normalizes advantages to mean 0 / std 1 per minibatch — `normalize_advantage=
true` — a final variance-control step.)

---

## 7. Why pair *this* network with *this* algorithm

- **IMPALA CNN** solves the *perception* problem: enough depth + residual
  gradient paths to represent a cluttered bullet-hell scene where the Nature-CNN
  plateaus. A better `s → features` map.
- **PPO** solves the *credit assignment + stability* problem: the clipped
  objective gives reliable, low-variance on-policy updates without the
  brittleness of vanilla policy gradients or the off-policy bookkeeping (replay,
  target nets, V-trace) that DQN / the IMPALA-algorithm need.

They compose cleanly because the feature extractor is a swappable module. The
network changed *what* the agent could see; PPO's learning rule never changed.

---

## 8. v9 hyperparameter reference

| Group | Setting | Value |
|---|---|---|
| Env | observation | 84×84 grayscale, frame_stack 4 |
| | action_repeat | 8 (~7.5 Hz decisions) |
| | action space | `Discrete(9)` movement combos |
| | auto_fire period | 12 emulator frames (~5 Hz) |
| Model | feature extractor | IMPALA ResNet (`impala`) |
| | features_dim | 256 |
| | total params | 1,131,098 |
| PPO | n_envs | 8 |
| | n_steps | 128 (→ 1024 / rollout) |
| | batch_size | 256 |
| | n_epochs | 4 |
| | gamma | 0.99 |
| | gae_lambda | 0.95 |
| | clip_range | 0.1 |
| | ent_coef | 0.02 → 0.001 (linear schedule) |
| | vf_coef | 0.5 |
| | learning_rate | 1e-4 |
| | max_grad_norm | 0.5 |
| | normalize_reward | true (VecNormalize, reward only) |
| Run | total_timesteps | 4,000,000 |
| | eval | every 100k, 5 deterministic episodes |
| | best result | eval return **7168 @ 2.7M** |

Configs: [`configs/ppo_v9.yaml`](../configs/ppo_v9.yaml),
[`configs/env_v9.yaml`](../configs/env_v9.yaml). Network:
[`src/retro_rl/models/impala.py`](../src/retro_rl/models/impala.py).
