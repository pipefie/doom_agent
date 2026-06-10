# Doom RL Agent

> A Reinforcement Learning project that trains agents to play Doom (ViZDoom) using Curriculum Learning — progressing from basic target practice to full Deathmatch combat. The work is split across two independent branches, each implementing a different RL algorithm.

---

## Repository Overview

This repository contains **two independent but parallel implementations** of a Doom-playing RL agent, developed in separate branches. Both share the same overall objective — master Doom through Curriculum Learning — but approach it with different algorithms, architectures, and engineering choices.

| Branch | Algorithm | Developer | Status |
|---|---|---|---|
| [**`mike` →**](#branch-mike--dqn-agent) | Deep Q-Network (DQN) | Mike | ✅ Complete — 5-level curriculum, full Deathmatch |
| [**`cleanRL-dev` →**](#branch-cleanrl-dev--ppo-agent) | Proximal Policy Optimization (PPO) | Pipe | ✅ Complete — Phase 5.13 "Matador" |

The `master` branch contains only the shared scaffolding (base PPO skeleton, environment wrapper, and `pyproject.toml`). All substantive work lives in the two branches above.

---

## The Shared Idea: Curriculum Learning

Both branches solve the same fundamental problem: throwing an RL agent directly into a Deathmatch fails — the task is too complex, the reward too sparse, and the agent never discovers useful behaviours. The solution in both cases is **Curriculum Learning**: a sequential training pipeline where the agent masters one skill at a time, and its learned weights are transferred to the next, harder scenario.

```
Basic (shoot stationary target)
  ↓
Defend the Center (360° aiming, multiple enemies)
  ↓
Deadly Corridor (navigate + fight, enemy-blocked path)
  ↓
Health Gathering Supreme (pure navigation, no shooting)
  ↓
Deathmatch (full combat integration)
```

The two branches differ in *how* they implement this pipeline — the algorithm, the network architecture, the reward engineering, and the engineering challenges they each solve.

---

## Branch `mike` — DQN Agent

📁 **[`mike` branch](https://github.com/pipefie/doom_agent/tree/mike)**

Implements a **Deep Q-Network** agent trained through a 5-level curriculum. The agent learns a Q-function mapping (state, action) → expected future return, and at each curriculum level it inherits the weights from the previous one.

**Algorithm:** DQN with experience replay, target network, and ε-greedy exploration.

**Architecture:** NatureCNN (3 Conv layers + 512-unit FC) → Q-head outputting 12 discrete action scores.

**Key engineering decisions:**
- **Universal 12-action space** enforced across all levels to prevent shape-mismatch errors on checkpoint loading.
- **Reward shaping per level:** sparse signals for Basic/Defend, hybrid damage/ammo/health shaping for Corridor, survival-only rewards for Health Gathering, hierarchical kill/resource/death rewards for Deathmatch.
- **Level 4 "Navigation Bridge":** disabling the attack button entirely at Level 4 forces the agent to master movement — this proved critical to produce an agile Deathmatch agent.
- **Sub-curriculum at Level 5:** 5 separate training phases stepping through Doom engine difficulty 1→5 (1M timesteps each).

**Trained model checkpoints included:** `doom_dqn_model.pth` through `doom_dqn_level5_final.pth` + individual skill checkpoints.

📖 **[Full `mike` branch README →](https://github.com/pipefie/doom_agent/blob/cleanRL-dev/README.md)**

---

## Branch `cleanRL-dev` — PPO Agent

📁 **[`cleanRL-dev` branch](https://github.com/pipefie/doom_agent/tree/cleanRL-dev)**

Implements a **Proximal Policy Optimization** agent using the CleanRL framework, progressing through 5 phases (13 total iterations) up to the "Matador" policy — a Deathmatch agent that strafe-dodges while engaging enemies.

**Algorithm:** PPO with clipped surrogate objective, GAE advantage estimation, and entropy regularization.

**Architecture:** NatureCNN (3 Conv layers → 512-unit FC) + **LSTM** (512 units) → Actor head (7 actions) + Critic head (value function).

**Key engineering decisions:**
- **LSTM for partial observability:** recurrent memory is essential for the Deadly Corridor — the agent must remember enemies it saw around corners seconds ago.
- **"Zero-Sum" reward pivot:** engine ACS rewards (±20 pts) were silenced (`base_reward = 0.0`) and replaced entirely by Python-side curriculum signals, solving the "Suicider" convergence failure.
- **Universal 7-button standard:** shared action space `[FWD, BACK, TURN_L, TURN_R, STRAFE_L, STRAFE_R, ATK]` prevents head shape mismatches during transfer.
- **Multi-hot combo actions:** the agent picks from a pre-defined matrix of button combinations (e.g. `MOVE_FORWARD + ATTACK`), enabling simultaneous movement and shooting.
- **Phase 5.13 "Matador":** disabling the Pain Rage multiplier and raising the health penalty forces the agent to strafe and dodge rather than face-tank enemies.
- **8 parallel environments** via `SyncVectorEnv` for PPO rollout collection.
- **Custom OpenCV render loop** in `eval_doom_agent.py` to bypass `GLX BadValue` errors on headless Linux servers.

**Final results:**

| Phase | Scenario | Mean Reward | Notes |
|---|---|---|---|
| 1 | Basic | 100.96 / 101 max | Near-perfect, deterministic policy |
| 2 | Defend the Center | ~60 / 111 max | 360° aiming transferred |
| 3 | Deadly Corridor | Converged | LSTM "Slice the Pie" clearing |
| 4 | Health Gathering | ~22.0 | Navigation without combat |
| 5.13 | Deathmatch "Matador" | — | Strafe-dodge behaviour emerged |

📖 **[Full `cleanRL-dev` README →](https://github.com/pipefie/doom_agent/blob/cleanRL-dev/README.md)**

---

## Algorithm Comparison

| | DQN (`mike`) | PPO (`cleanRL-dev`) |
|---|---|---|
| **Update rule** | Off-policy (experience replay) | On-policy (rollout buffer) |
| **Memory** | Replay buffer (25K–100K transitions) | 8 × 128-step rollouts |
| **Parallelism** | 1 environment (standard DQN) | 8 parallel environments |
| **Recurrence** | None | LSTM (512 units) |
| **Action space** | 12 discrete combos | 7 buttons, multi-hot combos |
| **Exploration** | ε-greedy (1.0 → 0.05) | Entropy regularization |
| **Partial observability** | Frame stacking only | Frame stacking + LSTM |
| **Curriculum depth** | 5 levels + 5 Deathmatch sub-phases | 5 phases + 13 iterations |
| **Engine reward** | Used (scaled) | Silenced, Python-side only |

---

## Shared Components (master branch)

The `master` branch provides the scaffolding that both branches start from:

```
doom_agent/
├── ppo.py              # Base CleanRL PPO implementation (CartPole template)
├── wrappers.py         # Base VizDoomGym wrapper (3-action, single-channel)
├── make_env.py         # Environment factory with FrameStack
├── main.py             # Entry point stub
├── custom_corridor.cfg # Shared corridor config
├── pyproject.toml      # Shared dependencies
└── _vizdoom.ini        # ViZDoom engine configuration
```

---

## Quick Start

### Install dependencies (both branches use `uv`)

```bash
uv sync
```

Or with pip:

```bash
pip install vizdoom gymnasium torch numpy opencv-python tensorboard tyro
```

### Run the DQN agent (`mike` branch)

```bash
# Watch the final Deathmatch agent
python play_dqn_level5.py

# Train from scratch (Level 1 → 5 in sequence)
python train_dqn.py
python train_dqn_level2.py --load_model doom_dqn_model.pth
python train_dqn_level3.py --load_model doom_dqn_level2.pth
python train_dqn_level4.py --load_model doom_dqn_level3.pth
python train_dqn_level5.py --load_model doom_dqn_level4.pth
```

### Run the PPO agent (`cleanRL-dev` branch)

```bash
# Evaluate the Matador (Phase 5.13)
uv run python eval_doom_agent.py \
    --checkpoint checkpoints/deathmatch_phase5_matador_...pt \
    --scenario-cfg configs/deathmatch_simple.cfg \
    --use-lstm --lstm-hidden-size 512 \
    --health-penalty 0.5 --pain-rage-multiplier 1.0 \
    --episodes 5 --render --cuda

# Train from a scenario
uv run python doom_ppo_deadly_corridor.py \
    --scenario-cfg configs/basic.cfg \
    --use-lstm --use-combo-actions --cuda --track
```

---

## Requirements

| Package | Purpose |
|---|---|
| `vizdoom >= 1.2.4` | Doom engine and environment |
| `gymnasium >= 1.2.2` | RL environment API |
| `torch >= 2.9.1` | Neural network training |
| `numpy >= 2.2.6` | Array operations |
| `opencv-python >= 4.11.0.86` | Frame preprocessing and rendering |
| `tensorboard >= 2.20.0` | Training metrics logging |
| `tyro >= 0.9.35` | CLI argument parsing |

---

*Course: Advanced Machine Learning*
