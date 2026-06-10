# Doom DQN Agent — Curriculum Learning with Deep Q-Network

> A Deep Q-Network agent that masters Doom through a 5-level curriculum, transferring learned weights sequentially from basic target practice to full Deathmatch combat.

**Branch:** `mike` | **Algorithm:** DQN | **Engine:** ViZDoom + Gymnasium + PyTorch

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Neural Network Architecture](#3-neural-network-architecture)
4. [Environment & Wrappers](#4-environment--wrappers)
5. [Curriculum Learning Pipeline](#5-curriculum-learning-pipeline)
6. [Training Configuration](#6-training-configuration)
7. [Key Engineering Decisions](#7-key-engineering-decisions)
8. [Inference / How to Watch](#8-inference--how-to-watch)
9. [Results & Observations](#9-results--observations)
10. [How to Run](#10-how-to-run)
11. [Dependencies](#11-dependencies)

---

## 1. Project Overview

This branch implements a **Deep Q-Network (DQN)** agent trained to play Doom (via the ViZDoom simulator) at increasing levels of difficulty. The core insight is that Deathmatch is too complex to learn directly from scratch — the reward is too sparse and the task space too large. Instead, the agent is trained sequentially through **5 scenarios of increasing complexity**, with its network weights carried forward from one level to the next.

The result is an agent that builds skills incrementally: first learning that pressing buttons causes things to happen, then learning to aim, then to navigate, then to fight in a full combat environment.

---

## 2. Repository Structure

```
doom_agent-mike/
├── configs/
│   ├── README.md
│   ├── basic.cfg                     # Scenario 1: Stationary target
│   ├── basic.wad
│   ├── defend_the_center.cfg         # Scenario 2: 360° surrounded
│   ├── defend_the_center.wad
│   ├── deadly_corridor.cfg           # Scenario 3: Navigate + fight
│   ├── deadly_corridor.wad
│   ├── health_gathering_supreme.cfg  # Scenario 4: Survival only
│   ├── health_gathering_supreme.wad
│   ├── deathmatch_simple.cfg         # Scenario 5: Full combat
│   └── deathmatch.wad
├── wrappers.py                       # VizDoomGym: reward shaping, action encoding
├── make_env.py                       # Environment factory (grayscale, FrameStack)
├── train_dqn.py                      # Level 1 trainer (Basic)
├── train_dqn_level2.py               # Level 2 trainer (Defend the Center)
├── train_dqn_level3.py               # Level 3 trainer (Deadly Corridor)
├── train_dqn_level4.py               # Level 4 trainer (Health Gathering)
├── train_dqn_level5.py               # Level 5 trainer (Deathmatch, skills 1–5)
├── play_dqn.py                       # Inference: Basic
├── play_dqn_level2.py                # Inference: Defend the Center
├── play_dqn_level3.py                # Inference: Deadly Corridor
├── play_dqn_level4.py                # Inference: Health Gathering
├── play_dqn_level5.py                # Inference: Deathmatch (final)
├── play_ppo.py                       # Inference: PPO model (compatibility)
├── check_actions.py                  # Utility: inspect available buttons in a CFG
├── ppo.py                            # PPO implementation (carried from master)
├── doom_dqn_model.pth                # Checkpoint: Level 1
├── doom_dqn_level2.pth               # Checkpoint: Level 2
├── doom_dqn_level3.pth               # Checkpoint: Level 3
├── doom_dqn_level4.pth               # Checkpoint: Level 4
├── doom_dqn_level5_final.pth         # Checkpoint: Level 5 (final)
├── doom_dqn_level5_skill1..5.pth     # Checkpoints: Level 5 sub-skills
├── doom_agent_model.pth              # Checkpoint: initial PPO exploration
├── runs/                             # TensorBoard event files
├── wrapper_level3.txt                # Reference wrapper for Level 3
├── wrappers_level4.txt               # Reference wrapper for Level 4
├── pyproject.toml
└── uv.lock
```

---

## 3. Neural Network Architecture

The agent uses the **Nature DQN** architecture (Mnih et al., 2015), adapted for the Doom screen buffer.

### Q-Network

```
Input: (batch, 4, 84, 84)   ← 4 stacked grayscale frames
         ↓
  Conv2d(4  → 32,  kernel=8, stride=4)  + ReLU
         ↓
  Conv2d(32 → 64,  kernel=4, stride=2)  + ReLU
         ↓
  Conv2d(64 → 64,  kernel=3, stride=1)  + ReLU
         ↓
  Flatten → Linear(3136 → 512)          + ReLU
         ↓
  Linear(512 → 12)   ← Q-value for each of 12 action combos
```

All layers are initialised with **orthogonal initialisation** (std=√2) and zero bias, following the CleanRL convention for stable early training.

### Why 4 Stacked Frames?

A single frame provides no velocity information — the agent cannot tell whether an enemy is moving toward it or away. Stacking 4 consecutive grayscale frames gives the network an implicit representation of motion without requiring an explicit recurrent architecture. The frame skip of 4 (acting every 4 engine ticks) extends the effective memory window to ~0.4 seconds of game time per stack.

### DQN vs Actor-Critic

Unlike the PPO branch which uses an Actor-Critic (two output heads), DQN is a **value-based** method with a single head outputting Q-values. The policy is implicit: at inference time, take the action with the highest Q-value (`argmax`); during training, act ε-greedily to explore.

---

## 4. Environment & Wrappers

### `VizDoomGym` (`wrappers.py`)

A custom `gymnasium.Env` wrapper around the raw ViZDoom game object. Its main responsibilities:

**1. Multi-hot action encoding.**
ViZDoom accepts button vectors (e.g. `[1, 0, 0, 0, 0, 0, 1]` = Forward + Attack). Standard Gym environments expose a single integer action. The wrapper maintains a fixed list of 12 pre-defined button combinations and maps integer → button vector internally:

```python
self.actions = [
    [0,0,0,0,0,0,0],  # 0: Idle
    [1,0,0,0,0,0,0],  # 1: Forward
    [0,0,1,0,0,0,0],  # 2: Turn Left
    [0,0,0,1,0,0,0],  # 3: Turn Right
    [0,0,0,0,1,0,0],  # 4: Strafe Left
    [0,0,0,0,0,1,0],  # 5: Strafe Right
    [0,0,0,0,0,0,1],  # 6: Attack
    [1,0,0,0,0,0,1],  # 7: Forward + Attack  ← enables Run & Gun
    [0,0,0,0,1,0,1],  # 8: Strafe Left + Attack
    [0,0,0,0,0,1,1],  # 9: Strafe Right + Attack
    [0,0,1,0,0,0,1],  # 10: Turn Left + Attack
    [0,0,0,1,0,0,1],  # 11: Turn Right + Attack
]
```

**2. Observation preprocessing.**
Raw ViZDoom screen buffers are grayscale (84×84 after `set_screen_format(GRAY8)`) — the wrapper exposes them as `(84, 84)` arrays compatible with the FrameStack wrapper.

**3. Reward shaping per level.**
Different wrappers are used at different curriculum stages (see Section 5 for per-level details). The wrapper tracks `prev_health`, `prev_kills`, `prev_ammo`, and `prev_armor` across steps to compute delta-based reward signals.

### `make_env.py`

Environment factory that chains:
1. `VizDoomGym(game)` — raw ViZDoom → Gym wrapper
2. `gym.wrappers.RecordEpisodeStatistics` — tracks episode return and length
3. `gym.wrappers.FrameStackObservation(env, stack_size=4)` — stacks 4 frames for motion perception

Key configuration flags set before `game.init()`:
- `set_window_visible(False)` — headless training
- `set_mode(vzd.Mode.PLAYER)` — standard player mode
- `set_screen_format(vzd.ScreenFormat.GRAY8)` — forces single grayscale channel
- `set_sound_enabled(False)` — prevents audio backend errors on Linux

---

## 5. Curriculum Learning Pipeline

The agent trains through 5 sequential levels. Each level **loads the checkpoint from the previous one** (`--load_model`), enabling knowledge transfer. The action space (12 actions) is kept constant across all levels to ensure the network's final linear layer never changes shape.

### Level 1 — Basic

**Scenario:** A stationary imp in a room. The agent must move and shoot it.

**Reward logic:** Sparse — default engine signals only (+1 kill, -1 death). The task is simple enough that the Q-network can discover the kill reward through random exploration.

**Key challenge:** The agent must learn that pressing the Attack button while facing the enemy produces a reward. Without any shaping, this credit-assignment problem is solvable because the corridor is short and there is only one enemy.

**Output checkpoint:** `doom_dqn_model.pth`

**Training:**
```bash
python train_dqn.py --scenario_path configs/basic.cfg --total_timesteps 500000
```

---

### Level 2 — Defend the Center

**Scenario:** The agent stands in the centre of a circular arena while enemies approach from all directions.

**Reward logic:** Sparse (default engine). The skill being acquired is **360° rotation and multi-target aiming** — the agent can no longer face one direction and win.

**Transfer benefit:** The convolutional filters from Level 1 (edge detection, enemy silhouette recognition) are directly useful here. Only the higher-level action-selection policy needs to change.

**Output checkpoint:** `doom_dqn_level2.pth`

**Training:**
```bash
python train_dqn_level2.py --load_model doom_dqn_model.pth
```

---

### Level 3 — Deadly Corridor

**Scenario:** A corridor populated by enemies of increasing strength. The agent must fight through to a green health vest at the far end.

**Reward logic:** Hybrid reward shaping overrides engine defaults:

| Signal | Delta variable | Weight |
|---|---|---|
| Kill reward | `KILLCOUNT` delta | +100 per kill |
| Health penalty | `HEALTH` delta | proportional loss |
| Ammo efficiency | `AMMO2` delta | −0.1 per bullet spent |
| Living reward | Per step | +0.01 |

The ammo penalty discourages random spraying; the living reward encourages forward progress over camping.

**Output checkpoint:** `doom_dqn_level3.pth`

**Training:**
```bash
python train_dqn_level3.py --load_model doom_dqn_level2.pth
```

---

### Level 4 — Health Gathering Supreme (The Bridge)

**Scenario:** An acid-floor maze where the agent takes constant damage and must collect health kits to survive. There are no enemies to shoot.

**Reward logic:** Survival-only:

| Signal | Value |
|---|---|
| Health kit pickup | +1.0 per kit |
| Living reward | +0.05 per frame |
| Death penalty | −5.0 |

**Action space change:** The Attack button is removed at this level (restricted to 6 navigation-only actions). The network head still has 12 outputs, but attack-related actions receive zero reward and the policy is not encouraged to use them.

**Why this level is critical.** The direct jump from Deadly Corridor to Deathmatch produces an agent that stands still and shoots. Level 4 breaks this failure mode by forcing the agent to develop pure movement skills — forward motion, turning, and collision avoidance — in an environment where shooting provides no benefit. The resulting Level 4 weights encode navigation competence that Level 5 can build combat on top of.

**Output checkpoint:** `doom_dqn_level4.pth`

**Training:**
```bash
python train_dqn_level4.py --load_model doom_dqn_level3.pth
```

---

### Level 5 — Deathmatch (The Exam)

**Scenario:** Full Deathmatch on a custom map with respawning enemies. The agent must integrate all previously learned skills: navigation, aiming, resource gathering, and survival.

**Reward logic:** Hierarchical — kills dominate, with resource and survival signals providing supplementary structure:

| Signal | Value |
|---|---|
| Kill (FRAGCOUNT delta) | +100 |
| Armor pickup (ARMOR delta) | +0.5 per unit |
| Ammo pickup (AMMO delta) | +0.1 per round |
| Health change (HEALTH delta) | +1.0 per HP gained, −1.0 per HP lost |
| Living reward | +0.01 per step |

**Sub-curriculum (5 difficulty tiers):** The Doom engine exposes an internal difficulty parameter (`doom_skill` 1–5) that controls enemy speed, damage, and aggressiveness. Level 5 is trained in five consecutive 1M-timestep phases, stepping through each difficulty:

```
Skill 1 (easiest): 1M steps → doom_dqn_level5_skill1.pth
Skill 2:          1M steps → doom_dqn_level5_skill2.pth
Skill 3:          1M steps → doom_dqn_level5_skill3.pth
Skill 4:          1M steps → doom_dqn_level5_skill4.pth
Skill 5 (hardest): 1M steps → doom_dqn_level5_skill5.pth → doom_dqn_level5_final.pth
```

Each skill stage starts with a high initial epsilon (`start_e = 0.6`) to allow the agent to re-explore under the new difficulty before settling into the learned policy.

**Total training:** ~15M timesteps across skills 1–5.

**Training:**
```bash
python train_dqn_level5.py --load_model doom_dqn_level4.pth --scenario_path configs/deathmatch_simple.cfg
```

---

## 6. Training Configuration

Key DQN hyperparameters shared across levels (from `train_dqn.py`):

| Hyperparameter | Level 1–4 | Level 5 |
|---|---|---|
| `buffer_size` | 25,000 | 100,000 |
| `batch_size` | 32 | 32 |
| `gamma` | 0.99 | 0.99 |
| `learning_rate` | 2.5e-4 | 2.5e-4 |
| `target_network_frequency` | 1,000 steps | 1,000 steps |
| `tau` (update style) | 1.0 (hard copy) | 1.0 (hard copy) |
| `start_e` (ε initial) | 1.0 | 0.6 |
| `end_e` (ε final) | 0.05 | 0.01 |
| `exploration_fraction` | 0.5 | 0.5 |
| `learning_starts` | 10,000 | 10,000 |
| `train_frequency` | 4 steps | 4 steps |
| `num_envs` | 1 | 1 |

**Why hard target network updates (`tau = 1.0`)?** Soft updates (`tau < 1.0`) blend target and online weights continuously, which can smooth learning but introduces lag. Hard copies every 1,000 steps provides a more stable reference Q-function for bootstrapping, which is the standard choice for the Nature DQN setting.

**Why 1 environment?** Standard DQN with experience replay does not benefit from parallel environments in the same way as on-policy algorithms like PPO. The replay buffer provides sufficient decorrelation between training samples.

---

## 7. Key Engineering Decisions

### Universal 12-Action Space

**Problem:** The initial Deadly Corridor scenario only needed 7 buttons; Deathmatch needed 12. Loading a Level 2 checkpoint into a Level 3 network with a different output size crashes with a shape mismatch in the final linear layer.

**Solution:** The 12-action combo space (`SHARED_ACTION_BUTTONS`) is enforced across **all** scenarios from Level 1 onwards. The CFG files for simpler scenarios are modified to expose all 7 buttons even if only a subset is useful. This keeps the Q-head dimension constant throughout the entire curriculum, making checkpoint loading trivially safe.

### Reward Shaping as a Curriculum Design Choice

Sparse rewards work at Levels 1 and 2 because the task is simple enough for random exploration to discover kills. At Level 3 (Deadly Corridor), the agent must fight through multiple enemies with limited ammo — the chance of randomly reaching the vest and receiving the sparse end-reward is essentially zero. Reward shaping (damage, kill, health, ammo signals) provides a dense gradient that guides exploration toward useful behaviours.

At Level 5, the hierarchical reward structure (kill >> resource >> survival) encodes the priority ordering of objectives: winning fights matters more than collecting ammo, but both matter more than staying alive passively.

### The Health Gathering Bridge

Empirically, the jump from Level 3 to Level 5 (skipping Level 4) produces an agent with a pathological behaviour: it learns to stand in a corner and shoot, because that is the safest way to accumulate kill rewards in the early stages. Level 4 breaks this by penalising all actions that don't involve movement and by providing zero reward for shooting. The agent is forced to discover navigation patterns that Level 5 can integrate with combat.

### TensorBoard Logging

All training scripts write event files to `runs/` using `SummaryWriter`. Key logged metrics: episodic return, episodic length, epsilon value, Q-value loss, and SPS (steps per second). The `runs/` directory in this branch contains training logs from all development iterations.

---

## 8. Inference / How to Watch

Each level has a corresponding `play_dqn_levelN.py` script that:
1. Creates a **visible** ViZDoom window (`set_window_visible(True)`)
2. Loads the trained `.pth` checkpoint
3. Runs 20 greedy episodes (argmax Q-value, no exploration)

```bash
# Watch the final Deathmatch agent (Level 5)
python play_dqn_level5.py

# Watch earlier levels
python play_dqn.py          # Level 1: Basic
python play_dqn_level2.py   # Level 2: Defend the Center
python play_dqn_level3.py   # Level 3: Deadly Corridor
python play_dqn_level4.py   # Level 4: Health Gathering
```

A `time.sleep(0.05)` delay between steps slows playback to near-human speed for visibility.

---

## 9. Results & Observations

| Level | Scenario | Timesteps | Key Outcome |
|---|---|---|---|
| 1 | Basic | 500K | Agent reliably kills stationary target |
| 2 | Defend the Center | 500K+ | 360° aiming acquired via transfer |
| 3 | Deadly Corridor | 1M+ | Hybrid shaping enables corridor navigation + combat |
| 4 | Health Gathering | 500K | Pure navigation mastered; attack drive suppressed |
| 5 | Deathmatch (skills 1–5) | ~10M total | Combat integration; increasingly aggressive opponents |

**Standardisation insight:** Adding `MOVE_LEFT` / `MOVE_RIGHT` (strafing) to the Basic config from the start was critical. Without it, the Level 1 agent's Q-head has a different output size than Level 3 onwards, breaking weight loading. Standardising the action space early costs nothing (the agent simply never uses strafe at Level 1) but saves all checkpoint compatibility issues downstream.

**The Health Gathering bridge:** The clearest empirical finding is that the Level 3 → Level 5 jump produces a stationary shooter, while the Level 3 → Level 4 → Level 5 path produces a mobile, evasive combatant. Level 4 provides the movement "muscle memory" that allows Level 5 combat to be dynamic rather than static.

---

## 10. How to Run

### Install

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install gymnasium vizdoom torch numpy opencv-python tensorboard tyro
```

### Train (in sequence)

```bash
# Level 1
python train_dqn.py --scenario_path configs/basic.cfg --total_timesteps 500000

# Level 2 (loads Level 1 checkpoint)
python train_dqn_level2.py --load_model doom_dqn_model.pth

# Level 3 (loads Level 2 checkpoint)
python train_dqn_level3.py --load_model doom_dqn_level2.pth

# Level 4 (loads Level 3 checkpoint)
python train_dqn_level4.py --load_model doom_dqn_level3.pth

# Level 5 (loads Level 4 checkpoint, trains through 5 difficulty tiers)
python train_dqn_level5.py --load_model doom_dqn_level4.pth
```

### Watch TensorBoard

```bash
tensorboard --logdir runs/
```

### Inspect action space of a scenario

```bash
python check_actions.py
# Edit the script to change the target CFG file
```

---

## 11. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `vizdoom` | ≥ 1.2.4 | Doom engine, WAD loading, screen buffer |
| `gymnasium` | ≥ 1.2.2 | RL environment API, FrameStackObservation |
| `torch` | ≥ 2.9.1 | Q-Network training and inference |
| `numpy` | ≥ 2.2.6 | Array operations |
| `opencv-python` | ≥ 4.11.0.86 | Frame resize and grayscale conversion |
| `tensorboard` | ≥ 2.20.0 | Training metrics visualisation |
| `tyro` | ≥ 0.9.35 | CLI argument parsing via dataclasses |

---

*Course: Advanced Machine Learning — Doom RL Agent, branch `mike`*
