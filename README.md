# Doom DQN Agent with Curriculum Learning 👹🤖

This repository implements a **Deep Q-Network (DQN)** agent capable of playing **Doom** (VizDoom) at a high proficiency level. The core of this project is the use of **Curriculum Learning**: instead of throwing the agent directly into a Deathmatch, we train it sequentially through 5 levels of increasing complexity, transferring the learned weights and "skills" from one scenario to the next.


## 🧠 The Curriculum Learning Strategy

The agent undergoes a 5-stage training process. To ensure **knowledge transfer**, we standardized the action space (neural network output) across all scenarios, even if some actions (like strafing) weren't originally necessary for the simpler levels.

| Stage | Scenario | Focus Skill | Difficulty |
| :--- | :--- | :--- | :--- |
| **1** | `basic.cfg` | **Causality**: Moving forward & Shooting. | Very Easy |
| **2** | `defend_the_center.cfg` | **Aiming**: Turning & shooting multiple enemies. | Easy |
| **3** | `deadly_corridor.cfg` | **Efficiency**: Killing while dodging. Damage/Ammo management. | Medium |
| **4** | `health_gathering.cfg` | **Navigation**: Surviving *without* shooting. Pure movement. | Hard |
| **5** | `deathmatch.cfg` | **Combat**: Full integration of all skills. | Nightmare (Skill 1-5) |

---

## 🛠️ Architecture & Setup

### Requirements
This project is managed with `uv`. To install dependencies:

```bash
uv sync
# Or manually via pip
pip install gymnasium vizdoom torch numpy opencv-python tensorboard tyro
```

### Neural Network (DQN)
We use a standard CNN (based on the Nature DQN paper) adapted for the Doom screen buffer:
*   **Input**: 84x84 Grayscale images, Stack of 4 frames (FrameStack).
*   **Backbone**: 3 Convolutional Layers + ReLU.
*   **Head**: Fully Connected Layers (512 units) $\to$ **12 Discrete Actions**.

### Environment Wrappers & Reward Shaping
One of the biggest challenges in RL is sparse rewards. We implemented custom Gym Wrappers to shape the rewards and accelerate convergence.

| Level | Action Space | Reward Logic (Shaping) |
| :--- | :--- | :--- |
| **1-2 (Basic/Defend)** | **12 Actions** (Modified CFG to include strafe) | **Sparse (Default):** The agent learns purely from the game engine signals (+1 Kill, -1 Death). |
| **3 (Corridor)** | **12 Actions** | **Hybrid:** We override the game rewards. <br>• `+` Damage dealt & Kills.<br>• `-` Ammo usage (efficiency).<br>• `-` Health loss (defense). |
| **4 (Health)** | **6 Actions** (Restricted: No Attack) | **Survival:**<br>• `+1.0` Health Kit pickup.<br>• `+0.05` Living reward (per frame).<br>• `-5.0` Death penalty.<br>*Forces the agent to learn movement/strafing without relying on shooting.* |
| **5 (Deathmatch)** | **12 Actions** | **Hierarchical:**<br>• `+100` Kill (Top priority).<br>• `+0.5` Armor / `+0.05` Ammo (Resource gathering).<br>• `-100` Death (Severe punishment). |

---

## 🚀 Training Pipeline

The training is sequential. You must train the levels in order, as each script loads the `.pth` model from the previous level.

### 1. Basic Training
Establishes the connection between visual input and button presses.
```bash
python train_dqn.py --scenario_path configs/basic.cfg --total_timesteps 500000
```
*Output:* `doom_dqn_model.pth`

### 2. Defend the Center (Transfer)
Loads the Basic model. Teaches the agent to rotate and manage 360° threats.
```bash
python train_dqn_level2.py --load_model doom_dqn_model.pth
```
*Output:* `doom_dqn_level2.pth`

### 3. Deadly Corridor
Loads Level 2. Introduces the concept of "damage penalty" and ammo conservation.
```bash
python train_dqn_level3.py --load_model doom_dqn_level2.pth
```
*Output:* `doom_dqn_level3.pth`

### 4. Health Gathering Supreme (Critical)
**Drastic change:** We disable the attack button. The agent loads the Level 3 weights but must learn to navigate strictly to survive. This breaks the local minimum where the agent just stands still and shoots.
```bash
python train_dqn_level4.py --load_model doom_dqn_level3.pth
```
*Output:* `doom_dqn_level4.pth`

### 5. Deathmatch (The Exam)
Finally, we train on Deathmatch. This stage is further divided into sub-curriculums by increasing the internal Doom Engine difficulty (`doom_skill` 1 to 5) for 1M timesteps each.
```bash
python train_dqn_level5.py --load_model doom_dqn_level4.pth --scenario_path configs/deathmatch_simple.cfg
```

---

## 🎮 How to Play (Inference)

To watch the trained agent play in real-time (with the window visible):

```bash
# Watch the final Deathmatch agent
python play_dqn_level5.py

# Watch previous levels
python play_dqn.py         # Basic
python play_dqn_level2.py  # Defend
```

---

## 📂 File Structure

```text
├── configs/                 # .cfg and .wad files (Doom scenarios)
│   ├── basic.cfg            # Modified to match action space
│   ├── defend_the_center.cfg
│   ├── deadly_corridor.cfg
│   ├── health_gathering_supreme.cfg
│   └── deathmatch.cfg
├── wrappers.py              # Custom Gymnasium wrappers (Reward Shaping logic)
├── make_env.py              # Environment factory (Grayscale, FrameStack)
├── train_dqn.py             # Level 1 Trainer
├── train_dqn_level2.py      # Level 2 Trainer (Defend)
├── train_dqn_level3.py      # Level 3 Trainer (Corridor)
├── train_dqn_level4.py      # Level 4 Trainer (Health)
├── train_dqn_level5.py      # Level 5 Trainer (Deathmatch)
├── play_dqn_*.py            # Inference scripts
├── requirements.txt         # (or uv.lock)
└── README.md
```

## 📈 Results & Observations

*(Place your Tensorboard graphs here: Loss, Average Reward per Episode)*

*   **Standardization:** Adding `Move Left/Right` to the Basic config was crucial to allow weight loading in later levels without shape mismatch errors in the final Linear layer.
*   **The "Health Gathering" Bridge:** The jump from *Deadly Corridor* to *Deathmatch* was too high. The agent would get stuck. Introducing *Health Gathering* forced the agent to master movement, which resulted in a much more agile agent in the final Deathmatch.

---

## 📜 Credits

Based on the [ViZDoom](https://github.com/mwydmuch/ViZDoom) environment and the Gymnasium API.