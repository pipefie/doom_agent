# ViZDoom PPO Agent: A Comprehensive Engineering Report

**Project Status:** 🟢 Completed (Phase 5.13 "Matador")
**Date:** 2025-12-11
**Engine:** ViZDoom + Gymnasium + PyTorch (CleanRL)

---

## 1. Executive Summary
This project implements a **Proximal Policy Optimization (PPO)** agent capable of mastering increasingly complex Doom scenarios through **Curriculum Learning**. Starting from a stationary target practice ("Basic"), the agent evolves into a tactical survivor capable of navigating deadly corridors, managing resources ("Scavenger"), and evading enemy fire ("Matador") in a 3D Deathmatch environment.

This document acts as a complete **Technical Project Memoire**, detailing the architecture, every phase of experimentation, specific training commands, quantitative results, and deep-dive explanations of the engineering challenges solved along the way.

---

## 2. System Architecture

### Neural Network Architecture
The agent uses a standard Actor-Critic architecture tailored for 3D inputs:
*   **Visual Encoder:** `NatureCNN`
    *   Conv1: 32 filters, 8×8 kernel, stride 4
    *   Conv2: 64 filters, 4×4 kernel, stride 2
    *   Conv3: 64 filters, 3×3 kernel, stride 1
    *   Flatten → Linear to 512 units with ReLU.
*   **Memory (Recurrent):** **LSTM** (512 units)
    *   Crucial for "Deadly Corridor" to handle partial observability (e.g., remembering enemies around corners).
    *   Hidden states are masked on `done` flags to ensure clean episodic memory resets.
*   **Heads:**
    *   **Actor:** Categorical distribution over 7 discrete buttons (`FWD`, `BACK`, `TURN_L`, `TURN_R`, `STRAFE_L`, `STRAFE_R`, `ATK`).
    *   **Critic:** Predicts value function $V(s)$ (orthogonal init, std=1.0).

### Environment Stack
*   **Engine:** ViZDoom (Doom II WADs).
*   **Interface:** `Gymnasium` (Standard API) with `SyncVectorEnv` (8 parallel envs).
*   **Dependency Management:** `uv` (`pyproject.toml` + `uv.lock`) for reproducible builds.

---

## 3. Curriculum Learning Log (Detailed)

This section documents the chronological progression of the agent.

### Phase 1: Basic (The Foundation)
**Goal:** Verify the agent can move and shoot stationary targets.
*   **Command:**
    ```bash
    uv run python doom_ppo_deadly_corridor.py --scenario-cfg configs/basic.cfg --use-combo-actions --use-lstm --norm-adv --anneal-lr --ent-coef 0.01 --total-timesteps 1500000
    ```
*   **Results:**
    *   **Mean Reward:** 100.96 (Max Possible: ~101)
    *   **Entropy:** Decayed to 0.05 (Deterministic Policy)
    *   **Value Loss:** ~0.3 (Stable Critic)
*   **Interpretation:** The agent mastered the scenario, learning to immediately kill the monster. The low entropy indicates extreme confidence.
*   **Technical Challenge:** The "Stuck Shooter" bug (see Section 6).

### Phase 2: Defend the Center
**Goal:** 360-degree aiming and turret behavior (encircled by enemies).
*   **Command:**
    ```bash
    uv run python doom_ppo_deadly_corridor.py --scenario-cfg configs/defend_the_center.cfg --load-checkpoint checkpoints/basic_phase1...pt --total-timesteps 2000000
    ```
*   **Results:**
    *   **Mean Reward:** ~60 (Max: ~111)
    *   **Entropy:** ~0.7-0.8 (Healthy Exploration)
*   **Interpretation:** Successfully transferred aiming skills. Hgher entropy was retained, allowing the agent to scan 360 degrees rather than collapsing into a single viewing direction.

### Phase 3: Deadly Corridor (The Zig-Zag Maze)
*   **Goal:** Navigate a corridor filled with enemies to reach a green vest.
*   **Context:** This was the hardest transfer learning step, requiring a shift from "Turret" (Phase 2) to "Navigator" (Phase 3).
*   **Iteration 3.1 (Failed): The "Rusher"**
    *   *Settings:* High `progress_scale` (0.02), Moderate `kill_reward` (15.0).
    *   *Result:* Agent ignored enemies, rushed forward, and died.
    *   *Diagnosis:* The reward for moving forward outweighed the risk of death. The agent learned that "Distance = Points" before "Death = Stop".
*   **Iteration 3.2 (Failed): The "Suicider" (Native Reward Trap)**
    *   *Observation:* Agent vibrated in corners or rushed the first sector, receiving massive random rewards.
    *   *Diagnosis:* Native WAD rewards (±20 point swings) created noise.
    *   *Fix:* **Zero-Sum Pivot** (See Technical Deep Dive C). We silenced the engine (`base_reward=0.0`) to force the agent to obey only our Python curriculum.
*   **Iteration 3.3 (Pivot): "Survival First"**
    *   *Settings:* `progress_scale` lowered to 0.005. `kill_reward` raised to 30.0. `death_penalty` raised to 25.0.
    *   *Result:* The agent learned it *must* clear the path to survive. Moving forward is only profitable if the threat is neutralized.
*   **Iteration 3.4 (Architecture Fix): Action Space Compatibility**
    *   *Problem:* Phase 2 had 3 actions, Phase 3 needed 7. Loading the checkpoint crashed the actor head.
    *   *Solution:* Implemented the **Universal 7-Button Standard** (See Technical Deep Dive B) to allow checkpoint loading.
*   **Iteration 3.5 (Final): "Corner Peeking"**
    *   *Implementation:* Enabled **LSTM** (`--use-lstm`).
    *   *Result:* The recurrent memory allowed the agent to handle partial observability ("I saw an imp around that corner 2 seconds ago"), leading to the "Slice the Pie" tactical behavior where it clears angles before advancing.

### Phase 4: Health Gathering Supreme
**Goal:** Survive in an acid maze (Survival only, no shooting).
*   **Solution:** `living_penalty = -0.1` taught the agent that "Time = Health".
*   **Result:** Mean Return ~22.0. The agent successfully learned navigation without combat.

---

### Phase 5: The Deathmatch Experiment (Detailed Logs)
*This phase required extensive tuning of the agent's psychology/economy.*

*   **Phase 5.1 (Failed): "The Wall Spam Incident"**
    *   *Settings:* `ammo_penalty=0.0`.
    *   *Observation:* Agent hoarded ammo and punched walls. Zero cost to exist led to lazy local optima.
*   **Phase 5.2 - 5.6 (Failed): Economy Tuning**
    *   Tried various penalties (`ammo_penalty`, `wall_penalty`). Agent became either a "Pacifist" (too afraid to shoot) or a "Camper" (hiding in rooms).
*   **Phase 5.7 (Pivot): "The Revenge Mechanic" (Pain Rage)**
    *   *Hypothesis:* Make getting hit explicitly trigger a "Berserker" state.
    *   *Implementation:* `pain_rage_multiplier=4.0`. When damaged, kill rewards quadruple for 2 seconds.
    *   *Result:* Created a "Kamikaze" agent that traded its life for rage points. It worked technically, but failed strategically.
*   **Phase 5.10: "Robust Visualization"**
    *   *Engineering Fix:* `GLX BadValue` errors prevented headless evaluation. We implemented a custom OpenCV render loop in `eval_doom_agent.py` to pipe the raw internal High-Res buffer to the screen, bypassing the native windowing system.
*   **Phase 5.11: "The Arms Dealer"**
    *   *Problem:* Agent ignored Rockets/Plasma.
    *   *Solution:* Rebalanced economy. `ammo_reward` increased (0.05 -> 0.20), `kill_reward` decreased (15.0 -> 10.0).
*   **Phase 5.12: "The Scavenger" (Universal Vision)**
    *   *Problem:* Agent was "blind" to non-bullet ammo.
    *   *Solution:* Modified `VizDoomGymnasiumEnv` to track `AMMO2`-`AMMO5`. Added **Panic Penalty** if `SELECTED_WEAPON_AMMO == 0`.
*   **Phase 5.13: "The Matador" (Final Policy)**
    *   *Problem:* "Rage" mechanic encouraged face-tanking.
    *   *Solution:* **Disabled Rage** (`multiplier=1.0`). Increased `health_penalty` to **0.5**.
    *   *Result:* Taking damage is now strictly net-negative. Agent finally learned to strafe and dodge while fighting.

---

## 4. Technical Challenges & Solutions

### A. The 'Stuck Shooter' Debugging Log (Basic Scenario)
*   **Observation:** Initial agent stood still and shot the wall.
*   **Diagnosis (Sparse Reward):** Kills are rare events. The agent couldn't link the action of "aiming" to the delayed reward of "killing".
*   **Solution:** Introduced **Dense Reward** (`damage_reward=1.0`). This provided immediate feedback for every bullet impact, effectively solving the credit assignment problem for aiming.

### B. Action Space Mismatch (Curriculum Transfer)
*   **Problem:** Transferring from `defend_the_center` (3 actions) to `deadly_corridor` (7 actions) crashed the Neural Net due to shape mismatches in the Actor Head.
*   **Solution:** **Universal 7-Button Standard.**
    We enforced a superset action space for ALL scenarios:
    ```python
    [MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT, ATTACK]
    ```
    This ensures that weights from Phase 2 are fully compatible with Phase 3/5, enabling true Curriculum Learning.

### C. The "Zero-Sum" Pivot: Silencing the Engine (Crucial)
*   **Context:** Early in Phase 3 (Deadly Corridor), the agent exhibited bizarre behavior: suicide-rushing the first sector or vibrating in corners.
*   **Investigation:** We discovered that the WAD file itself contained hidden ACS (Action Code Script) logic that dispensed rewards based on arbitrary sector crossings. These native rewards were massive (±20.0) compared to our shaping (0.01), creating a deafening "noise" that drowned out our learning signal.
*   **The Crucial Fix:** We implemented a **"Python-First" Reward Authority**.
    ```python
    # In VizDoomGymnasiumEnv.step():
    self.game.make_action(...)
    
    # CRITICAL: The Doom engine returns a reward based on internal WAD logic.
    # We explicitly discard this to prevent "Exploding Reward" artifacts.
    base_reward = 0.0 
    
    # We then construct the TOTAL reward strictly from our own calculated deltas:
    reward = base_reward + shape_reward_func(...)
    ```
*   **Why this works:** By force-setting `base_reward = 0.0`, we effectively "deafen" the agent to the game designer's original intentions (which were meant for humans, not RL) and force it to listen **ONLY** to our curriculum (Kill, Survive, Scavenge). This was the turning point that allowed the agent to converge in complex scenarios.

### D. Action Encoding & Combo Sets (Multi-Hot Architecture)
*   **The Challenge:** ViZDoom exposes independent buttons (move, turn, shoot). A standard `OneHot` wrapper prevents "Run & Gun" behavior (e.g., you can't Move Forward + Shoot in the same tick).
*   **The Solution:** We implemented a **Multi-Hot Combo Wrapper**.
    *   Instead of picking 1 button, the agent picks 1 *row* from a pre-defined matrix of combos.
    *   Example Row: `[1, 0, 0, 0, 0, 0, 1]` => `MOVE_FORWARD` + `ATTACK`.
*   **Curriculum Standardization:**
    To ensure checkpoints load across all phases, we forced a **Universal 7-Button Standard** (`SHARED_ACTION_BUTTONS`) across Basic, Defend, and Deadly Corridor.
    *   `[FWD, BACK, TURN_L, TURN_R, MOVE_L, MOVE_R, ATK]`
    *   This keeps the Policy Head size constant (N_logits) even as the scenario complexity grows, preventing shape mismatch errors during Transfer Learning.

### D. Action Encoding & Combo Sets (Multi-Hot Architecture)
*   **The Challenge:** ViZDoom exposes independent buttons (move, turn, shoot). A standard `OneHot` wrapper prevents "Run & Gun" behavior (e.g., you can't Move Forward + Shoot in the same tick).
*   **The Solution:** We implemented a **Multi-Hot Combo Wrapper**.
    *   Instead of picking 1 button, the agent picks 1 *row* from a pre-defined matrix of combos.
    *   Example Row: `[1, 0, 0, 0, 0, 0, 1]` => `MOVE_FORWARD` + `ATTACK`.
*   **Curriculum Standardization:**
    To ensure checkpoints load across all phases, we forced a **Universal 7-Button Standard** (`SHARED_ACTION_BUTTONS`) across Basic, Defend, and Deadly Corridor.
    *   `[FWD, BACK, TURN_L, TURN_R, MOVE_L, MOVE_R, ATK]`
    *   This keeps the Policy Head size constant (N_logits) even as the scenario complexity grows, preventing shape mismatch errors during Transfer Learning.

---

## 5. Theoretical Framework: PPO & The Dual-Head Architecture

### A. Proximal Policy Optimization (PPO)
PPO is a policy gradient method that balances **sample efficiency** with **training stability**. Unlike "Vanilla" Policy Gradient (REINFORCE) which can suffer from destructive heavy updates, PPO constrains the update step to stay "proximal" (close) to the previous policy.

**The Core Mechanism: Clipped Surrogate Objective**
The PPO loss function is designed to prevent the new policy $\pi_\theta$ from diverging too far from the old policy $\pi_{\theta_{old}}$.
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

*   $r_t(\theta)$ is the probability ratio: $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.
*   $\hat{A}_t$ is the **Advantage** (how much better this action was than average).
*   $\epsilon$ is the clipping parameter (usually 0.1 or 0.2).

**Why it matters for Doom:**
In a chaotic environment like a Deathmatch, a single lucky frame could produce a massive reward signal (e.g., fragging 3 enemies). Without clipping, the gradient update would smash the network weights towards that specific lucky action, destroying the delicate visual filters learned so far. PPO's clipping ensures we learn *steadily* from the experience without overreacting.

### B. The Dual-Head Architecture (Actor-Critic)
Our agent is not just one network; it is two distinct "brains" sharing a single pair of eyes (the CNN Encoder).

#### 1. The Encoder (The Eyes)
*   **Input:** 4 Stacked Grayscale Frames (84x84x4).
*   **Function:** Extracts spatial features (corners, enemies, walls) into a 512-dimensional latent vector.
*   **Role:** Both the Actor and Critic rely on this shared understanding of "What am I looking at?".

#### 2. The Actor Head (The Policy / $\pi_{\theta}$)
*   **Type:** Categorical Distribution (Softmax).
*   **Output:** A set of **Logits** (unnormalized scores), one for each valid Action Combo (0 to 6).
*   **Process:**
    1.  Receives the 512-d feature vector.
    2.  Outputs 7 scores (e.g., `[FWD: 2.5, ATK: 0.1, ...`).
    3.  **Training:** Samples stochastically based on probability (exploration).
    4.  **Evaluation:** Takes the `argmax` (greedy exploitation).
*   **Role:** "The Pilot". It decides *what to do* given the current view.

#### 3. The Critic Head (The Value Function / $V(s)$)
*   **Type:** Scalar Regressor (Linear layer).
*   **Output:** A single floating-point number representing the **Expected Future Return** ($V(s)$) from the current state.
*   **Training Signal:** Minimizes the Mean Squared Error (MSE) between its prediction and the actual rewards received (plus GAE bootstrap).
*   **Role:** "The Coach". It tells the Actor how good the current situation is.
    *   If the Actor takes an action and gets a reward, the Critic compares it to its expectation.
    *   **Advantage Calculation:** $A = \text{Actual Reward} - V(s)$.
    *   If $A > 0$ (result was better than expected), the Actor is encouraged to do it again.
    *   If $A < 0$ (result was worse/disappointing), the Actor is discouraged.

### C. The Temporal Stack: Skipping vs. Stacking
*   **The User Question:** "We skip 4 frames, but stack 4 frames. Does skipping delete the frames we need?"
*   **The Answer:** No. Skipping drops the *intermediate* frames to extend our memory horizon.
    *   **Without Skip:** Stack = $[t, t-1, t-2, t-3]$. (History: 0.1 sec). Agent sees smooth motion but has no memory of the past.
    *   **With Skip (4):** Stack = $[t, t-4, t-8, t-12]$. (History: 0.4 sec).
*   **Visualizing the Timeline:**
    ```text
    Tick 1: Rendered -> Stored in Stack[0]
    Tick 2: Skipped (Physics runs, but Agent ignores it)
    Tick 3: Skipped (Physics runs, but Agent ignores it)
    Tick 4: Skipped (Physics runs, but Agent ignores it)
    Tick 5: Rendered -> Stored in Stack[1]
    ...
    ```
*   **Why this is good:** By ignoring the intermediate frames, the agent's memory spans a much longer period of time, allowing it to understand complex movements (like an enemy circling around) without needing a massive neural network.

---

## 6. Usage Guide

### Prerequisites
Values are managed via `uv`.
```bash
uv sync
```

### Training Command (Standard Template)
```bash
uv run python doom_ppo_deadly_corridor.py \
    --scenario-cfg configs/deathmatch_simple.cfg \
    --exp-name my_experiment \
    --use-lstm --lstm-hidden-size 512 \
    --cuda --track
```

### Evaluation Command (Standard Template)
```bash
uv run python eval_doom_agent.py \
    --checkpoint checkpoints/my_checkpoint.pt \
    --scenario-cfg configs/deathmatch_simple.cfg \
    --use-lstm --render --cuda
```

---

## 6. Evaluation Cheat Sheet (Final Verified Commands)
*Use these commands to demonstrate the project's progression to an audience.*

### Phase 1: Basic (The Foundation)
**Goal:** Verify the agent can move and shoot.
```bash
uv run python eval_doom_agent.py \
    --checkpoint checkpoints/basic_phase1_clean_basic_lstm_2025-12-04_14-10-21_seed42_step1499136.pt \
    --scenario-name basic \
    --use-lstm --lstm-hidden-size 512 \
    --episodes 5 --render --cuda \
    --sleep-per-step 0.05 --damage-reward 1.0
```

### Phase 2: Defend the Center
**Goal:** Check 360-degree aiming and turret behavior.
```bash
uv run python eval_doom_agent.py \
    --checkpoint checkpoints/defend_center_phase2_defend_the_center_lstm_2025-12-04_16-21-49_seed42_step1999872.pt \
    --scenario-name defend_the_center \
    --use-lstm --lstm-hidden-size 512 \
    --episodes 5 --render --cuda \
    --sleep-per-step 0.05
```

### Phase 3: Deadly Corridor
**Goal:** Survive the corridor and reach the vest.
```bash
uv run python eval_doom_agent.py \
    --checkpoint checkpoints/deadly_corridor_phase3_deadly_corridor_lstm_2025-12-04_23-06-15_seed42_step4999168.pt \
    --scenario-name deadly_corridor \
    --use-lstm --lstm-hidden-size 512 \
    --episodes 5 --render --cuda \
    --sleep-per-step 0.05
```

### Phase 4: Health Gathering Supreme
**Goal:** Verify navigation and healing logic (no shooting).
```bash
uv run python eval_doom_agent.py \
    --checkpoint checkpoints/health_gathering_phase4_health_gathering_supreme_lstm_2025-12-05_11-54-11_seed42_step999424.pt \
    --scenario-cfg configs/health_gathering_supreme.cfg \
    --use-lstm --lstm-hidden-size 512 \
    --episodes 3 --render --cuda \
    --sleep-per-step 0.05
```

### Phase 5.12: "The Scavenger" (Ammo Aware)
**Goal:** Show the agent picking up different ammo types and managing panic.
```bash
uv run python eval_doom_agent.py \
    --checkpoint checkpoints/deathmatch_phase5_scavenger_fixed_deathmatch_simple_lstm_2025-12-09_00-08-11_seed42_step9999360.pt \
    --scenario-cfg configs/deathmatch_simple.cfg \
    --use-lstm --lstm-hidden-size 512 \
    --ammo-reward 0.5 \
    --kill-reward 5.0 \
    --episodes 3 --render --cuda \
    --sleep-per-step 0.05
```

### Phase 5.13: "The Matador" (Current Best)
**Goal:** Demonstrate evasion. The agent should NOT Rage-face-tank but instead strafe/dodge.
```bash
uv run python eval_doom_agent.py \
    --checkpoint checkpoints/deathmatch_phase5_matador_deathmatch_simple_lstm_2025-12-09_12-31-04_seed42_step9999360.pt \
    --scenario-cfg configs/deathmatch_simple.cfg \
    --use-lstm --lstm-hidden-size 512 \
    --health-penalty 0.5 \
    --pain-rage-multiplier 1.0 \
    --episodes 5 --render --cuda \
    --sleep-per-step 0.05
```