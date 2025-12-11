# Technical Explanation of `doom_ppo_deadly_corridor.py`

This document provides a comprehensive, line-by-line technical analysis of the `doom_ppo_deadly_corridor.py` script. The script implements a Proximal Policy Optimization (PPO) agent with a Recurrent Neural Network (LSTM) backend, specifically tailored for the highly partial-observable environment of ViZDoom. It rigorously adheres to architectural decisions designed to maximize stability and performance in the "Deadly Corridor" scenario.

## 1. Configuration and Hyperparameter Management: The Source of Truth

The script begins by establishing a rigorous configuration schema using Python's `dataclasses`. The `DoomConfig` class (lines 24-50) serves as the centralized definition for all environment parameters, specifically focusing on reward shaping coefficients.

Unlike standard Reinforcement Learning implementations that might treat rewards as opaque signals, this architecture explicitly exposes every reward component (`kill_reward`, `ammo_penalty`, `health_penalty`, etc.) as a tunable hyperparameter. This design allows for "Reward Scaling" at the source—modifying the magnitude of the raw reward signal—rather than relying on post-hoc normalization techniques like Return Normalization, which is explicitly disabled in this pipeline to preserve the critic's ability to distinguish value magnitude.

The strict separation of `SCENARIO_SHAPING_DEFAULTS` ensures that each scenario (like `deadly_corridor` or `basic`) has a deterministic set of reward coefficients that override general defaults, guaranteeing that the agent's objective function is mathematically precise for the specific task at hand.

## 2. The Environment Wrapper: `VizDoomGymnasiumEnv`

The `VizDoomGymnasiumEnv` class adapts the raw ViZDoom API into a Gymnasium-compliant environment suitable for modern CleanRL-style PPO implementations.

### Action Space Representation: The Multi-Hot Wrapper
A critical architectural decision is found in `_build_actions_by_scenario` (line 244). Standard ViZDoom wrappers often treat actions as simple discrete indices mapping to single buttons. This implementation uses a **Multi-Hot Wrapper** approach.

The `SHARED_ACTION_BUTTONS` list defines a fixed vocabulary of 7 atomic actions: `[FWD, BACK, TURN_L, TURN_R, STRAFE_L, STRAFE_R, ATK]`. The wrapper then constructs a discrete action space where each integer action index maps to a specific *combination* of these atomic buttons (a boolean vector).

For `deadly_corridor` (lines 293-309), the action space is size 12. This allows the agent to execute complex compound behaviors like "Strafe Left + Attack" (Index 8) or "Move Forward + Attack" (Index 7) as single atomic decisions. This significantly reduces the temporal horizon required to learn combat maneuvers compared to a system that requires alternating between "Move" and "Attack" frames.

### Reward Shaping Engine
The `_shape_reward` method (line 341) is the core logic that transforms the raw environment state into a dense scalar reward signal.

1.  **Delta-Based Tracking**: The method tracks the frame-to-frame difference (`delta`) of game variables like Health, Ammo, and Position. This ensures that rewards are Markovian updates based on immediate state transitions.
2.  **Universal Ammo Handling**: The logic at line 378 handles multiple ammo types simultaneously. This prevents the agent from being penalized for "losing ammo" when simply switching weapons (which acts as a variable swap in Doom). It calculates net ammo change across all slots.
3.  **Panic Penalty**: Line 370 introduces a penalty for having zero ammo in the selected weapon, actively discouraging the agent from holding empty weapons during combat.
4.  **Position Tracking**: For `deadly_corridor`, the `progress_scale` (line 394) rewards the agent for maximizing its X-coordinate (pushing down the corridor). The logic uses `max_posx` to ensure the agent is only rewarded for *new* progress, preventing "reward hacking" loops where the agent moves back and forth.

## 3. Neural Architecture: PPO + NatureCNN + LSTM

The agent is implemented in the `PPOAgent` class (line 596), combining a convolutional visual encoder with a recurrent memory core.

### Visual Encoder (`NatureCNN`)
The inputs are processed by `NatureCNN` (line 577), a standard architecture derived from the original DQN Nature paper, optimized for 84x84 inputs.
- **Layer 1**: Conv2d, 32 filters, 8x8 kernel, stride 4.
- **Layer 2**: Conv2d, 64 filters, 4x4 kernel, stride 2.
- **Layer 3**: Conv2d, 64 filters, 3x3 kernel, stride 1.
- **Flatten & Projection**: The output is flattened and projected to a 512-dimensional feature vector.

### Recurrent Core (LSTM)
Critically, the 512-dim visual features are passed to an LSTM (Long Short-Term Memory) layer (line 606) if `use_lstm` is enabled.
- **Hidden State Management**: The LSTM maintains a hidden state `(h, c)` across time steps. In the rollout loop (line 840), the hidden states are preserved and carried forward, allowing the agent to integrate information over time (essential for spotting enemies in a partially observable 3D maze).
- **Sequence Handling during Training**: During optimization (line 926), the training data—which is collected as a flat batch of transitions—must be restructured into sequences. The code does not perform full sequence reconstruction for backpropagation through time (BPTT) over the entire episode but relies on the stored hidden states at the start of each minibatch (`mb_h0`, `mb_c0` at line 939) to approximate the recurrent context.

## 4. The PPO Training Loop: Algorithm Core

The main loop implements the PPO algorithm (Proximal Policy Optimization) using Generalized Advantage Estimation (GAE).

### Rollout Phase
The code collects `args.num_steps` of interactions.
- **Observation Stack**: The buffer `obs` stores the *current* state.
- **LSTM States**: It explicitly stores `lstm_states_h` and `lstm_states_c` at every timestep (line 841). This "snapshotting" of the hidden state is vital for the PPO update phase, as it allows the optimization loop to restart the LSTM context correctly for any random minibatch start point.

### Advantage Estimation (GAE)
Lines 887-898 implement GAE.
- `delta` is calculated as the Temporal Difference (TD) error: `reward + gamma * V(next) - V(current)`.
- `advantages` are recursively calculated using `gae_lambda`.
- **Normalization (Critical)**: Per the established source of truth, **Advantage Normalization** is applied (line 918) if arguments permit (`--norm-adv`). This centers the advantages to mean 0 and std 1, stabilizing the policy gradient.
- **Return Normalization (Explicitly Absent)**: There is **no code** that normalizes `b_returns`. This is a deliberate design choice. By keeping returns unscaled, the Critic `v_loss` retains the true magnitude of the rewards (controlled by `DoomConfig`). This allows the agent to distinguish between a "high reward" episode (lots of kills) and a "low reward" episode meaningfully, rather than compressing everything to a unit Gaussian.

### PPO Update Phase
The optimization loop (lines 925-1002) iterates through the collected batch.
1.  **Policy Loss (`pg_loss`)**: It constructs the ratio of probabilities `new_prob / old_prob`. It then applies the PPO "Clip" objective: `min(ratio * adv, clamp(ratio, 1-eps, 1+eps) * adv)`. This pessimistic bound prevents the policy from updating too violently in any single step.
2.  **Value Loss (`v_loss`)**: calculated as the MSE between the Critic's predicted value and the target Return. The code includes a "clipped" version of the value loss (line 962), ensuring the Value function also doesn't drift too far from its previous state, though often the unclipped loss dominates.
3.  **Entropy Loss**: Added to the objective to encourage exploration, preventing premature convergence to a deterministic policy.
4.  **Optimization**: The final `loss` is a weighted sum of Policy, Value, and Entropy losses. Gradients are clipped (`max_grad_norm`, line 971) to prevent exploding gradients, a common issue in RNN training.

### LSTM Loop Logic (`get_action_and_value_lstm_loop`)
A specialized static method (line 1050) is used during the update phase for LSTM agents. Because the PPO update samples random minibatches from the trajectory, it cannot simply run the network forward. Instead, it takes the **saved hidden states** (`mb_h0`) corresponding to the start of that minibatch, and sequentially unrolls the LSTM for `minibatch_size` steps. This regenerates the accurate `newlogprob` and `newvalue` for the current policy version, allowing for a valid gradient update.

## 5. Part 2: Deep Dive into Helper Functions and Neural Details

### 5.1 Helper Functions Breakdown

#### `_register_shaping_variables` (Line 218)
**Function**: Dynamically registers game variables (like AMMO or HEALTH) with the ViZDoom engine based on the current configuration.
**Reason**: ViZDoom by default only exposes a subset of variables. To calculate specific rewards (shaping), we need access to specific internal stats.
- **Mechanism**: Use `self.game.add_available_game_variable(var)` to tell the engine to include this data in the state update.
- **Why it matters**: If `kill_reward` is nonzero, we *must* register `KILLCOUNT`. If `ammo_penalty` is nonzero, we *must* register `AMMO` variables. Without this, the agent would be blind to the stats it's supposed to be rewarded for.

#### `_find_var_index` (Line 239)
**Function**: Helper to locate the array index of a specific `GameVariable`.
**Reason**: `state.game_variables` returns a flat numpy array of floats. The order depends on the registration order.
- **Mechanism**: Iterates `self.available_vars` to find the matching enum and returns the integer index.
- **Why it matters**: It ensures robustness. Instead of hardcoding "index 3 is health", we look it up dynamically, preventing bugs if variable registration order changes.

#### `_build_actions_by_scenario` (Line 244)
**Function**: Constructs the Discrete Action Space for Gymnasium.
**Mechanism**:
1.  **Atomic Buttons**: The agent has 7 atomic buttons: `[FWD, BACK, TL, TR, ML, MR, ATK]`.
2.  **Multi-Hot Construction**: We define meaningful *combinations* of these buttons.
    - Example: `make_action_row(2, 6)` means setting index 2 (Turn Left) AND index 6 (Attack) to `1`.
    - Result: `[0, 0, 1, 0, 0, 0, 1]`.
3.  **Mapping to Buttons**: When `self.game.make_action(action)` is called with this array, ViZDoom presses "Turn Left" and "Attack" simultaneously in the same game tick.
**Why it matters**: It allows the agent to perform complex maneuvers (strafing while shooting) as a single atomic decision, essential for high-level Deathmatch play.

#### `_process_frame` (Line 312)
**Function**: Preprocesses the raw screen buffer from ViZDoom.
**Steps**:
1.  **Layout Correction**: Helper to handle Channel-First vs Channel-Last dimensions.
2.  **Grayscale**: Converts RGB to Grayscale if `grayscale=True`.
3.  **Resize**: Resizes to 84x84 using `cv2.resize`.
4.  **Normalize**: Divides by 255.0 to scale pixel values to `[0.0, 1.0]`.
**Why it matters**: Neural networks train faster and more stably with normalized float inputs (0-1) rather than raw integers (0-255). 84x84 is the standard DeepMind resolution.

#### `_get_obs` (Line 322)
**Function**: Returns the current observation.
**Mechanism**: Returns `self._frames.copy()`. `_frames` is a stack of the last 4 processed frames.
**Why it matters**: Frame Stacking (4 frames) provides temporal context (velocity, acceleration) to the Convolutional Network without needing Recurrence for local motion perception.

#### `_update_game_vars` (Line 325)
**Function**: Caches the current game state variables into `self.prev_*` variables.
**Why it matters**: To calculate "Shaping Reward", we need the *derivative* (change) of the state. `delta_health = current_health - prev_health`. This function updates the baseline for the *next* step's comparison.

#### `_shape_reward` (Line 341) - DEEP DIVE
**Function**: The "Brain" of the reward system. Calculates the scalar reward for the current timestep.
1.  **Base Condition**: Takes `base_reward`. In ViZDoom, this is usually 0 unless we pick up items, but we largely ignore native rewards to control shaping fully.
2.  **Calculations**:
    - **Kill Reward**: `(current_kills - prev_kills) * constant`. Immediate positive feedback for eliminating loss.
    - **Ammo Handling (Complex)**: Iterates effectively over ALL ammo slots. If `ammo_total` went up, `+reward`. If down, `-penalty`. This generalizes to all weapons.
    - **Health**: Penalize loss (`health_penalty`), reward gain (`health_delta_scale`).
    - **Progress (Corridor)**: `p - max_pos_x`. Only reward *new* exploration forward.
3.  **Critcal Rule**: It modifies `shaped` incrementally `shaped += ...`.
**Why it matters**: This function defines the *optimization landscape*. By disabling native return normalization and using these precise coefficients, we explicitly tell the agent: "A kill (5.0) is worth 500x more than moving forward 1 unit (0.01)". This hierarchy is lost if we normalize rewards arbitrarily.

### 5.2 Neural Network Details

#### `layer_init` (Line 572)
```python
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer
```
- **Orthogonal Initialization**: Initializes weights such that the correlation between rows is zero (orthogonal matrix).
    - **Why**: Essential for Deep Networks and RNNs. It preserves the magnitude of the equivalent gradient during backpropagation (eigenvalues ~ 1), preventing Vanishing or Exploding gradients at the start of training. `std=sqrt(2)` is the specific gain recommended for ReLU activations to maintain variance.
- **Constant Bias (0.0)**: Initializes biases to 0.
    - **Why**: Ensures neurons don't fire arbitrarily at initialization; they start neutral and learn to activate.

#### LSTM State Masking (`h` and `c`) (Line 624)
```python
h = h * (1.0 - done)
c = c * (1.0 - done)
```
- **Context**: `h` is the Hidden State (Short-term memory), `c` is the Cell State (Long-term memory highway).
- **The Operation**:
    - `done` is a boolean tensor (1.0 if episode ended, 0.0 if ongoing).
    - `(1.0 - done)` creates a mask: `0.0` for finished episodes, `1.0` for continuing ones.
- **Why we do this**:
    - In a Vectorized Environment, we run $N$ parallel episodes. When Episode 1 in Slot 1 finishes, Ep 2 immediately starts in Slot 1 *without* resetting the batch index.
    - If we didn't mask, the LSTM would carry the memory of "dying in the corridor" (Episode 1) into the "start of the new run" (Episode 2). This would horribly confuse the agent ("Why am I scared? I just started").
    - **The Mask zeroes out the memory** instantly when an episode terminates, forcing the LSTM to start fresh for the new episode.

## 6. Part 3: Deep Dive into PPO Implementation

This section details exactly **where** and **how** the PPO (Proximal Policy Optimization) logic is implemented, identifying the Actor, Critic, and Core Algorithm.

### 6.1 The Architecture: Actor and Critic

In this implementation, the **Actor** and **Critic** share a common visual "backbone" (CNN + LSTM) but have separate "heads" (final layers). This is a **Shared Parameter Architecture**.

#### The Backbone (Shared)
- **Defined**: In `PPOAgent.__init__` (Line 603, 606).
- **Components**: `self.features` (NatureCNN) and `self.lstm` (Core Memory).
- **Function**: Converts raw pixels ($84 \times 84 \times 4$) into a dense 512-dimensional hidden state vector ($h_{t}$). This vector represents the agent's complete understanding of the current reality.

#### The Heads (Separate)
1.  **The Actor (`self.actor`)** - Line 608
    - **Code**: `self.actor = layer_init(nn.Linear(core_output_dim, action_space.n), std=0.01)`
    - **Purpose**: "The Decider". It takes the hidden state $h_t$ and outputs **Logits** for every possible action (Size 12).
    - **Why Low Std (0.01)?**: We initialize weights to be tiny so that initial probabilities are uniform (random). If the actor started with strong preferences, it would stop exploring immediately.
    
2.  **The Critic (`self.critic`)** - Line 609
    - **Code**: `self.critic = layer_init(nn.Linear(core_output_dim, 1), std=1.0)`
    - **Purpose**: "The Evaluator". It takes the *exact same* hidden state $h_t$ and outputs a **Scalar Value** ($V(s)$). This predicts "How much total discounted reward do I expect to get from here until the end of the episode?".
    - **Why High Std (1.0)?**: The critic needs to output potentially large values (Returns like 20.0 or 50.0). A standard initialization enables it to reach these magnitudes faster.

### 6.2 The "Core": Forward Pass (`get_action_and_value`)

The method `get_action_and_value` (Line 616) is the **Interface** where PPO happens during interaction.

1.  **Forward Pass**: `features = self.features(x)` -> `lstm_out = self.lstm(features)`.
2.  **Actor Output**: `logits = self.actor(features)`.
3.  **Critic Output**: `value = self.critic(features)`.
4.  **Sampling (Stochasticity)**:
    - We interpret `logits` as a `Categorical` distribution (Line 630).
    - We sample `action = dist.sample()`. This is why the agent explores; it doesn't just pick the max logic, it rolls the dice based on probability.
5.  **Returns**: `action` (what to do), `log_prob` (how confident we were), `entropy` (how random we were), `value` (our prediction).

### 6.3 Usage: The Training Loop

The implementation follows the standard **Rollout -> Bootstrap -> Update** cycle.

#### Step 1: Data Collection (Rollout Phase)
For `num_steps` (128 steps), the agent interacts with the environment (Lines 833-876).
- Crucially, we store **Obs, Actions, LogProbs, Rewards, Dones, and Values** in massive arrays `(num_steps, num_envs)`.
- This creates the "Dataset" for the next PPO update. Unlike Q-Learning, PPO throws this dataset away after every update (On-Policy).

#### Step 2: GAE (Generalized Advantage Estimation)
Before updating, we need to know **how good** each action actually was compared to expectations. This is the **Advantage** ($A_t$).
- **Calculated at Lines 887-898**.
- **The Formula**: $A_t = \delta_t + (\gamma \lambda) A_{t+1}$
- **Meaning**: If $A_t > 0$, the action was *better* than expected. If $A_t < 0$, it was worse.
- **Why GAE?**: It balances Bias (using V(s)) and Variance (using raw rewards) with the hyperparameter $\lambda$ (`gae_lambda`).

#### Step 3: PPO Optimization (The Update)
We loop `update_epochs` (4) times over the data (Lines 925-1002).

1.  **Recalculation**: We run the *current* model on the *old* data to get `newlogprob` and `newvalue` (Line 942/978).
2.  **Probability Ratio ($r_t(\theta)$)**:
    ```python
    logratio = newlogprob - old_logprob
    ratio = logratio.exp()
    ```
    This measures "How much more likely is this action now compared to when we collected the data?".

3.  **Policy Loss (Clipped Surrogate Objective)** - Lines 956-958:
    ```python
    pg_loss1 = -advantage * ratio
    pg_loss2 = -advantage * clamp(ratio, 1-eps, 1+eps)
    pg_loss = max(pg_loss1, pg_loss2)
    ```
    - **The Core Idea**: We want to increase probability of good actions (Pos Advantage). **BUT**, if `ratio` changes too much (e.g., action becomes 10x more likely), we **Clip** the gradient.
    - **Why?**: This prevents "Policy Collapse". If we update too hard, we might ruin the policy. The Clip forces "Proximal" (small) updates.

4.  **Value Loss** - Lines 961-964:
    We train the Critic to minimize `(predicted_value - real_return)^2`.
    - We *also* clip the value loss to prevent the Critic from jumping too wildly in one update.

5.  **Entropy Bonus** - Line 966:
    `loss -= ent_coef * entropy`
    - We subtract entropy from the loss (maximizing entropy). This forces the agent to keep its options open and prevents it from committing to a single strategy too early (premature convergence).
