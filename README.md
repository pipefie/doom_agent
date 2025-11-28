# Doom PPO Curriculum Agent

## Project Overview
- CleanRL-style PPO agent for ViZDoom with a single training script (`doom_ppo_deadly_corridor.py`) and a dedicated evaluation script (`eval_doom_agent.py`).
- Curriculum focus: start on **Basic**, progress to **Defend the Center**, and finish on **Deadly Corridor**.
- Uses Gymnasium’s API with vectorized environments to keep throughput high while keeping the code path simple.
- Reward shaping is scenario-aware to avoid the Deadly Corridor “suicide quickly” failure mode noted in Doom RL research.

## Environment Setup (ViZDoom + Gymnasium + uv)
- Dependencies are managed via `uv` (`pyproject.toml` + `uv.lock`). Run commands with `uv run python ...` to ensure the right environment.
- ViZDoom configs and WADs live under `configs/` (e.g., `configs/basic.cfg`, `configs/defend_the_center.cfg`, `configs/deadly_corridor.cfg`).
- The wrapper `VizDoomGymnasiumEnv` normalizes grayscale frames to `[0, 1]`, stacks 4 frames of size 84×84, and exposes a discrete one-hot action space aligned with the available buttons in each scenario.

## Neural Network Architecture
- Feature extractor: NatureCNN
  - Conv1: 32 filters, 8×8 kernel, stride 4
  - Conv2: 64 filters, 4×4 kernel, stride 2
  - Conv3: 64 filters, 3×3 kernel, stride 1
  - Flatten → Linear to 512 units with ReLU
- Optional memory: single-layer LSTM after the CNN (`--use-lstm`, hidden size `--lstm-hidden-size`, default 512). Hidden state is masked by the done flags each step so memories reset when an env ends. This helps with partial observability (e.g., peeking corners in Deadly Corridor).
- Heads:
  - Actor: linear → categorical policy over discrete actions (orthogonal init, std=0.01)
  - Critic: linear → scalar value (orthogonal init, std=1.0)

## PPO Algorithm Details
- Vectorized rollout with `gym.vector.SyncVectorEnv`; batch size = `num_envs * num_steps`.
- Generalized Advantage Estimation (GAE) with `gamma` and `gae_lambda`.
- Clipped surrogate objective (`clip_coef`), entropy bonus (`ent_coef`), and value loss with clipping (`vf_coef`).
- Optional learning rate annealing to 0 over training (`--anneal-lr`).
- Gradient clipping via `nn.utils.clip_grad_norm_` to stabilize updates.
- Optional KL early-stop (`--target-kl`) and advantage normalization (`--norm-adv`).

## Reward Shaping
### Basic + Defend the Center
- Mild shaping only: reward per kill (`--kill-reward`, default 5.0) and ammo usage penalty (`--ammo-penalty`, default 0.01).
- Progress, health, and death shaping are off by default (set to 0.0).

### Deadly Corridor (anti-suicide shaping)
- Kill reward and ammo penalty remain active, with stronger defaults for this scenario.
- Progress shaping: `POSITION_X` delta is rewarded via `--progress-scale` (default 0.05) so forward motion is valuable even before the sparse terminal reward.
- Health shaping: `--health-penalty` (default 0.05) nudges the agent away from reckless damage intake.
- Death shaping: `--death-penalty` (default 5.0) lightly penalizes premature deaths so that the terminal reward still dominates.
- These signals counter the “suicide quickly” behavior by valuing survival and forward progress while keeping the end-of-corridor reward the primary objective.

## Curriculum Learning Strategy
- Choose scenarios by name (`--scenario-name basic|defend_the_center|deadly_corridor`) or by config path (`--scenario-cfg ...`).
- Warm-start the next stage with `--load-checkpoint` to reuse the policy (and optimizer state) learned on the previous scenario.
- Suggested manual sequence (timesteps are examples; adjust as needed):
  - Basic: `uv run python doom_ppo_deadly_corridor.py --scenario-name basic --total-timesteps 200000 --track`
  - Defend the Center (warm start): `uv run python doom_ppo_deadly_corridor.py --scenario-name defend_the_center --load-checkpoint <basic_ckpt> --total-timesteps 400000 --track`
  - Deadly Corridor (warm start with richer shaping): `uv run python doom_ppo_deadly_corridor.py --scenario-name deadly_corridor --load-checkpoint <defend_ckpt> --total-timesteps 800000 --track`
- `total-timesteps` controls how many steps this run will add; `global_step` continues from the checkpoint so logs remain contiguous.

## Training & Evaluation Usage
- Train a scenario (example with CUDA and TensorBoard):
  - `uv run python doom_ppo_deadly_corridor.py --scenario-name basic --total-timesteps 200000 --num-envs 8 --num-steps 128 --cuda --track`
- Enable recurrent policy (helps Deadly Corridor): add `--use-lstm --lstm-hidden-size 512` to the command. When using LSTM, `num_envs` must be divisible by `num_minibatches` (recurrent batching is by envs, not flattened steps).
- Override shaping if needed (e.g., weaken ammo penalty): add `--ammo-penalty 0.005`.
- Evaluate a checkpoint with matching scenario/shaping:
  - `uv run python eval_doom_agent.py --checkpoint checkpoints/<ckpt>.pt --scenario-name deadly_corridor --episodes 5 --render --deterministic --cuda`
  - Evaluation accepts the same shaping flags to keep reward accounting aligned with training (`--progress-scale`, `--health-penalty`, etc.).
  - If the checkpoint was trained with LSTM, pass `--use-lstm --lstm-hidden-size 512` to evaluation so the architecture matches.

## Logging and Checkpoints
- TensorBoard logs: `runs/<exp_name>_<scenario>_<arch>_<timestamp>_seed<seed>` where `<arch>` is `ff` or `lstm`, making it easy to compare recurrent vs feedforward runs. Inspect with `tensorboard --logdir runs`.
- Checkpoints: saved under `checkpoints/` every `--save-interval` global steps and at the end of training. Filenames include the run name and `global_step`, and the run name carries the `ff`/`lstm` tag.
- `global_step` counts environment steps across all vectorized envs and continues from any loaded checkpoint so you can see uninterrupted progress across curriculum stages.

## Shared Action Set (Checkpoint Compatibility)
- Problem: scenarios expose different buttons (Basic: 3 actions; Defend: 3; Deadly Corridor: 7). Checkpoints trained on fewer actions fail to load into wider action heads.
- Solution: the training/eval scripts now default to a unified 7-action list (forward/backward, turn left/right, strafe left/right, attack) via `--use-shared-actions` (on by default). This keeps the policy head shape fixed across all scenarios, so you can warm-start through the curriculum and evaluate any checkpoint anywhere.
- You can disable with `--no-use-shared-actions`, but then checkpoints will only load in matching scenarios.

## Curriculum Progress & Early Results
- Basic (500k steps, pre-shared-actions): produced a stable policy suitable for warm-starting Defend. (Action head size: 3.)
- Defend the Center (warm-started from Basic, 500k steps; head size 3):
  - Deterministic eval (5 eps): shaped reward ≈ 52–71 over ~164–226 steps → ~10–14 kills/episode given +5 kill reward and light ammo penalty. Indicates stable defense behavior with modest variance.
  - Value head improved from negative explained variance early to ~0.6–0.7, and entropy dropped to ~0.25–0.30, showing convergence toward a focused policy.
- Next: to continue into Deadly Corridor without shape mismatches, retrain Basic and Defend with the shared action set (default now on), then train Deadly Corridor from the Defend checkpoint. This aligns all checkpoints to the 7-action head.

## Full Script Walkthrough: `doom_ppo_deadly_corridor.py`

This section is a line-by-line style walkthrough so you can understand how every piece fits together. Follow along in the file from top to bottom.

1) Imports and utilities
- Standard libs: argparse (CLI), os/time/datetime, dataclasses (config handling), typing.Tuple for type hints.
- Core deps: OpenCV (`cv2`) for frame processing; Gymnasium as the environment API; NumPy for array ops; PyTorch (`torch`, `nn`, `optim`, `Categorical`) for the model/optimizer; TensorBoard `SummaryWriter` for logging; `vizdoom` for the underlying Doom engine.

2) Config dataclass and scenario defaults
- `DoomConfig` holds environment knobs: scenario path/name, frame preprocessing (84x84 grayscale, stack=4), action sharing, and reward-shaping hyperparameters (kill reward, ammo penalty, progress reward, health penalty, death penalty).
- `SCENARIO_CFG_MAP` maps friendly scenario names to `.cfg` files.
- `SCENARIO_SHAPING_DEFAULTS` provides per-scenario shaping. Deadly Corridor gets stronger shaping (kill=10, ammo penalty=0.02, progress=0.05, health penalty=0.05, death penalty=5) to counter the suicide trap.
- `SHARED_ACTION_BUTTONS` is the 7-button superset (move/turn/strafe + attack) used to keep the policy head compatible across Basic/Defend/Deadly.

3) Environment wrapper: `VizDoomGymnasiumEnv`
- Subclasses `gym.Env` and enforces Gymnasium API (`terminated`, `truncated`).
- Constructor: loads the ViZDoom config, optionally overrides available buttons to the shared list, registers required game variables for shaping (kills, ammo, position, health), and initializes the game.
- Action space: discrete one-hot over available buttons; uses an identity matrix as action list.
- Observation space: Box in [0,1] with shape (frame_stack, 84, 84); internal buffer `_frames` holds the stack.
- `_register_shaping_variables`: adds ViZDoom game variables before `init()` so they are tracked in `get_state()`. Only requests variables actually needed given shaping settings.
- `_find_var_index`: helper to locate variable indices in the game’s available variables array.
- `_process_frame`: handles layout differences (CHW vs HWC), converts to grayscale, resizes to 84x84, normalizes to [0,1]. This fixes the channel-order mismatch between ViZDoom/OpenCV (HWC) and PyTorch (CHW).
- `_get_obs`: returns a copy of the stacked frames buffer.
- `_update_game_vars`: caches killcount, ammo, position, and health from the latest state for delta-based shaping.
- `_shape_reward`: applies shaping on top of the base game reward:
  - +`kill_reward` per delta kill count
  - -`ammo_penalty` per ammo spent
  - +`progress_scale` * delta POSITION_X (Deadly Corridor progress)
  - -`health_penalty` per health lost
  - optional `death_penalty` on termination if no state is available (avoids suicide optima)
- `reset`: seeds if provided, starts a new episode, processes the first frame, fills the stack with it, and initializes shaping trackers.
- `step`: maps discrete action to one-hot Doom buttons, steps with frame skip, checks termination, processes next frame into the stack, shapes the reward, and returns `(obs, reward, terminated, truncated, info)`.
- `render`: noop because ViZDoom handles its own window when `render_mode="human"`.
- `close`: closes the ViZDoom game instance.

4) Env factory helpers
- `make_vizdoom_env` returns a thunk to create a single wrapped env, optionally rendering the first env. Seeds each env with `seed + idx` and wraps with `RecordEpisodeStatistics` for episodic returns/lengths.
- `make_envs` builds a `SyncVectorEnv` of these thunks for vectorized rollouts.

5) Model architecture
- `layer_init`: orthogonal init with configurable std and bias; stabilizes PPO by controlling initial scales.
- `NatureCNN`: three conv layers (32@8x8/4, 64@4x4/2, 64@3x3/1) + flatten, then linear to 512 with ReLU. A dummy forward computes the flattened size dynamically.
- `PPOAgent`: holds the shared CNN body, optional LSTM, plus two heads:
  - Optional LSTM (`--use-lstm`): single-layer, hidden size configurable (`--lstm-hidden-size`, default 512). The LSTM input is the 512-d CNN features; its hidden state is masked by the `done` flags so memories reset per env episode. This improves partial observability (e.g., Deadly Corridor corners/projectiles).
  - Actor head: linear to `action_space.n` logits (std=0.01).
  - Critic head: linear to scalar value (std=1.0).
- `get_action_and_value` runs features (and LSTM if enabled), samples or evaluates provided actions via `Categorical`, and returns (action, logprob, entropy, value, new_lstm_state).

6) CLI parsing (`parse_args`)
- Env settings: scenario cfg/name, checkpoint loading, env count, rollout steps, total timesteps, render flag, seed, shared actions toggle.
- Reward shaping args (override defaults if passed): kill, ammo, progress, health, death.
- PPO hyperparams: LR (2.5e-4), optional anneal, gamma, GAE lambda, minibatches/epochs, advantage norm, clip coef, entropy/value coefs, max grad norm, target KL.
- Logging/checkpoints: TensorBoard toggle/dir, exp name, checkpoint save interval/dir.
- System: CUDA flag, deterministic cudnn toggle.

7) Main training loop
- Resolve scenario name → cfg path (via map) and derive defaults. Apply shaping defaults unless CLI overrides.
- Build `DoomConfig` with resolved shaping and shared-actions setting. Print scenario and shaping for visibility.
- Basic asserts, batch/minibatch sizing.
- Seeding: NumPy/torch; device selection respects `--cuda`. Optional cudnn deterministic mode.
- Create vectorized envs (`SyncVectorEnv`). Get single observation/action spaces for model construction.
- Instantiate `PPOAgent` and Adam optimizer (eps=1e-5).
- If `--use-lstm` is on: initialize per-env LSTM states, store them through the rollout, and batch updates by environments (not flattened steps). LSTM states are masked by `done` so episodes reset memory cleanly.
- Optional warm start: load checkpoint model/optimizer/global_step for curriculum continuation.
- Logging setup: `run_name` includes scenario and timestamp; TensorBoard `SummaryWriter` if `--track`; ensure checkpoint dir exists.
- Rollout storage buffers: obs, actions, logprobs, rewards, dones, values sized (num_steps, num_envs, ...).
- Reset envs to get initial `next_obs`; `next_done` zeros.
- Compute `num_updates = total_timesteps // batch_size`; print run banner with derived sizes.

Rollout collection (per update)
- Optional LR anneal: linear decay over updates; otherwise fixed LR. Track `lr_now` for logging.
- For each step in rollout:
  - Save obs/done flags.
  - Convert obs to torch on device; get actions, logprobs, values from agent (no grad).
  - Step envs with the sampled actions; receive reward, terminated, truncated; combine to `done`.
  - Store rewards and done flags; update `next_obs`/`next_done`.
  - If `RecordEpisodeStatistics` produces `final_info`, log episodic return/length to TensorBoard.
- After rollout: bootstrap value on `next_obs` for GAE.

GAE and returns
- Compute advantages backwards with GAE using gamma and lambda, handling terminal flags (`next_nonterminal`).
- Returns = advantages + values.

Batch flattening and preprocessing
- Flatten buffers to shape (batch_size, …) and convert to torch tensors on device.
- Optional advantage normalization (`--norm-adv`).

PPO update
- Shuffle indices each epoch; iterate minibatches.
- For each minibatch:
  - Forward pass to get new logprobs, entropy, values.
  - Compute probability ratio, estimate KL, and track clip fractions.
  - Policy loss: clipped surrogate max of unclipped vs clipped.
  - Value loss: clipped vs unclipped MSE, take max, scaled by 0.5.
  - Entropy bonus: encourages exploration.
  - Total loss = policy loss - ent_coef * entropy + vf_coef * value loss.
  - Backprop, zero grads, apply grad clipping (`clip_grad_norm_`) to prevent exploding gradients, optimizer step.
  - Early-stop inner loop if `target_kl` exceeded.

Logging per update
- Explained variance is computed as a sanity check for value function fit.
- If TensorBoard enabled: log LR, clip fraction, losses (value/policy/entropy), approx KL, explained variance.
- Every 10 updates (or final): print FPS and latest loss/entropy/KL/EV snapshot.

Checkpointing
- Save model/optimizer/global_step/args every `save_interval` steps or at the end. File name includes run name and global step.

Teardown
- Close envs and TensorBoard writer, print total training time and final step counts.

Execution entry
- `if __name__ == "__main__": main()` so running `python doom_ppo_deadly_corridor.py ...` starts training with the above pipeline.
