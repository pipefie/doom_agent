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
- Override shaping if needed (e.g., weaken ammo penalty): add `--ammo-penalty 0.005`.
- Evaluate a checkpoint with matching scenario/shaping:
  - `uv run python eval_doom_agent.py --checkpoint checkpoints/<ckpt>.pt --scenario-name deadly_corridor --episodes 5 --render --deterministic --cuda`
  - Evaluation accepts the same shaping flags to keep reward accounting aligned with training (`--progress-scale`, `--health-penalty`, etc.).

## Logging and Checkpoints
- TensorBoard logs: `runs/<exp_name>_<scenario>_<timestamp>_seed<seed>` when `--track` is set. Inspect with `tensorboard --logdir runs`.
- Checkpoints: saved under `checkpoints/` every `--save-interval` global steps and at the end of training. Filenames include the run name and `global_step`.
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
