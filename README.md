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
- Episodic stats in TensorBoard (`charts/episodic_return`, `charts/episodic_length`) come from `RecordEpisodeStatistics`. They appear when an env terminates/truncates; we log both `infos["final_info"]` and a direct `infos["episode"]` fallback to cover different Gymnasium versions.

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
- Experimental shaping adjustments (Deadly Corridor):
  - High progress_scale (0.02) + moderate kill reward (15) led to a rushing, enemy-ignoring policy.
  - Raising kill reward (25) and lowering progress (0.005) reduced rushing but still produced short, brittle episodes.
  - Pushing further toward combat/survival (kill 30–50, death 25–50) while zeroing progress and easing ammo penalty increases the incentive to clear threats.
- Increasing entropy (e.g., 0.02–0.05) and lowering frame_skip (e.g., 2) are used to break deterministic “rush one flank” habits and improve control.
- Always match eval shaping to training; evaluating with default shaping can give misleading returns and behavior.
- Added living_penalty + kill_grace_steps: a small per-step penalty when no kill has occurred recently, reset on each kill, to discourage “run forward without clearing threats.” Use carefully; if too high it can reintroduce rushing.
- Added forward_penalty gated on kill_grace_steps: if `steps_since_kill > kill_grace_steps` and POSITION_X increases, subtract `forward_penalty * delta_x` to discourage advancing while ignoring enemies.
- Despite these tweaks, multiple runs on skill 5 still converge to deterministic rushing (short episodes, minimal kills). Next mitigations: lower `doom_skill` temporarily to teach clearing both sides, run with high entropy (e.g., 0.1–0.15) and frame_skip=1, then fine-tune back at full difficulty with the same survival-heavy shaping. Consider simplifying the action set or using a short high-entropy schedule for Deadly to avoid early collapse.
- Additional shaping/struggles log:
  - Added `damage_reward` (reward per DAMAGECOUNT) and optional `health_delta_scale` (symmetric reward/penalty for health changes) to encourage hits/health preservation; still seeing rush/die at skill 5.
  - Living/forward penalties kept tiny (or 0) to avoid incentivizing suicide; forward_penalty uses POSITION_X delta gated on `steps_since_kill`.
  - Action sets per scenario when `--use-combo-actions` is on: Basic uses a small set (idle, forward, turn L/R, attack, forward+attack, turn+attack) to reduce motion noise; Defend uses a mid-size set (adds strafe+attack); Deadly uses the full 13-action set (adds strafe, backward + attack). Head size stays consistent within a run; loading older 7-action checkpoints drops the actor head and skips optimizer state to avoid shape errors.
  - Entropy warm-up flags (`--ent-coef-warm`, `--ent-warm-steps`, `--ent-coef-final`) added; high entropy early can prevent deterministic collapse but has not fully fixed the rush/die failure at skill 5.
  - Basic training with the small combo set and low entropy: agent still tends to wander/shoot randomly; higher kill reward (e.g., 10) and lower entropy (e.g., 0.002) are being tried to force purposeful shooting; convergence remains slow.

### Deadly Corridor: Detailed Walkthrough of Current Approach
- Problem observed: At skill 5 the agent repeatedly converges to “rush forward and die” with very short episodes. Even with high kill/death rewards and zero progress reward, entropy collapses early and PPO updates become tiny, so the policy stays deterministic. One-side bias (looks left/right only) persists, and the agent often ignores enemies after the first pair.
- Action set change: For Deadly Corridor only, we replaced one-hot actions with a curated combo set (idle; forward/turn/strafe; attack; forward+attack; strafe+attack; turn+attack; backward+attack). This removes pure “run forward” actions and forces turning/shooting combinations.
- Reward shaping now available:
  - Kill/death: high (e.g., kill 80, death 100).
  - Progress: set to 0 to eliminate forward lure.
  - Health penalties: `health_penalty` for loss; optional symmetric `health_delta_scale` to reward medkit gains/penalize loss.
  - Damage reward: per point of DAMAGECOUNT to reward hits even before kills.
  - Living penalty + kill_grace_steps: small per-step cost when no kill has happened recently (use tiny values or 0 to avoid incentivizing suicide by rushing).
  - Forward penalty: gated on `steps_since_kill > kill_grace_steps`, subtracts `forward_penalty * delta_x` to discourage advancing without clearing threats.
- Entropy schedule: Added warm start (`--ent-coef-warm`, `--ent-warm-steps`, `--ent-coef-final`) so Deadly can run with high entropy early (e.g., 0.2 for 200k steps) and decay to target (e.g., 0.05) to avoid early deterministic collapse.
- Frame skip: use `frame_skip=1` for finer control/aiming in Deadly; frame_skip >1 made rushing more attractive.
- Checkpoint loading: When action-space changes (7 → 13 actions), checkpoints are loaded with `strict=False` to reuse shared layers and reinit the new head. Eval does the same.
- What hasn’t solved it yet at skill 5: even with the above, runs often stay short; more kills on the first pair but still stuck. If keeping skill 5, combine the combo action set + high-entropy warm phase + survival-heavy shaping and keep living/forward penalties very small.
- If all else fails: temporarily lower `doom_skill` in `deadly_corridor.cfg` to teach clearing both sides, then fine-tune back at skill 5 with the same flags; or further simplify the action set and run a shorter high-entropy schedule.

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

## Recurrent PPO Notes (LSTM)
- LSTM is optional (`--use-lstm`) and sits after the CNN encoder. Hidden size is configurable (`--lstm-hidden-size`, default 512).
- Hidden states are masked with the `done` flags each step so episodes reset memory cleanly per env.
- Rollout storage keeps LSTM states as (time, env, layer, hidden) and reorders to (layer, env, hidden) right before feeding the PyTorch LSTM. This avoids the hidden-shape mismatch bug (expected [1, batch, hidden], got [2, batch, hidden]) seen in earlier runs.
- When LSTM is on, minibatching is by environments (`num_envs` must be divisible by `num_minibatches`); this preserves temporal order within each env during the recurrent update.

## Curriculum Progress & Early Results
- Basic (500k steps, pre-shared-actions): produced a stable policy suitable for warm-starting Defend. (Action head size: 3.)
- Defend the Center (warm-started from Basic, 500k steps; head size 3):
  - Deterministic eval (5 eps): shaped reward ≈ 52–71 over ~164–226 steps → ~10–14 kills/episode given +5 kill reward and light ammo penalty. Indicates stable defense behavior with modest variance.
  - Value head improved from negative explained variance early to ~0.6–0.7, and entropy dropped to ~0.25–0.30, showing convergence toward a focused policy.
- Next: to continue into Deadly Corridor without shape mismatches, retrain Basic and Defend with the shared action set (default now on), then train Deadly Corridor from the Defend checkpoint. This aligns all checkpoints to the 7-action head.
- Basic (500k steps, shared-actions + LSTM, run `doom_ppo_deadly_corridor_basic_lstm_2025-12-01_09-29-19_seed42`):
  - Entropy fell from ~1.9 → ~0.12, so the policy became highly deterministic by the end of training.
  - Value loss started extremely high (~1800) and decayed below ~50 near the end; explained variance was hugely negative early and hovered around ~0 to mildly negative later, so the critic is still imperfect but much improved versus the start.
  - Policy loss moved from large positive to small negative/near-zero, and KL stayed tiny throughout, indicating stable but conservative PPO updates.
  - Episodic returns should be read from TensorBoard (`charts/episodic_return`) for the final verdict on performance; the logs above show optimization stabilizing but the value fit is not yet strong. Consider a short extra run or modest vf_coef/entropy tweaks if returns plateau too low.
- Basic (500k steps, shared-actions + LSTM, vf_coef=0.4, run `doom_ppo_deadly_corridor_basic_lstm_2025-12-01_11-04-00_seed42`):
  - Entropy decreased from ~1.9 to ~0.45 and then hovered there, indicating the policy retained some stochasticity instead of collapsing fully deterministic.
  - Value loss dropped from ~1900 to ~200 but explained variance remained negative, so the critic still underfits; however policy updates stayed stable (KL ~0, policy loss small/negative).
  - Smoothed episodic return reached ~16.6 by the end (from TensorBoard), which is a reasonable improvement on Basic; if you want more, you can extend training or gently adjust entropy/vf weights, but this checkpoint is usable for warm-starting Defend with the LSTM + shared action head.
- Defend the Center (500k steps, shared-actions + LSTM warm start from the above Basic, vf_coef=0.4, run `doom_ppo_deadly_corridor_defend_the_center_lstm_2025-12-01_12-10-19_seed42`):
  - Entropy fell from ~1.7 to ~0.17, so the policy converged to a mostly deterministic defender.
  - Value loss dropped to low single digits and explained variance climbed to ~0.9+, indicating a well-fit critic on this task.
  - Policy losses stayed small/negative and KL remained tiny, suggesting stable PPO updates. This checkpoint is ready to warm-start Deadly Corridor with the LSTM + shared action head.
- Deadly Corridor (800k steps, shared-actions + LSTM warm start from Defend, vf_coef=0.4, run `doom_ppo_deadly_corridor_deadly_corridor_lstm_2025-12-01_13-30-55_seed42`):
  - Entropy collapsed to ~0.1–0.2, so the policy is very deterministic and visually rushes the corridor, often ignoring enemies.
  - Value loss remained high (hundreds–thousands) with negative explained variance, indicating a poorly fit critic and noisy returns; episodic returns fluctuated heavily.
  - Current shaping (kill=15, progress=0.02, health_penalty=0.05, death_penalty=5) likely over-rewards forward progress. Next run should rebalance toward combat: raise kill reward (e.g., 20–25), lower progress_scale (e.g., 0.005–0.01), and increase death penalty (e.g., 15–20) while keeping health penalty on. Continue from this checkpoint to test the new shaping quickly.
  - Visual inspection: policy often turns right/rushes, gets killed frequently. This aligns with the critic underfit and the progress-heavy shaping; prioritize kill/death shaping over progress to shift behavior toward clearing enemies.
- Deadly Corridor (800k steps, shared-actions + LSTM warm start, vf_coef=0.4, stronger kill/death shaping run `doom_ppo_deadly_corridor_deadly_corridor_lstm_2025-12-01_15-11-20_seed42` with kill=25, progress=0.005, ammo_penalty=0.01, health_penalty=0.05, death_penalty=15):
  - Entropy stayed low (~0.1–0.25), so policy remained highly deterministic; episodic_return smoothed around ~9.7—still weak.
  - Value loss was very large early but later EV hovered ~0.7–0.85; critic improved but returns remained modest, suggesting the policy is still brittle (e.g., simple turning/rushing, frequent deaths).
  - Next steps: further tilt toward combat/survival (e.g., even lower progress_scale, higher death_penalty, maintain kill reward), or consider modest entropy increase to escape the deterministic rush policy.
  - Visual inspection: agent kills one nearby enemy then rushes, ignoring the right side; episodes end quickly without reaching the goal.
- Deadly Corridor eval note (run `doom_ppo_deadly_corridor_deadly_corridor_lstm_2025-12-01_15-11-20_seed42_step1799168.pt`):
  - Evaluating with default shaping (kill=10, progress=0.05, death=5, ammo=0.02) produced very short episodes (~50 steps) and high shaped returns (~800) but behavior was still brittle (clears first pair, dies on second). This mismatch shows eval must mirror training shaping.
  - Re-evaluate with the training shaping flags (kill=25, progress=0.005, ammo=0.01, health_penalty=0.05, death_penalty=15) to get meaningful metrics. If behavior remains stuck, continue training with the more survival-focused shaping (kill/death higher, progress lower/zero, slight entropy bump) described above.
- Deadly Corridor (in progress): retraining from the Defend LSTM checkpoint with a stronger combat/survival tilt to break the rush-only policy:
  - Shaping: `--kill-reward 30`, `--death-penalty 25`, `--progress-scale 0.001`, `--health-penalty 0.05`, `--ammo-penalty 0.005`; entropy bumped to `--ent-coef 0.02`.
  - Rationale: de-emphasize forward progress, heavily reward kills and penalize deaths, keep health pressure, and add a bit more exploration to escape the deterministic left-side-only rush.

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

## Recent Experiments, Troubles, and Lessons (Ongoing Log)
- **Action-set curriculum:** We now use scenario-specific combo actions when `--use-combo-actions` is enabled: Basic = small set (idle, forward, turn L/R, attack, forward+attack, turn+attack) to reduce motion noise; Defend = medium (adds strafe±attack); Deadly = full 13 combos (adds strafe, backward+attack, run-and-gun variants). Loading a checkpoint into a wider head will drop the old actor head and reinit a new one; optimizer state is skipped in that case.
- **Entropy scheduling:** Added `--ent-coef-warm`, `--ent-warm-steps`, `--ent-coef-final` to keep exploration high early (especially in Deadly) and decay later. Without this, policies collapse to deterministic rush/turn behaviors. Very low entropy (e.g., 0.0005) is being tested on Basic to force aim/shoot instead of spinning.
- **Shaping knobs in play:** Besides kill/death/health/progress, we added `damage_reward` (per DAMAGECOUNT), `living_penalty` with `kill_grace_steps` (tiny or zero to avoid incentivizing suicide), `forward_penalty` gated on no recent kills (to discourage advancing while ignoring enemies), and optional `health_delta_scale` for symmetric health changes. Forward shaping is often set to 0 in Deadly to avoid rush policies.
- **Observed failures:** At skill 5, Deadly often converges to “rush forward and die,” sometimes biased to one side, even with high kill/death rewards and zero progress reward. High entropy warm phases and frame_skip=1 help but have not fully broken the pattern. Evaluations must match the training shaping; default eval shaping can mask issues with inflated returns.
- **Basic with combo actions:** Multiple runs (500k steps) still show wandering/spinning and negative episodic returns when entropy is moderate. Current mitigation is to slam entropy down (≤0.001) and raise kill/damage incentives (kill ≥10–20, damage_reward ~1.0) so the policy latches onto “face target + shoot.” If this fails, we will prune the Basic action set further (e.g., idle, turn L/R, attack, forward+attack only) to remove aimless motion.
- **Action mapping fix:** The combo action matrices are now explicitly aligned with the shared button order (MOVE_FWD, MOVE_BACK, TURN_L, TURN_R, MOVE_L, MOVE_R, ATTACK). Misalignment previously caused the “attack” intent to trigger turns, leading to spinning. The Basic/Defend/Deadly sets were rebuilt accordingly.

## Action Encoding & Combo Sets
- **Why multi-hot combos?** ViZDoom exposes independent buttons (move/turn/strafe/attack). Using a one-hot over single buttons prevents “run-and-gun” behaviors and makes the policy head shape scenario-dependent. We switched to multi-hot combo rows so one discrete index can trigger several buttons at once (e.g., forward+attack), enabling strafe-shooting and turn-and-shoot behaviors that are critical in Deadly Corridor.
- **What the vector means:** Each row in the combo matrix is a multi-hot vector over the shared buttons in fixed order `[MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT, ATTACK]`. A 1 in a position means “press that Doom button this step.” For example, `[1,0,0,0,0,0,1]` means “move forward + attack.” ViZDoom’s `make_action` consumes this boolean list directly, so the encoding is exactly the set of buttons to press simultaneously.
- **Shared button order:** We force a consistent button list via `SHARED_ACTION_BUTTONS` in `doom_ppo_deadly_corridor.py`: `[MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT, ATTACK]`. This order is applied to the game (`set_available_buttons`) so the policy head and the Doom engine agree on indices.
- **Scenario-specific combo sets (`--use-combo-actions`):**
  - Basic: small, aim-focused set (idle; forward; turn L/R; attack; forward+attack; turn±attack) to reduce wander and teach “face + shoot.”
  - Defend the Center: medium set adds strafe±attack to handle circle-strafing enemies.
  - Deadly Corridor: full 13-action run-and-gun set (forward/back/strafe/turn combinations with attack) to cover all movement/shooting needs.
  If combo actions are off, we fall back to an identity matrix over the available buttons.
- **Checkpoint compatibility:** Keeping the shared button order means the actor head size is predictable. Changing the combo set size will reinit the actor head when loading a checkpoint into a different action count, but the base CNN/LSTM weights are reused. Keeping the same set across a curriculum stage avoids head resets.
- **Config relationship:** The ViZDoom scenario `.cfg` files list available buttons, but we override them with `SHARED_ACTION_BUTTONS` to ensure consistency. Frame skip comes from the CLI (train) or the config (eval, currently fixed), so keep train/eval skip aligned to avoid control mismatch.
- **Defend warm starts:** Defend-from-Basic LSTM checkpoints with vf_coef≈0.3–0.4 and modest entropy have behaved reasonably (scores ~50–70 shaped reward per episode), so we keep using Defend as the warm start for Deadly.
- **Deadly corridor attempts:** With stronger combat shaping (kill up to 80, death up to 100, progress=0, health_penalty>0, ammo_penalty low) and the 13-action set, the agent still tends to short episodes, rushing or looking only one side. Value losses stay high and EV negative or near zero, indicating critic underfit and unstable returns. If skill 5 remains mandatory, consider (a) even higher early entropy warm (0.15–0.2 for ~200k steps), (b) zero/near-zero living/forward penalties, (c) more damage_reward and health_delta to bias toward engaging threats, and (d) a simplified action set for a short “aiming bootcamp” before re-expanding.
- **Checkpoint compatibility:** When action-space size changes (e.g., 7→13), training now drops incompatible actor params and reinitializes the head; optimizer state is reset. Evaluation does the same with `strict=False` so checkpoints remain usable, but warm-start benefits for the actor logits are lost when the head size changes.
- **Next planned tests:** Run Basic with a minimal action set and very low entropy + strong kill/damage reward to get a clean “face and shoot” prior; then Defend with the medium set; then Deadly with the full set, high-entropy warm phase, progress=0, high kill/death, modest damage_reward, and tiny/zero living/forward penalties. Keep doom_skill at 5 per current constraint; if plateau persists, temporarily lowering skill to teach clearing both sides is an option for later consideration.

## Debugging Log: The 'Stuck Shooter' Problem on Basic Scenario

This log details the step-by-step process of diagnosing and fixing a common reinforcement learning problem where the agent adopted a suboptimal, repetitive behavior instead of learning the intended task.

### 1. The Initial Problem: Suboptimal Local Minimum

- **Observation:** When training on the `basic` scenario, the agent learned to move to the side of the arena and shoot continuously at a wall. It was not actively pursuing or killing the enemy.
- **Interpretation:** The agent found a "local minimum" in its policy. This behavior, while not optimal, was "safe" and produced a consistent (though not good) outcome according to its flawed understanding of the environment. It learned that shooting was an action, but not how to connect that action to a successful outcome.

### 2. Diagnosis Part 1: Sparse Rewards

- **Investigation:** The initial training logs for the `basic` scenario showed the following reward shaping: `(kill=20.0, ammo_penalty=0.0075, ...)` and crucially, no `damage_reward`.
- **Diagnosis:** This is a **sparse reward** problem. The agent only received a large positive reward upon successfully *killing* an enemy. Kills are infrequent for an untrained agent, so there can be hundreds of actions between rewards. This makes the **credit assignment problem** nearly impossible to solve; the agent cannot determine which of its many past actions were responsible for the eventual reward.
- **Theoretical Explanation:** The goal of a reinforcement learning agent is to maximize the expected future discounted reward, $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$. If the reward $R$ is almost always zero, the agent gets no gradient signal to update its policy. By providing smaller, more frequent (i.e., "dense") rewards for intermediate steps that lead to the goal, we provide a much clearer learning signal.
- **Solution:** We introduced a **dense reward** by adding `damage_reward=1.0` to the default shaping for the `basic` scenario. This provides an immediate positive reward for the action of successfully hitting an enemy, directly solving the credit assignment problem for aiming.

### 3. Anomaly & Diagnosis Part 2: The Evaluation Mismatch

- **Observation:** After training with the new `damage_reward`, an evaluation run showed catastrophically bad performance (e.g., rewards of -300). This was contrary to the slight improvement seen during training.
- **Investigation:** We compared the log output from the training script and the evaluation script.
  - Training log: `... damage_reward=1.0 ...`
  - Evaluation log: `... damage_reward` was missing.
- **Diagnosis:** A bug was found where the evaluation script (`eval_doom_agent.py`) was not correctly loading or logging the `damage_reward` from the scenario defaults. The agent was being trained in one environment (with damage rewards) and evaluated in another (without them). Its policy was nonsensical in the evaluation environment, leading to the terrible scores.
- **Solution:** The `eval_doom_agent.py` script was patched to correctly apply and log all reward shaping parameters, ensuring the training and evaluation environments were identical.

### 4. Diagnosis Part 3: Unstable Value Function (Critic Failure)

- **Observation:** After fixing the evaluation script, a new evaluation showed a "brittle" or unstable policy. In some episodes, the agent performed well and killed the enemy quickly. In others, it reverted to the old behavior of getting stuck. The skill was learned, but not applied reliably.
- **Investigation:** We analyzed the training logs again, focusing on the `ExplainedVar` (Explained Variance) metric. Throughout the entire training run, `ExplainedVar` remained highly negative (e.g., -0.857 at the end).
- **Diagnosis:** The core problem was an unstable **value function** (the "Critic" in our Actor-Critic model).
- **Theoretical Explanation:** Explained Variance measures how well the Critic's predictions of future rewards match the actual rewards the agent received. A value of 1.0 is a perfect prediction; a negative value means the Critic's predictions are worse than simply guessing the average outcome. The PPO algorithm updates the policy (the "Actor") based on an "advantage" calculation, which is heavily dependent on the Critic's predictions (`Advantage ≈ Actual Reward - Predicted Reward`). If the Critic's predictions are garbage, the advantage calculation is also garbage, and the Actor receives a noisy, unreliable gradient. The agent literally cannot tell if its actions are leading to good or bad states.
- **The PPO Loss Function:** The total loss is a combination of policy loss, value loss, and an entropy bonus: `Loss = Loss_policy + vf_coef * Loss_value - ent_coef * Loss_entropy`. In our case, `Loss_value` was massive and incorrect. Because it's part of the total loss, its huge, noisy gradients were likely overwhelming the smaller, more useful gradients from the policy and entropy losses, destabilizing the entire learning process.

### 5. Current Solution: Stabilizing the Critic

- **Solution:** To combat the unstable Critic, we are continuing training from the last checkpoint but with a significantly reduced **value function coefficient** (`--vf-coef 0.2`).
- **Rationale:** By lowering `vf-coef`, we are reducing the weight of the `Loss_value` in the total loss function. This effectively tells the optimizer: "For now, don't trust the Critic's updates so much. Pay more attention to the Actor's updates, which are guided by the dense `damage_reward` we added." This should stabilize the shared layers of the neural network, allowing the policy to improve consistently while giving the critic more time to gradually learn a better value estimate from a more stable policy. The next training run's logs will be monitored for an increasing `ExplainedVar`, which will signal that this approach is working.

## Debugging Log: The 'Stuck Shooter' Problem on Basic Scenario

This log details the step-by-step process of diagnosing and fixing a common reinforcement learning problem where the agent adopted a suboptimal, repetitive behavior instead of learning the intended task.

### 1. The Initial Problem: Suboptimal Local Minimum

- **Observation:** When training on the `basic` scenario, the agent learned to move to the side of the arena and shoot continuously at a wall. It was not actively pursuing or killing the enemy.
- **Interpretation:** The agent found a "local minimum" in its policy. This behavior, while not optimal, was "safe" and produced a consistent (though not good) outcome according to its flawed understanding of the environment. It learned that shooting was an action, but not how to connect that action to a successful outcome.

### 2. Diagnosis Part 1: Sparse Rewards

- **Investigation:** The initial training logs for the `basic` scenario showed the following reward shaping: `(kill=20.0, ammo_penalty=0.0075, ...)` and crucially, no `damage_reward`.
- **Diagnosis:** This is a **sparse reward** problem. The agent only received a large positive reward upon successfully *killing* an enemy. Kills are infrequent for an untrained agent, so there can be hundreds of actions between rewards. This makes the **credit assignment problem** nearly impossible to solve; the agent cannot determine which of its many past actions were responsible for the eventual reward.
- **Theoretical Explanation:** The goal of a reinforcement learning agent is to maximize the expected future discounted reward, $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$. If the reward $R$ is almost always zero, the agent gets no gradient signal to update its policy. By providing smaller, more frequent (i.e., "dense") rewards for intermediate steps that lead to the goal, we provide a much clearer learning signal.
- **Solution:** We introduced a **dense reward** by adding `damage_reward=1.0` to the default shaping for the `basic` scenario. This provides an immediate positive reward for the action of successfully hitting an enemy, directly solving the credit assignment problem for aiming.

### 3. Anomaly & Diagnosis Part 2: The Evaluation Mismatch

- **Observation:** After training with the new `damage_reward`, an evaluation run showed catastrophically bad performance (e.g., rewards of -300). This was contrary to the slight improvement seen during training.
- **Investigation:** We compared the log output from the training script and the evaluation script.
  - Training log: `... damage_reward=1.0 ...`
  - Evaluation log: `... damage_reward` was missing.
- **Diagnosis:** A bug was found where the evaluation script (`eval_doom_agent.py`) was not correctly loading or logging the `damage_reward` from the scenario defaults. The agent was being trained in one environment (with damage rewards) and evaluated in another (without them). Its policy was nonsensical in the evaluation environment, leading to the terrible scores.
- **Solution:** The `eval_doom_agent.py` script was patched to correctly apply and log all reward shaping parameters, ensuring the training and evaluation environments were identical.

### 4. Diagnosis Part 3: Unstable Value Function (Critic Failure)

- **Observation:** After fixing the evaluation script, a new evaluation showed a "brittle" or unstable policy. In some episodes, the agent performed well and killed the enemy quickly. In others, it reverted to the old behavior of getting stuck. The skill was learned, but not applied reliably.
- **Investigation:** We analyzed the training logs again, focusing on the `ExplainedVar` (Explained Variance) metric. Throughout the entire training run, `ExplainedVar` remained highly negative (e.g., -0.857 at the end).
- **Diagnosis:** The core problem was an unstable **value function** (the "Critic" in our Actor-Critic model).
- **Theoretical Explanation:** Explained Variance measures how well the Critic's predictions of future rewards match the actual rewards the agent received. A value of 1.0 is a perfect prediction; a negative value means the Critic's predictions are worse than simply guessing the average outcome. The PPO algorithm updates the policy (the "Actor") based on an "advantage" calculation, which is heavily dependent on the Critic's predictions (`Advantage ≈ Actual Reward - Predicted Reward`). If the Critic's predictions are garbage, the advantage calculation is also garbage, and the Actor receives a noisy, unreliable gradient. The agent literally cannot tell if its actions are leading to good or bad states.
- **The PPO Loss Function:** The total loss is a combination of policy loss, value loss, and an entropy bonus: `Loss = Loss_policy + vf_coef * Loss_value - ent_coef * Loss_entropy`. In our case, `Loss_value` was massive and incorrect. Because it's part of the total loss, its huge, noisy gradients were likely overwhelming the smaller, more useful gradients from the policy and entropy losses, destabilizing the entire learning process.

### 5. Current Solution: Stabilizing the Critic

- **Solution:** To combat the unstable Critic, we are continuing training from the last checkpoint but with a significantly reduced **value function coefficient** (`--vf-coef 0.2`).
- **Rationale:** By lowering `vf-coef`, we are reducing the weight of the `Loss_value` in the total loss function. This effectively tells the optimizer: "For now, don't trust the Critic's updates so much. Pay more attention to the Actor's updates, which are guided by the dense `damage_reward` we added." This should stabilize the shared layers of the neural network, allowing the policy to improve consistently while giving the critic more time to gradually learn a better value estimate from a more stable policy. The next training run's logs will be monitored for an increasing `ExplainedVar`, which will signal that this approach is working.
