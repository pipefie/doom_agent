"""
Evaluation script for a trained PPO Doom agent.

This script is intentionally separate from the training script
(`doom_ppo_deadly_corridor.py`) to keep things clean:

- It only loads a checkpoint, never trains.
- It always uses a SINGLE environment.
- It runs episodes with rendering, so you can actually watch the agent play.
- It is easy to modify without touching the PPO training code.

Usage example:

    uv run python eval_doom_agent.py \
        --checkpoint checkpoints/doom_ppo_deadly_corridor_basic_..._step100000.pt \
        --scenario-cfg configs/basic.cfg \
        --episodes 5 \
        --render \
        --deterministic \
        --cuda

"""

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch

# Import the environment and agent from the training script.
# This assumes `eval_doom_agent.py` and `doom_ppo_deadly_corridor.py`
# live in the same directory.
from doom_ppo_deadly_corridor import (
    VizDoomGymnasiumEnv,
    DoomConfig,
    PPOAgent,
    SCENARIO_CFG_MAP,
    SCENARIO_SHAPING_DEFAULTS,
    SHARED_ACTION_BUTTONS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Doom agent")

    # Which checkpoint and scenario to use
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint saved by the training script",
    )
    parser.add_argument(
        "--scenario-cfg",
        type=str,
        default="configs/basic.cfg",
        help="Path to ViZDoom .cfg scenario (e.g. configs/basic.cfg, configs/deadly_corridor.cfg)",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Scenario name shortcut; overrides --scenario-cfg when provided",
    )

    # Evaluation settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=10_000,
        help="Safety cap on steps per episode to avoid infinite loops",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="If set, use argmax policy instead of sampling from the action distribution",
    )

    # Rendering & device
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (should almost always be True for evaluation)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available",
    )

    # Random seed just for consistency in env reset
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for the environment",
    )

    parser.add_argument(
        "--sleep-per-step",
        type=float,
        default=0.0,
        help="Optional delay (in seconds) between env steps during evaluation, e.g. 0.03",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Override frame skip for evaluation to match training (default: use scenario cfg)",
    )
    parser.add_argument(
        "--use-shared-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the unified action set across scenarios (matches training defaults)",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Enable LSTM memory (must match training setting)",
    )
    parser.add_argument(
        "--use-combo-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use curated combo actions (applies to all scenarios if set; must match training)",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=512,
        help="Hidden size of the LSTM when enabled",
    )

    # Reward shaping to keep metrics consistent with training
    parser.add_argument(
        "--kill-reward",
        type=float,
        default=None,
        help="Extra reward per enemy kill (delta KILLCOUNT)",
    )
    parser.add_argument(
        "--ammo-penalty",
        type=float,
        default=None,
        help="Penalty per ammo used (AMMO2 delta)",
    )
    parser.add_argument(
        "--progress-scale",
        type=float,
        default=None,
        help="Scale for POSITION_X progress reward (Deadly Corridor)",
    )
    parser.add_argument(
        "--health-penalty",
        type=float,
        default=None,
        help="Penalty per point of health lost",
    )
    parser.add_argument(
        "--death-penalty",
        type=float,
        default=None,
        help="Additional penalty when dying early",
    )
    parser.add_argument(
        "--living-penalty",
        type=float,
        default=None,
        help="Per-step penalty when no recent kill (encourages clearing enemies)",
    )
    parser.add_argument(
        "--kill-grace-steps",
        type=int,
        default=None,
        help="Number of steps after a kill before living-penalty resumes",
    )
    parser.add_argument(
        "--forward-penalty",
        type=float,
        default=None,
        help="Penalty per forward delta when no recent kill (discourages rushing past threats)",
    )
    parser.add_argument(
        "--damage-reward",
        type=float,
        default=None,
        help="Reward per point of DAMAGECOUNT (damage inflicted)",
    )

    return parser.parse_args()


def build_eval_config(args) -> DoomConfig:
    """
    Mirror the training-side scenario + shaping resolution so evaluation metrics
    match the reward signals the policy was trained on.
    """
    if args.scenario_name:
        scenario_name = args.scenario_name.lower().replace("-", "_")
        if scenario_name in SCENARIO_CFG_MAP:
            args.scenario_cfg = SCENARIO_CFG_MAP[scenario_name]
        else:
            raise ValueError(f"Unknown scenario name: {args.scenario_name}")
    else:
        scenario_name = os.path.splitext(os.path.basename(args.scenario_cfg))[0].lower()
        scenario_name = scenario_name.replace("-", "_")

    defaults = SCENARIO_SHAPING_DEFAULTS.get(
        scenario_name, SCENARIO_SHAPING_DEFAULTS["basic"]
    )
    kill_reward = defaults["kill_reward"] if args.kill_reward is None else args.kill_reward
    ammo_penalty = defaults["ammo_penalty"] if args.ammo_penalty is None else args.ammo_penalty
    progress_scale = defaults["progress_scale"] if args.progress_scale is None else args.progress_scale
    health_penalty = defaults["health_penalty"] if args.health_penalty is None else args.health_penalty
    death_penalty = defaults["death_penalty"] if args.death_penalty is None else args.death_penalty
    living_penalty = defaults.get("living_penalty", 0.0) if args.living_penalty is None else args.living_penalty
    kill_grace_steps = defaults.get("kill_grace_steps", 0) if args.kill_grace_steps is None else args.kill_grace_steps
    forward_penalty = defaults.get("forward_penalty", 0.0) if args.forward_penalty is None else args.forward_penalty
    damage_reward = defaults["damage_reward"] if args.damage_reward is None else args.damage_reward

    return DoomConfig(
        scenario_cfg=args.scenario_cfg,
        scenario_name=scenario_name,
        frame_skip=args.frame_skip if args.frame_skip is not None else 4,
        use_combo_actions=args.use_combo_actions,
        use_shared_actions=args.use_shared_actions,
        kill_reward=kill_reward,
        ammo_penalty=ammo_penalty,
        progress_scale=progress_scale,
        health_penalty=health_penalty,
        death_penalty=death_penalty,
        living_penalty=living_penalty,
        kill_grace_steps=kill_grace_steps,
        forward_penalty=forward_penalty,
        damage_reward=damage_reward,
    )


def load_agent(
    checkpoint_path: str,
    obs_shape,
    action_space,
    device: torch.device,
    use_lstm: bool = False,
    lstm_hidden_size: int = 512,
) -> PPOAgent:
    """
    Create a PPOAgent with the correct input/output sizes and load its weights
    from the checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build a fresh agent with the same architecture as in training
    fake_obs_space = type("ObsSpace", (), {"shape": obs_shape})
    fake_action_space = action_space

    agent = PPOAgent(
        fake_obs_space,
        fake_action_space,
        use_lstm=use_lstm,
        lstm_hidden_size=lstm_hidden_size,
    ).to(device)
    ckpt_state = checkpoint["model_state_dict"]
    for key in ["actor.weight", "actor.bias"]:
        if key in ckpt_state:
            if key in agent.state_dict() and ckpt_state[key].shape != agent.state_dict()[key].shape:
                print(f"[WARN] Dropping incompatible {key} from checkpoint (ckpt {ckpt_state[key].shape} vs model {agent.state_dict()[key].shape})")
                ckpt_state.pop(key)
    agent.load_state_dict(ckpt_state, strict=False)
    agent.eval()  # Put in evaluation mode (disables dropout, etc.)

    print("[INFO] Agent weights loaded successfully.")
    return agent


def select_action(
    agent: PPOAgent,
    obs: np.ndarray,
    device: torch.device,
    lstm_state,
    done: bool,
    deterministic: bool = False,
) -> tuple[int, tuple]:
    """
    Given a single observation (C, H, W), returns an action (int).

    - deterministic=False → sample from Categorical(logits)
    - deterministic=True  → take argmax(logits)
    """
    # Add batch dimension and convert to torch tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    done_tensor = torch.tensor([float(done)], device=device)
    with torch.no_grad():
        action_tensor, _, _, _, new_state = agent.get_action_and_value(
            obs_tensor, lstm_state, done_tensor, deterministic=deterministic
        )

    action = int(action_tensor.cpu().numpy()[0])
    return action, new_state


def main():
    args = parse_args()

    # -------------------------
    # 1. Device selection
    # -------------------------
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # 2. Create a single Doom env for evaluation
    # -------------------------
    # We reuse the same DoomConfig and VizDoomGymnasiumEnv classes from training,
    # but here we only create ONE env (no vectorization).
    cfg = build_eval_config(args)
    print(
        f"[INFO] Scenario '{cfg.scenario_name}' using cfg={cfg.scenario_cfg} "
        f"(kill={cfg.kill_reward}, ammo_penalty={cfg.ammo_penalty}, "
        f"progress_scale={cfg.progress_scale}, health_penalty={cfg.health_penalty}, "
        f"death_penalty={cfg.death_penalty}, damage_reward={cfg.damage_reward}, "
        f"shared_actions={cfg.use_shared_actions})"
    )
    render_mode = "human" if args.render else None
    env = VizDoomGymnasiumEnv(cfg, render_mode=render_mode)

    if cfg.use_combo_actions:
        print(f"[INFO] Action space size: {env.action_space.n} (Expected: {len(SHARED_ACTION_BUTTONS)})")


    # Gymnasium-style spaces from this single env
    obs_space = env.observation_space
    action_space = env.action_space

    # -------------------------
    # 3. Load the trained agent
    # -------------------------
    agent = load_agent(
        checkpoint_path=args.checkpoint,
        obs_shape=obs_space.shape,
        action_space=action_space,
        device=device,
        use_lstm=args.use_lstm,
        lstm_hidden_size=args.lstm_hidden_size,
    )

    # -------------------------
    # 4. Run evaluation episodes
    # -------------------------
    print(
        f"[INFO] Starting evaluation: {args.episodes} episodes, "
        f"scenario={args.scenario_cfg}, deterministic={args.deterministic}"
    )

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        step_count = 0
        lstm_state = agent.get_initial_state(1, device) if args.use_lstm else None

        print(f"\n=== Episode {ep + 1}/{args.episodes} ===")

        while not done and step_count < args.max_steps_per_episode:
            action, lstm_state = select_action(
                agent,
                obs,
                device,
                lstm_state,
                done,
                deterministic=args.deterministic,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            step_count += 1

            if args.sleep_per_step > 0.0:
                time.sleep(args.sleep_per_step)

            # Optional: print every N steps if you want to see live reward
            # if step_count % 20 == 0:
            #     print(f"Step {step_count:04d} | reward={reward:.2f} | cum={ep_reward:.2f}")

        print(
            f"Episode {ep + 1} finished in {step_count} steps "
            f"with total shaped reward = {ep_reward:.2f}"
        )

    env.close()
    print("\n[INFO] Evaluation completed.")


if __name__ == "__main__":
    main()
