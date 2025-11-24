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
from torch.distributions.categorical import Categorical

# Import the environment and agent from the training script.
# This assumes `eval_doom_agent.py` and `doom_ppo_deadly_corridor.py`
# live in the same directory.
from doom_ppo_deadly_corridor import VizDoomGymnasiumEnv, DoomConfig, PPOAgent


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

    return parser.parse_args()


def load_agent(
    checkpoint_path: str,
    obs_shape,
    action_space,
    device: torch.device,
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

    agent = PPOAgent(fake_obs_space, fake_action_space).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()  # Put in evaluation mode (disables dropout, etc.)

    print("[INFO] Agent weights loaded successfully.")
    return agent


def select_action(
    agent: PPOAgent,
    obs: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
) -> int:
    """
    Given a single observation (C, H, W), returns an action (int).

    - deterministic=False → sample from Categorical(logits)
    - deterministic=True  → take argmax(logits)
    """
    # Add batch dimension and convert to torch tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        features = agent.features(obs_tensor)  # (1, feat_dim)
        logits = agent.actor(features)         # (1, n_actions)

        if deterministic:
            action_tensor = torch.argmax(logits, dim=-1)
        else:
            dist = Categorical(logits=logits)
            action_tensor = dist.sample()

    action = int(action_tensor.cpu().numpy()[0])
    return action


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
    cfg = DoomConfig(scenario_cfg=args.scenario_cfg)
    render_mode = "human" if args.render else None
    env = VizDoomGymnasiumEnv(cfg, render_mode=render_mode)

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

        print(f"\n=== Episode {ep + 1}/{args.episodes} ===")

        while not done and step_count < args.max_steps_per_episode:
            action = select_action(agent, obs, device, deterministic=args.deterministic)
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
