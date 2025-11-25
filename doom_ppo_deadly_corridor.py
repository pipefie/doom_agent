import argparse
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Tuple

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import vizdoom as vzd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


# =========================
# 1. Config dataclass
# =========================

@dataclass
class DoomConfig:
    scenario_cfg: str
    scenario_name: str = "basic"
    frame_skip: int = 4
    frame_width: int = 84
    frame_height: int = 84
    frame_stack: int = 4
    grayscale: bool = True
    use_shared_actions: bool = True
    # Reward shaping parameters (tuned per-scenario by CLI)
    kill_reward: float = 5.0
    ammo_penalty: float = 0.01
    progress_scale: float = 0.0
    health_penalty: float = 0.0
    death_penalty: float = 0.0


# Scenario helpers so CLI users can pick stages by name and inherit sensible defaults.
SCENARIO_CFG_MAP = {
    "basic": "configs/basic.cfg",
    "defend_the_center": "configs/defend_the_center.cfg",
    "deadly_corridor": "configs/deadly_corridor.cfg",
}

# Reward shaping defaults are intentionally light for early tasks and richer for Deadly Corridor.
SCENARIO_SHAPING_DEFAULTS = {
    "basic": {
        "kill_reward": 5.0,
        "ammo_penalty": 0.01,
        "progress_scale": 0.0,
        "health_penalty": 0.0,
        "death_penalty": 0.0,
    },
    "defend_the_center": {
        "kill_reward": 5.0,
        "ammo_penalty": 0.01,
        "progress_scale": 0.0,
        "health_penalty": 0.0,
        "death_penalty": 0.0,
    },
    "deadly_corridor": {
        # Stronger shaping to offset sparse rewards and punish early suicides.
        "kill_reward": 10.0,
        "ammo_penalty": 0.02,
        "progress_scale": 0.05,  # Reward forward motion down the corridor.
        "health_penalty": 0.05,
        "death_penalty": 5.0,
    },
}

# Unified action set used across scenarios so checkpoint heads always match.
# This is the superset of buttons required by Deadly Corridor and works in earlier tasks too.
SHARED_ACTION_BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK,
]


# =========================
# 2. Environment wrapper
# =========================

class VizDoomGymnasiumEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around ViZDoom.

    - Returns observations as stacked grayscale frames: (C, H, W),
      where C = frame_stack.
    - Uses Gymnasium API: step() -> (obs, reward, terminated, truncated, info).
    - Reward shaping is scenario-aware:
        * Basic / Defend the Center → kill reward + ammo penalty
        * Deadly Corridor → adds progress along the corridor, health penalties,
          and a small optional death penalty to avoid the suicide trap.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: DoomConfig, render_mode: str | None = None):
        super().__init__()
        assert render_mode in (None, "human")
        self.cfg = cfg
        self.scenario_name = (cfg.scenario_name or "basic").lower()
        self.render_mode = render_mode

        # Initialize Doom game
        self.game = vzd.DoomGame()
        self.game.load_config(self.cfg.scenario_cfg)
        if self.cfg.use_shared_actions:
            # Force a consistent action list across scenarios so policy heads stay compatible
            # when loading checkpoints during curriculum transfers.
            self.game.set_available_buttons(SHARED_ACTION_BUTTONS)
        self._register_shaping_variables()
        self.game.init()

        # Define action space based on available buttons
        n_buttons = len(self.game.get_available_buttons())
        # Simple one-hot actions; later you can define a more compact action set
        self._actions = np.eye(n_buttons, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(len(self._actions))

        # Observation space: stacked grayscale frames (C, H, W), values in [0, 1]
        c = self.cfg.frame_stack
        h = self.cfg.frame_height
        w = self.cfg.frame_width
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(c, h, w),
            dtype=np.float32,
        )

        # Internal buffer for frame stacking
        self._frames = np.zeros((self.cfg.frame_stack, h, w), dtype=np.float32)

        # Game variables for reward shaping
        self.available_vars = self.game.get_available_game_variables()
        self.kill_idx = self._find_var_index(vzd.GameVariable.KILLCOUNT)
        self.ammo_idx = self._find_var_index(vzd.GameVariable.AMMO2)
        self.progress_idx = self._find_var_index(vzd.GameVariable.POSITION_X)
        self.health_idx = self._find_var_index(vzd.GameVariable.HEALTH)

        self.prev_killcount = 0.0
        self.prev_ammo = 0.0
        self.prev_posx = 0.0
        self.prev_health = 0.0

    def _register_shaping_variables(self):
        """
        Ensure the game exposes the variables we need for shaping.
        We add them before init() so ViZDoom will track them in get_state().
        """
        current = set(self.game.get_available_game_variables())
        requested = []

        if self.cfg.kill_reward != 0.0 and vzd.GameVariable.KILLCOUNT not in current:
            requested.append(vzd.GameVariable.KILLCOUNT)
        if self.cfg.ammo_penalty != 0.0 and vzd.GameVariable.AMMO2 not in current:
            requested.append(vzd.GameVariable.AMMO2)

        # Deadly Corridor (or any scenario using progress/health penalties) needs more signals.
        if self.scenario_name == "deadly_corridor" or self.cfg.progress_scale != 0.0:
            if vzd.GameVariable.POSITION_X not in current:
                requested.append(vzd.GameVariable.POSITION_X)
        if self.cfg.health_penalty != 0.0 or self.scenario_name == "deadly_corridor":
            if vzd.GameVariable.HEALTH not in current:
                requested.append(vzd.GameVariable.HEALTH)

        for var in requested:
            self.game.add_available_game_variable(var)

    def _find_var_index(self, target_var: vzd.GameVariable) -> int | None:
        """
        Return the index of target_var in the game's available variables list, or None.
        """
        for i, var in enumerate(self.available_vars):
            if var == target_var:
                return i
        return None

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert raw ViZDoom frame to processed (H, W) grayscale:
        - Handle both CHW and HWC layouts.
        - Resize to (frame_height, frame_width).
        - Normalize to [0, 1].
        """
        img = frame

        # ---- 1) Handle layout: CHW vs HWC vs HW ----
        if img.ndim == 3:
            # If first dim looks like channels (1,3,4) and last dim does NOT,
            # we assume CHW and transpose to HWC.
            if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                # (C, H, W) -> (H, W, C)
                img = np.transpose(img, (1, 2, 0))

            # Now we expect HWC
            if img.shape[2] == 1:
                # Already single-channel
                img = img[:, :, 0]
            elif img.shape[2] in (3, 4):
                # ViZDoom uses RGB format by default
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError(f"Unexpected number of channels in frame: {img.shape}")
        elif img.ndim == 2:
            # Already grayscale (H, W)
            pass
        else:
            raise ValueError(f"Unexpected frame ndim: {img.ndim}, shape={img.shape}")

        # ---- 2) Resize to target resolution ----
        img = cv2.resize(
            img,
            (self.cfg.frame_width, self.cfg.frame_height),
            interpolation=cv2.INTER_AREA,
        )

        # ---- 3) Normalize to [0, 1] float32 ----
        img = img.astype(np.float32) / 255.0 # Pixel values are originally 0-255

        return img  # (H, W)


    def _get_obs(self) -> np.ndarray:
        """
        Return the current stacked observation as (C, H, W).
        """
        return self._frames.copy()

    def _update_game_vars(self, state: vzd.GameState | None):
        """
        Update internal tracking of game variables for reward shaping.
        """
        if state is None:
            return
        vars_ = state.game_variables  # np.array
        if self.kill_idx is not None:
            self.prev_killcount = float(vars_[self.kill_idx])
        if self.ammo_idx is not None:
            self.prev_ammo = float(vars_[self.ammo_idx])
        if self.progress_idx is not None:
            self.prev_posx = float(vars_[self.progress_idx])
        if self.health_idx is not None:
            self.prev_health = float(vars_[self.health_idx])

    def _shape_reward(
        self,
        base_reward: float,
        state: vzd.GameState | None,
        terminated: bool,
    ) -> float:
        """
        Reward shaping:
        - +kill_reward * (delta_killcount)
        - -ammo_penalty * (ammo_used)
        - +progress_scale * (delta POSITION_X) for Deadly Corridor
        - -health_penalty * (health lost)
        - Optional death_penalty on episode end to discourage intentional suicides
        """
        shaped = base_reward
        if state is None:
            if terminated and self.cfg.death_penalty != 0.0:
                shaped -= self.cfg.death_penalty
            return shaped

        vars_ = state.game_variables
        # Kill reward
        if self.kill_idx is not None:
            killcount = float(vars_[self.kill_idx])
            delta_kill = killcount - self.prev_killcount
            if delta_kill > 0:
                shaped += self.cfg.kill_reward * delta_kill
            self.prev_killcount = killcount
        # Ammo penalty
        if self.ammo_idx is not None:
            ammo = float(vars_[self.ammo_idx])
            ammo_used = max(0.0, self.prev_ammo - ammo)
            if ammo_used > 0:
                shaped -= self.cfg.ammo_penalty * ammo_used
            self.prev_ammo = ammo

        # Progress shaping (Deadly Corridor): reward forward movement along X-axis.
        if self.progress_idx is not None and self.cfg.progress_scale != 0.0:
            posx = float(vars_[self.progress_idx])
            delta_x = posx - self.prev_posx
            # Negative delta_x naturally produces a penalty; no separate hyperparameter.
            if delta_x != 0.0:
                shaped += self.cfg.progress_scale * delta_x
            self.prev_posx = posx

        # Health shaping: penalize health loss to discourage reckless damage intake.
        if self.health_idx is not None and self.cfg.health_penalty != 0.0:
            health = float(vars_[self.health_idx])
            lost = self.prev_health - health
            if lost > 0.0:
                shaped -= self.cfg.health_penalty * lost
            self.prev_health = health

        return shaped

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Gymnasium reset() -> (obs, info)
        """
        if seed is not None:
            super().reset(seed=seed)

        self.game.new_episode()

        state = self.game.get_state()
        frame = state.screen_buffer  # (H, W, C) uint8
        processed = self._process_frame(frame)  # (H, W)

        # Fill stack with initial frame
        for i in range(self.cfg.frame_stack):
            self._frames[i] = processed

        # Initialize game variable trackers
        self._update_game_vars(state)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Gymnasium step() -> (obs, reward, terminated, truncated, info)
        """
        doom_action = self._actions[action].tolist()
        base_reward = self.game.make_action(doom_action, self.cfg.frame_skip)

        terminated = self.game.is_episode_finished()
        truncated = False  # no explicit time-limit truncation here

        if terminated:
            # No new state; keep last observation
            obs = self._frames
            info = {}
            state = None
        else:
            state = self.game.get_state()
            frame = state.screen_buffer  # (H, W, C)
            processed = self._process_frame(frame)  # (H, W)

            # Update frame stack: shift left, add new frame at end
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = processed

            obs = self._get_obs()
            info = {}

        # Reward shaping: adjust reward before returning
        reward = self._shape_reward(float(base_reward), state, terminated)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # ViZDoom handles rendering via its own window when configured.
            pass

    def close(self):
        self.game.close()


# =========================
# 3. Env factory (vectorized)
# =========================

def make_vizdoom_env(base_cfg: DoomConfig, seed: int, idx: int, render: bool = False):
    """
    Returns a thunk to create a single env instance, for use with SyncVectorEnv.
    """
    def thunk():
        env_cfg = DoomConfig(**asdict(base_cfg))
        env = VizDoomGymnasiumEnv(
            env_cfg,
            render_mode=("human" if render and idx == 0 else None),
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env
    return thunk


def make_envs(args, doom_cfg: DoomConfig):
    return gym.vector.SyncVectorEnv(
        [
            make_vizdoom_env(
                base_cfg=doom_cfg,
                seed=args.seed,
                idx=i,
                render=args.render,
            )
            for i in range(args.num_envs)
        ]
    )


# =========================
# 4. PPO Model (NatureCNN-style)
# =========================

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """
    Orthogonal initialization is a common trick in PPO / A2C:
    - Helps stabilize learning by controlling initial weight scales.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """
    Classic 3-layer CNN used in Atari / Doom-like PPO setups.
    Input: (B, C, H, W) with C=4 (stacked frames), H=W=84.
    Output: flattened feature vector.
    """

    def __init__(self, in_channels: int, features_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output shape by doing a forward pass with dummy input
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.zeros(1, in_channels, 84, 84)
            ).shape[1]

        self.linear = layer_init(nn.Linear(n_flatten, features_dim))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class PPOAgent(nn.Module):
    """
    Actor-Critic network for PPO:
    - NatureCNN feature extractor
    - Policy head (Categorical over discrete actions)
    - Value head
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space):
        super().__init__()
        assert isinstance(action_space, gym.spaces.Discrete)
        self.in_channels = obs_space.shape[0]
        self.features = NatureCNN(self.in_channels, features_dim=512)
        self.actor = layer_init(nn.Linear(512, action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.critic(features)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ):
        features = self.features(x)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features)
        return action, logprob, entropy, value


# =========================
# 5. Argparse
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Doom - Deadly Corridor (CleanRL-style)")

    # Env & Doom
    parser.add_argument("--scenario-cfg", type=str, default="configs/basic.cfg",
                        help="Path to ViZDoom .cfg scenario")
    parser.add_argument("--scenario-name", type=str, default=None,
                        help="Scenario name shortcut; overrides --scenario-cfg when provided")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint to warm start model+optimizer")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Number of environment steps per update (per env)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)

    parser.add_argument("--render", action="store_true", help="Render first env window")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-shared-actions", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a unified action set across scenarios to keep policy heads compatible")

    # Reward shaping (scenario defaults kick in when args are left None)
    parser.add_argument("--kill-reward", type=float, default=None,
                        help="Extra reward per enemy kill (delta KILLCOUNT)")
    parser.add_argument("--ammo-penalty", type=float, default=None,
                        help="Penalty per ammo used (AMMO2 delta)")
    parser.add_argument("--progress-scale", type=float, default=None,
                        help="Scale for POSITION_X progress reward (Deadly Corridor)")
    parser.add_argument("--health-penalty", type=float, default=None,
                        help="Penalty per point of health lost")
    parser.add_argument("--death-penalty", type=float, default=None,
                        help="Additional penalty when dying early")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--anneal-lr", action="store_true",
                        help="Linearly anneal the learning rate to 0 over training")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", action="store_true",
                        help="Normalize advantages")
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # Logging & checkpoints
    parser.add_argument("--track", action="store_true", help="Enable TensorBoard tracking")
    parser.add_argument("--tb-logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--exp-name", type=str, default="doom_ppo_deadly_corridor", help="Experiment name")
    parser.add_argument("--save-interval", type=int, default=100_000,
                        help="Save checkpoint every N global steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # System
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--torch-deterministic", action="store_true",
                        help="Sets torch.backends.cudnn.deterministic=True for reproducibility")
    


    args = parser.parse_args()
    return args


# =========================
# 6. Main PPO training loop
# =========================

def main():
    args = parse_args()

    # Scenario resolution and reward shaping defaults
    if args.scenario_name:
        scenario_name = args.scenario_name.lower().replace("-", "_")
        if scenario_name in SCENARIO_CFG_MAP:
            args.scenario_cfg = SCENARIO_CFG_MAP[scenario_name]
        else:
            raise ValueError(f"Unknown scenario name: {args.scenario_name}")
    else:
        scenario_name = os.path.splitext(os.path.basename(args.scenario_cfg))[0].lower()
        scenario_name = scenario_name.replace("-", "_")
    args.scenario_name = scenario_name

    if not os.path.isfile(args.scenario_cfg):
        raise FileNotFoundError(f"Scenario cfg not found: {args.scenario_cfg}")

    defaults = SCENARIO_SHAPING_DEFAULTS.get(
        scenario_name, SCENARIO_SHAPING_DEFAULTS["basic"]
    )
    # Fill in shaping hyperparameters; None means "use scenario default".
    args.kill_reward = defaults["kill_reward"] if args.kill_reward is None else args.kill_reward
    args.ammo_penalty = defaults["ammo_penalty"] if args.ammo_penalty is None else args.ammo_penalty
    args.progress_scale = defaults["progress_scale"] if args.progress_scale is None else args.progress_scale
    args.health_penalty = defaults["health_penalty"] if args.health_penalty is None else args.health_penalty
    args.death_penalty = defaults["death_penalty"] if args.death_penalty is None else args.death_penalty

    doom_cfg = DoomConfig(
        scenario_cfg=args.scenario_cfg,
        scenario_name=scenario_name,
        use_shared_actions=args.use_shared_actions,
        kill_reward=args.kill_reward,
        ammo_penalty=args.ammo_penalty,
        progress_scale=args.progress_scale,
        health_penalty=args.health_penalty,
        death_penalty=args.death_penalty,
    )

    print(f"[INFO] Scenario '{scenario_name}' using cfg={args.scenario_cfg}")
    print("[INFO] Reward shaping "
          f"(kill={doom_cfg.kill_reward}, ammo_penalty={doom_cfg.ammo_penalty}, "
          f"progress_scale={doom_cfg.progress_scale}, health_penalty={doom_cfg.health_penalty}, "
          f"death_penalty={doom_cfg.death_penalty})")

    # Basic asserts & setup
    assert args.num_steps > 0
    assert args.num_envs > 0
    batch_size = args.num_envs * args.num_steps
    assert batch_size % args.num_minibatches == 0, \
        "batch_size must be divisible by num_minibatches"
    minibatch_size = batch_size // args.num_minibatches

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Envs
    envs = make_envs(args, doom_cfg)
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space

    # Agent
    agent = PPOAgent(obs_space, act_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Optional warm start for curriculum or resuming training
    global_step = 0
    if args.load_checkpoint is not None:
        if not os.path.isfile(args.load_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = int(checkpoint.get("global_step", 0))
        print(f"[INFO] Loaded checkpoint from {args.load_checkpoint} "
              f"(global_step={global_step})")

    # Logging
    run_name = (
        f"{args.exp_name}_"
        f"{os.path.basename(args.scenario_cfg).replace('.cfg', '')}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_seed{args.seed}"
    )
    writer = None
    if args.track:
        os.makedirs(args.tb_logdir, exist_ok=True)
        log_dir = os.path.join(args.tb_logdir, run_name)
        writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs at: {log_dir}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Storage for PPO
    obs_shape = obs_space.shape
    obs = np.zeros((args.num_steps, args.num_envs, *obs_shape), dtype=np.float32)
    actions = np.zeros((args.num_steps, args.num_envs), dtype=np.int64)
    logprobs = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    dones = np.zeros((args.num_steps, args.num_envs), dtype=np.bool_)
    values = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    initial_global_step = global_step
    start_time = time.time()

    next_obs, _ = envs.reset()
    next_done = np.zeros(args.num_envs, dtype=np.bool_)

    num_updates = args.total_timesteps // batch_size

    print(f"[INFO] Starting training from global_step={global_step} "
          f"for {args.total_timesteps} timesteps, "
          f"{num_updates} updates, batch_size={batch_size}, minibatch_size={minibatch_size}.")

    for update in range(1, num_updates + 1):
        # Anneal the learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now
        else:
            lr_now = args.learning_rate

        # Collect rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Convert obs to torch tensor
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, logprob_tensor, _, value_tensor = agent.get_action_and_value(obs_tensor)
            action_np = action_tensor.cpu().numpy()
            value_np = value_tensor.squeeze(-1).cpu().numpy()
            logprob_np = logprob_tensor.cpu().numpy()

            actions[step] = action_np
            values[step] = value_np
            logprobs[step] = logprob_np

            # Step envs
            next_obs, reward, terminated, truncated, infos = envs.step(action_np)
            done = np.logical_or(terminated, truncated)
            rewards[step] = reward
            next_done = done

            # Logging episodic returns from RecordEpisodeStatistics (Gymnasium)
            if writer is not None and "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        ep_r = info["episode"]["r"]
                        ep_l = info["episode"]["l"]
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)

        # Bootstrap value for last observation
        with torch.no_grad():
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)
            _, _, _, next_value = agent.get_action_and_value(next_obs_tensor)
            next_value = next_value.squeeze(-1).cpu().numpy()

        # Compute GAE advantages
        advantages = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_nonterminal = 1.0 - next_done.astype(np.float32)
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones[t + 1].astype(np.float32)
                next_values = values[t + 1]
            delta = (
                rewards[t]
                + args.gamma * next_values * next_nonterminal
                - values[t]
            )
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            )
        returns = advantages + values

        # Flatten the batch
        b_obs = torch.tensor(obs.reshape((-1, *obs_shape)), device=device)
        b_actions = torch.tensor(actions.reshape(-1), device=device)
        b_logprobs = torch.tensor(logprobs.reshape(-1), device=device)
        b_advantages = torch.tensor(advantages.reshape(-1), device=device)
        b_returns = torch.tensor(returns.reshape(-1), device=device)
        b_values = torch.tensor(values.reshape(-1), device=device)

        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

        # PPO update
        inds = np.arange(batch_size)
        clip_fracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs_old = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values_old = b_values[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, mb_actions
                )
                newvalue = newvalue.view(-1)

                logratio = newlogprob - mb_logprobs_old
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                clip_fracs.append(
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values_old + torch.clamp(
                    newvalue - mb_values_old,
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: avoids exploding gradients in deep networks,
                # which can destabilize training.
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                print(f"[INFO] Early stopping at epoch {epoch} due to KL={approx_kl:.5f}")
                break

        # Logging
        explained_var = np.nan
        if np.var(values) > 0.0:
            explained_var = 1 - np.var(returns - values) / np.var(values)

        if writer is not None:
            writer.add_scalar("charts/learning_rate", lr_now, global_step)
            writer.add_scalar("charts/clip_fraction", np.mean(clip_fracs), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl, global_step)
            writer.add_scalar("charts/explained_variance", explained_var, global_step)

        if update % 10 == 0 or update == num_updates:
            steps_this_run = max(global_step - initial_global_step, 1)
            fps = int(steps_this_run / (time.time() - start_time))
            print(
                f"[UPDATE {update}/{num_updates}] "
                f"global_step={global_step} | fps={fps} | "
                f"Vloss={v_loss.item():.4f} | Ploss={pg_loss.item():.4f} | "
                f"Entropy={entropy_loss.item():.4f} | KL={approx_kl:.5f} | "
                f"ExplainedVar={explained_var:.3f}"
            )

        # Save checkpoint
        if global_step % args.save_interval == 0 or update == num_updates:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"{run_name}_step{global_step}.pt"
            )
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[INFO] Saved checkpoint to {ckpt_path}")

    envs.close()
    if writer is not None:
        writer.close()

    total_time = time.time() - start_time
    print(
        f"[INFO] Training finished in {total_time/60:.2f} minutes, "
        f"final_global_step={global_step}, steps_this_run={global_step - initial_global_step}"
    )


if __name__ == "__main__":
    main()
