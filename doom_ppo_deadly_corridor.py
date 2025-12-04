import argparse
import os
import time
from collections import deque
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
    use_combo_actions: bool = False
    frame_skip: int = 4
    frame_width: int = 84
    frame_height: int = 84
    frame_stack: int = 4
    grayscale: bool = True
    use_shared_actions: bool = True
    # Reward shaping
    kill_reward: float = 5.0
    ammo_penalty: float = 0.01
    progress_scale: float = 0.0
    health_penalty: float = 0.0
    health_delta_scale: float = 0.0
    death_penalty: float = 0.0
    living_penalty: float = 0.0
    kill_grace_steps: int = 0
    forward_penalty: float = 0.0
    damage_reward: float = 0.0


SCENARIO_CFG_MAP = {
    "basic": "configs/basic.cfg",
    "defend_the_center": "configs/defend_the_center.cfg",
    "deadly_corridor": "configs/deadly_corridor.cfg",
}

SCENARIO_SHAPING_DEFAULTS = {
    "basic": {
        "kill_reward": 1.0,
        "ammo_penalty": 0.01,
        "progress_scale": 0.0,
        "health_penalty": 0.0,
        "health_delta_scale": 0.0,
        "death_penalty": 0.0,
        "living_penalty": 0.01,
        "kill_grace_steps": 0,
        "forward_penalty": 0.0,
        "damage_reward": 0.0,
    },
    "defend_the_center": {
        "kill_reward": 1.0,
        "ammo_penalty": 0.01,
        "progress_scale": 0.0,
        "health_penalty": 0.1,
        "health_delta_scale": 0.0,
        "death_penalty": 1.0,
        "living_penalty": 0.0,
        "kill_grace_steps": 0,
        "forward_penalty": 0.0,
        "damage_reward": 0.5,
    },
    "deadly_corridor": {
        "kill_reward": 2.0,
        "ammo_penalty": 0.02,
        "progress_scale": 0.01,
        "health_penalty": 0.05,
        "health_delta_scale": 0.0,
        "death_penalty": 1.0,
        "living_penalty": 0.0,
        "kill_grace_steps": 0,
        "forward_penalty": 0.0,
        "damage_reward": 0.5,
    },
}

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
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg: DoomConfig, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.scenario_name = (cfg.scenario_name or "basic").lower()
        self.render_mode = render_mode
        self.steps_since_kill = 0

        self.game = vzd.DoomGame()
        self.game.load_config(self.cfg.scenario_cfg)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24) # Force RGB24 for MoviePy
        
        if self.cfg.use_shared_actions:
            self.game.set_available_buttons(SHARED_ACTION_BUTTONS)
            
        self._register_shaping_variables()
        self.game.set_window_visible(False)
        if render_mode == "human":
            self.game.set_window_visible(True)
            
        self.game.init()

        self.screen_w = self.game.get_screen_width()
        self.screen_h = self.game.get_screen_height()

        if self.cfg.use_combo_actions:
            self._actions = self._build_actions_by_scenario()
            self.action_space = gym.spaces.Discrete(len(self._actions))
        else:
            n_buttons = len(self.game.get_available_buttons())
            self._actions = np.eye(n_buttons, dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(len(self._actions))

        c = self.cfg.frame_stack
        h = self.cfg.frame_height
        w = self.cfg.frame_width
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)
        self._frames = np.zeros((self.cfg.frame_stack, h, w), dtype=np.float32)

        self.available_vars = self.game.get_available_game_variables()
        self.kill_idx = self._find_var_index(vzd.GameVariable.KILLCOUNT)
        self.ammo_idx = self._find_var_index(vzd.GameVariable.AMMO2)
        self.progress_idx = self._find_var_index(vzd.GameVariable.POSITION_X)
        self.health_idx = self._find_var_index(vzd.GameVariable.HEALTH)
        self.damage_idx = self._find_var_index(vzd.GameVariable.DAMAGECOUNT)

        self.prev_killcount = 0.0
        self.prev_ammo = 0.0
        self.prev_posx = 0.0
        self.prev_health = 0.0
        self.prev_damage = 0.0

    def _register_shaping_variables(self):
        current = set(self.game.get_available_game_variables())
        requested = []
        if self.cfg.kill_reward != 0.0 and vzd.GameVariable.KILLCOUNT not in current:
            requested.append(vzd.GameVariable.KILLCOUNT)
        if self.cfg.ammo_penalty != 0.0 and vzd.GameVariable.AMMO2 not in current:
            requested.append(vzd.GameVariable.AMMO2)
        if self.scenario_name == "deadly_corridor" or self.cfg.progress_scale != 0.0:
            if vzd.GameVariable.POSITION_X not in current: requested.append(vzd.GameVariable.POSITION_X)
        if self.cfg.health_penalty != 0.0 or self.scenario_name == "deadly_corridor":
            if vzd.GameVariable.HEALTH not in current: requested.append(vzd.GameVariable.HEALTH)
        if self.cfg.damage_reward != 0.0 and vzd.GameVariable.DAMAGECOUNT not in current:
            requested.append(vzd.GameVariable.DAMAGECOUNT)
        for var in requested: self.game.add_available_game_variable(var)

    def _find_var_index(self, target_var: vzd.GameVariable) -> int | None:
        for i, var in enumerate(self.available_vars):
            if var == target_var: return i
        return None

    def _build_actions_by_scenario(self) -> np.ndarray:
        def make_action_row(*indices):
            row = np.zeros(7, dtype=np.uint8)
            for i in indices: row[i] = 1
            return row

        actions_list = []
        name = self.scenario_name
        if name == "basic":
            actions_list = [
                make_action_row(),                  
                make_action_row(4),                 
                make_action_row(5),                 
                make_action_row(6),                 
                make_action_row(4, 6),              
                make_action_row(5, 6),              
            ]
        elif name == "defend_the_center":
            actions_list = [
                make_action_row(),                  
                make_action_row(2),                 
                make_action_row(3),                 
                make_action_row(6),                 
                make_action_row(2, 6),              
                make_action_row(3, 6),              
            ]
        elif name == "deadly_corridor":
            actions_list = [
                make_action_row(),                  
                make_action_row(0),                 
                make_action_row(2),                 
                make_action_row(3),                 
                make_action_row(4),                 
                make_action_row(5),                 
                make_action_row(6),                 
                make_action_row(0, 6),              
                make_action_row(4, 6),              
                make_action_row(5, 6),              
                make_action_row(2, 6),              
                make_action_row(3, 6),              
            ]
        return np.array(actions_list, dtype=np.uint8)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        img = frame
        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4): img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 1: img = img[:, :, 0]
            elif img.shape[2] in (3, 4): img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.cfg.frame_width, self.cfg.frame_height), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return img

    def _get_obs(self) -> np.ndarray:
        return self._frames.copy()

    def _update_game_vars(self, state):
        if state is None: return
        vars_ = state.game_variables
        if self.kill_idx is not None: self.prev_killcount = float(vars_[self.kill_idx])
        if self.ammo_idx is not None: self.prev_ammo = float(vars_[self.ammo_idx])
        if self.progress_idx is not None: self.prev_posx = float(vars_[self.progress_idx])
        if self.health_idx is not None: self.prev_health = float(vars_[self.health_idx])
        if self.damage_idx is not None: self.prev_damage = float(vars_[self.damage_idx])

    def _shape_reward(self, base_reward: float, state: vzd.GameState | None, terminated: bool) -> float:
        shaped = base_reward
        if state is None:
            if terminated and self.cfg.death_penalty != 0.0: shaped -= self.cfg.death_penalty
            return shaped

        vars_ = state.game_variables
        if self.kill_idx is not None:
            kc = float(vars_[self.kill_idx])
            delta = kc - self.prev_killcount
            if delta > 0:
                shaped += self.cfg.kill_reward * delta
                self.steps_since_kill = 0
            self.prev_killcount = kc
        
        if self.ammo_idx is not None:
            ammo = float(vars_[self.ammo_idx])
            used = max(0.0, self.prev_ammo - ammo)
            if used > 0: shaped -= self.cfg.ammo_penalty * used
            self.prev_ammo = ammo

        delta_x = 0.0
        if self.progress_idx is not None:
            px = float(vars_[self.progress_idx])
            delta_x = px - self.prev_posx
            if self.cfg.progress_scale != 0.0 and delta_x != 0.0: shaped += self.cfg.progress_scale * delta_x
            self.prev_posx = px

        if (self.cfg.forward_penalty > 0.0 and self.progress_idx is not None and 
                delta_x > 0.0 and self.steps_since_kill > self.cfg.kill_grace_steps):
            shaped -= self.cfg.forward_penalty * delta_x

        if self.health_idx is not None:
            h = float(vars_[self.health_idx])
            if self.cfg.health_penalty != 0.0:
                lost = self.prev_health - h
                if lost > 0.0: shaped -= self.cfg.health_penalty * lost
            if self.cfg.health_delta_scale != 0.0:
                delta_h = h - self.prev_health
                if delta_h != 0.0: shaped += self.cfg.health_delta_scale * delta_h
            self.prev_health = h

        if self.damage_idx is not None and self.cfg.damage_reward != 0.0:
            d = float(vars_[self.damage_idx])
            delta_d = d - self.prev_damage
            if delta_d > 0.0: shaped += self.cfg.damage_reward * delta_d
            self.prev_damage = d

        return shaped

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None: super().reset(seed=seed)
        self.game.new_episode()
        self.steps_since_kill = 0
        state = self.game.get_state()
        frame = state.screen_buffer
        processed = self._process_frame(frame)
        for i in range(self.cfg.frame_stack): self._frames[i] = processed
        self._update_game_vars(state)
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        base_reward = self.game.make_action(self._actions[action].tolist(), self.cfg.frame_skip)
        terminated = self.game.is_episode_finished()
        truncated = False

        if terminated:
            obs = self._frames
            info = {}
            state = None
        else:
            state = self.game.get_state()
            frame = state.screen_buffer
            processed = self._process_frame(frame)
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = processed
            obs = self._get_obs()
            info = {}

        reward = self._shape_reward(float(base_reward), state, terminated)
        if (self.cfg.living_penalty > 0.0 and self.steps_since_kill >= self.cfg.kill_grace_steps and not terminated):
            reward -= self.cfg.living_penalty
        self.steps_since_kill += 1

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self.game.is_episode_finished():
                return np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            state = self.game.get_state()
            if state: return state.screen_buffer
        return None

    def close(self):
        self.game.close()


# =========================
# 3. Env factory (vectorized)
# =========================

def make_vizdoom_env(base_cfg: DoomConfig, seed: int, idx: int, run_name: str):
    def thunk():
        env_cfg = DoomConfig(**asdict(base_cfg))
        render_mode = "rgb_array" if idx == 0 else None
        env = VizDoomGymnasiumEnv(env_cfg, render_mode=render_mode)
        # Note: We REMOVED RecordEpisodeStatistics here to use manual tracking
        # to ensure 100% reliability with custom logs.
        
        if idx == 0:
            video_folder = os.path.join("videos", run_name)
            env = gym.wrappers.RecordVideo(
                env, 
                video_folder=video_folder, 
                episode_trigger=lambda x: x % 100 == 0,
                disable_logger=True
            )
        env.reset(seed=seed + idx)
        return env
    return thunk

def make_envs(args, doom_cfg: DoomConfig, run_name: str):
    return gym.vector.SyncVectorEnv(
        [make_vizdoom_env(base_cfg=doom_cfg, seed=args.seed, idx=i, run_name=run_name) 
         for i in range(args.num_envs)]
    )


# =========================
# 4. PPO Model
# =========================

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    def __init__(self, in_channels: int, features_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, in_channels, 84, 84)).shape[1]
        self.linear = layer_init(nn.Linear(n_flatten, features_dim))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.linear(x)
        return self.activation(x)

class PPOAgent(nn.Module):
    def __init__(self, obs_space: gym.Space, action_space: gym.Space, use_lstm: bool = False, lstm_hidden_size: int = 512):
        super().__init__()
        self.in_channels = obs_space.shape[0]
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = 1
        self.features = NatureCNN(self.in_channels, features_dim=512)
        core_output_dim = 512
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=core_output_dim, hidden_size=self.lstm_hidden_size, num_layers=self.num_layers)
            core_output_dim = self.lstm_hidden_size
        self.actor = layer_init(nn.Linear(core_output_dim, action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(core_output_dim, 1), std=1.0)

    def get_initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h, c)

    def get_action_and_value(self, x: torch.Tensor, lstm_state=None, done=None, action=None, deterministic=False):
        batch_size = x.shape[0]
        features = self.features(x)
        if self.use_lstm:
            if lstm_state is None: lstm_state = self.get_initial_state(batch_size, x.device)
            if done is None: done = torch.zeros(batch_size, device=x.device)
            done = done.view(1, batch_size, 1)
            h, c = lstm_state
            h = h * (1.0 - done)
            c = c * (1.0 - done)
            lstm_out, new_state = self.lstm(features.unsqueeze(0), (h, c))
            features = lstm_out.squeeze(0)
        else: new_state = lstm_state
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None: action = dist.sample() if not deterministic else torch.argmax(logits, dim=-1)
        return action, dist.log_prob(action), dist.entropy(), self.critic(features), new_state


# =========================
# 5. Argparse
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-cfg", type=str, default="configs/basic.cfg")
    parser.add_argument("--scenario-name", type=str, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--use-combo-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-shared-actions", action=argparse.BooleanOptionalAction, default=True)
    # Rewards
    parser.add_argument("--kill-reward", type=float, default=None)
    parser.add_argument("--ammo-penalty", type=float, default=None)
    parser.add_argument("--progress-scale", type=float, default=None)
    parser.add_argument("--health-penalty", type=float, default=None)
    parser.add_argument("--health-delta-scale", type=float, default=None)
    parser.add_argument("--death-penalty", type=float, default=None)
    parser.add_argument("--living-penalty", type=float, default=None)
    parser.add_argument("--kill-grace-steps", type=int, default=None)
    parser.add_argument("--forward-penalty", type=float, default=None)
    parser.add_argument("--damage-reward", type=float, default=None)
    # PPO
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", action="store_true")
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ent-coef-warm", type=float, default=None)
    parser.add_argument("--ent-warm-steps", type=int, default=0)
    parser.add_argument("--ent-coef-final", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    # Logging
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--tb-logdir", type=str, default="runs")
    parser.add_argument("--exp-name", type=str, default="doom_ppo")
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    # System
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--torch-deterministic", action="store_true")
    parser.add_argument("--use-lstm", action="store_true")
    parser.add_argument("--lstm-hidden-size", type=int, default=512)
    return parser.parse_args()


# =========================
# 6. Main PPO training loop
# =========================

def main():
    args = parse_args()

    if args.scenario_name:
        scenario_name = args.scenario_name.lower().replace("-", "_")
        if scenario_name in SCENARIO_CFG_MAP: args.scenario_cfg = SCENARIO_CFG_MAP[scenario_name]
    else:
        scenario_name = os.path.splitext(os.path.basename(args.scenario_cfg))[0].lower().replace("-", "_")
    args.scenario_name = scenario_name

    defaults = SCENARIO_SHAPING_DEFAULTS.get(scenario_name, SCENARIO_SHAPING_DEFAULTS["basic"])
    for key, val in defaults.items():
        if getattr(args, key) is None: setattr(args, key, val)
    args.ent_coef_warm = args.ent_coef if args.ent_coef_warm is None else args.ent_coef_warm
    args.ent_coef_final = args.ent_coef if args.ent_coef_final is None else args.ent_coef_final

    doom_cfg = DoomConfig(
        scenario_cfg=args.scenario_cfg, scenario_name=scenario_name,
        use_combo_actions=args.use_combo_actions, use_shared_actions=args.use_shared_actions,
        frame_skip=args.frame_skip, kill_reward=args.kill_reward, ammo_penalty=args.ammo_penalty,
        progress_scale=args.progress_scale, health_penalty=args.health_penalty,
        health_delta_scale=args.health_delta_scale, death_penalty=args.death_penalty,
        living_penalty=args.living_penalty, kill_grace_steps=args.kill_grace_steps,
        forward_penalty=args.forward_penalty, damage_reward=args.damage_reward,
    )

    print(f"[INFO] Scenario: {scenario_name} | Cfg: {args.scenario_cfg}")
    
    assert args.num_steps > 0
    assert args.num_envs > 0
    batch_size = args.num_envs * args.num_steps
    minibatch_size = args.num_envs // args.num_minibatches if args.use_lstm else batch_size // args.num_minibatches
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    lstm_tag = "lstm" if args.use_lstm else "ff"
    run_name = f"{args.exp_name}_{scenario_name}_{lstm_tag}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_seed{args.seed}"
    
    envs = make_envs(args, doom_cfg, run_name)
    agent = PPOAgent(envs.single_observation_space, envs.single_action_space, args.use_lstm, args.lstm_hidden_size).to(device)
    
    # Checkpoint Loading Logic
    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print(f"[INFO] Loading checkpoint from {args.load_checkpoint}...")
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            # We only load the model weights, not the optimizer, to allow for a fresh start in the new scenario
            agent.load_state_dict(checkpoint["model_state_dict"])
            print("[INFO] Checkpoint loaded successfully (Agent weights only).")
        else:
            print(f"[ERROR] Checkpoint file not found: {args.load_checkpoint}")
            exit(1)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    writer = SummaryWriter(os.path.join(args.tb_logdir, run_name)) if args.track else None
    if args.track: print(f"[INFO] TensorBoard: {os.path.join(args.tb_logdir, run_name)}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    obs = np.zeros((args.num_steps, args.num_envs, *envs.single_observation_space.shape), dtype=np.float32)
    actions = np.zeros((args.num_steps, args.num_envs), dtype=np.int64)
    logprobs = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    dones = np.zeros((args.num_steps, args.num_envs), dtype=np.bool_)
    values = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    
    lstm_states_h, lstm_states_c = None, None
    if args.use_lstm:
        lstm_states_h = np.zeros((args.num_steps, args.num_envs, agent.num_layers, agent.lstm_hidden_size), dtype=np.float32)
        lstm_states_c = np.zeros_like(lstm_states_h)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_done = np.zeros(args.num_envs, dtype=np.bool_)
    next_lstm_state = agent.get_initial_state(args.num_envs, device) if args.use_lstm else None

    # Stats tracking - MANUAL IMPLEMENTATION
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    current_ep_reward = np.zeros(args.num_envs)
    current_ep_len = np.zeros(args.num_envs)

    num_updates = args.total_timesteps // batch_size
    print(f"[INFO] Start Training: {args.total_timesteps} steps, {num_updates} updates.")

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            for pg in optimizer.param_groups: pg["lr"] = lr_now
        else: lr_now = args.learning_rate

        ent_coef_now = args.ent_coef
        if args.ent_warm_steps > 0:
            steps_elapsed = max(global_step, 0)
            if steps_elapsed < args.ent_warm_steps: ent_coef_now = args.ent_coef_warm
            else:
                decay_steps = max(args.total_timesteps - args.ent_warm_steps, 1)
                frac_ent = min(max((steps_elapsed - args.ent_warm_steps) / decay_steps, 0.0), 1.0)
                ent_coef_now = args.ent_coef_warm + frac_ent * (args.ent_coef_final - args.ent_coef_warm)

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)
                if args.use_lstm:
                    lstm_states_h[step] = next_lstm_state[0].permute(1, 0, 2).cpu().numpy()
                    lstm_states_c[step] = next_lstm_state[1].permute(1, 0, 2).cpu().numpy()
                    done_tensor = torch.tensor(next_done, dtype=torch.float32, device=device)
                    action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                        obs_tensor, next_lstm_state, done_tensor
                    )
                else:
                    action, logprob, _, value, _ = agent.get_action_and_value(obs_tensor)
            
            actions[step] = action.cpu().numpy()
            values[step] = value.squeeze(-1).cpu().numpy()
            logprobs[step] = logprob.cpu().numpy()

            next_obs, reward, terminated, truncated, infos = envs.step(actions[step])
            done = np.logical_or(terminated, truncated)
            rewards[step] = reward
            next_done = done

            # MANUAL TRACKING LOGIC
            current_ep_reward += reward
            current_ep_len += 1
            for i in range(args.num_envs):
                if done[i]:
                    # This specific env finished
                    r = current_ep_reward[i]
                    l = current_ep_len[i]
                    episode_returns.append(r)
                    episode_lengths.append(l)
                    if writer:
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)
                    # Reset stats for this env
                    current_ep_reward[i] = 0
                    current_ep_len[i] = 0

        # Bootstrap
        with torch.no_grad():
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)
            if args.use_lstm:
                next_done_tensor = torch.tensor(next_done, dtype=torch.float32, device=device)
                _, _, _, next_value, _ = agent.get_action_and_value(next_obs_tensor, next_lstm_state, next_done_tensor)
            else:
                _, _, _, next_value, _ = agent.get_action_and_value(next_obs_tensor)
            next_value = next_value.squeeze(-1).cpu().numpy()

        # GAE
        advantages = np.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done.astype(np.float32)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1].astype(np.float32)
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        # Flatten & Train
        b_obs = torch.tensor(obs, device=device)
        b_logprobs = torch.tensor(logprobs, device=device)
        b_actions = torch.tensor(actions, device=device)
        b_advantages = torch.tensor(advantages, device=device)
        b_returns = torch.tensor(returns, device=device)
        b_values = torch.tensor(values, device=device)
        b_dones = torch.tensor(dones, device=device).float()

        if not args.use_lstm:
            b_obs = b_obs.reshape((-1, *envs.single_observation_space.shape))
            b_logprobs = b_logprobs.reshape(-1)
            b_actions = b_actions.reshape(-1)
            b_advantages = b_advantages.reshape(-1)
            b_returns = b_returns.reshape(-1)
            b_values = b_values.reshape(-1)
            b_dones = b_dones.reshape(-1)

        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        env_inds = np.arange(args.num_envs)
        inds = np.arange(batch_size)
        clip_fracs = []

        for epoch in range(args.update_epochs):
            if args.use_lstm:
                np.random.shuffle(env_inds)
                for start in range(0, args.num_envs, minibatch_size):
                    end = start + minibatch_size
                    mb_env_inds = env_inds[start:end]
                    
                    mb_obs = b_obs[:, mb_env_inds]
                    mb_actions = b_actions[:, mb_env_inds]
                    mb_logprobs = b_logprobs[:, mb_env_inds]
                    mb_advantages = b_advantages[:, mb_env_inds]
                    mb_returns = b_returns[:, mb_env_inds]
                    mb_values = b_values[:, mb_env_inds]
                    mb_dones = b_dones[:, mb_env_inds]
                    mb_h0 = torch.tensor(lstm_states_h[0, mb_env_inds], device=device).permute(1, 0, 2)
                    mb_c0 = torch.tensor(lstm_states_c[0, mb_env_inds], device=device).permute(1, 0, 2)

                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value_lstm_loop(
                        agent, mb_obs, (mb_h0, mb_c0), mb_dones, mb_actions, args.num_steps
                    )
                    
                    newlogprob = newlogprob.reshape(-1)
                    newvalue = newvalue.reshape(-1)
                    entropy = entropy.reshape(-1)

                    logratio = newlogprob - mb_logprobs.reshape(-1)
                    ratio = logratio.exp()
                    with torch.no_grad(): approx_kl = ((ratio - 1) - logratio).mean().item()
                    clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    mb_adv_flat = mb_advantages.reshape(-1)
                    pg_loss1 = -mb_adv_flat * ratio
                    pg_loss2 = -mb_adv_flat * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    mb_ret_flat = mb_returns.reshape(-1)
                    v_loss_unclipped = (newvalue - mb_ret_flat) ** 2
                    v_clipped = mb_values.reshape(-1) + torch.clamp(newvalue - mb_values.reshape(-1), -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - mb_ret_flat) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef_now * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
            else:
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = inds[start:end]
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], action=b_actions[mb_inds])
                    newvalue = newvalue.view(-1)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad(): approx_kl = ((ratio - 1) - logratio).mean().item()
                    clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    mb_adv = b_advantages[mb_inds]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    mb_ret = b_returns[mb_inds]
                    v_loss_unclipped = (newvalue - mb_ret) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - mb_ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef_now * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl: break

        # Explained Variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Console Logging
        if update % 5 == 0:
            elapsed = time.time() - start_time
            fps = int(global_step / elapsed)
            mean_ret = np.mean(episode_returns) if len(episode_returns) > 0 else 0.0
            std_ret = np.std(episode_returns) if len(episode_returns) > 0 else 0.0
            max_ret = np.max(episode_returns) if len(episode_returns) > 0 else 0.0
            print(f"Update {update}/{num_updates} | Step {global_step} | FPS: {fps}")
            print(f"  Returns (Last 100): Mean={mean_ret:.2f} +/- {std_ret:.2f} | Max={max_ret:.2f}")
            print(f"  Losses: Value={v_loss.item():.3f} | Policy={pg_loss.item():.4f} | Ent={entropy_loss.item():.4f} | KL={approx_kl:.4f}")
            print("-" * 60)

        # TensorBoard Extra
        if writer:
            writer.add_scalar("charts/learning_rate", lr_now, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl, global_step)
            writer.add_scalar("charts/clip_fraction", np.mean(clip_fracs), global_step)
            writer.add_scalar("charts/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_histogram("charts/action_distribution", b_actions, global_step)

        if global_step % args.save_interval == 0 or update == num_updates:
            path = os.path.join(args.checkpoint_dir, f"{run_name}_step{global_step}.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "args": vars(args)
            }, path)
            print(f"[INFO] Saved Checkpoint: {path}")

    envs.close()
    if writer: writer.close()
    print(f"Done! Videos saved in videos/{run_name}")

def get_action_and_value_lstm_loop(agent, obs, lstm_state, dones, actions, num_steps):
    newlogprobs, entropies, newvalues = [], [], []
    cur_state = lstm_state
    for t in range(num_steps):
        _, logprob, entropy, value, cur_state = agent.get_action_and_value(obs[t], cur_state, dones[t], actions[t])
        newlogprobs.append(logprob)
        entropies.append(entropy)
        newvalues.append(value)
    return None, torch.stack(newlogprobs), torch.stack(entropies), torch.stack(newvalues), None

PPOAgent.get_action_and_value_lstm_loop = staticmethod(get_action_and_value_lstm_loop)

if __name__ == "__main__":
    main()