import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import vizdoom as vzd

class VizDoomGym(gym.Env):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.game.init()
        
        self.actions = [
            [0,0,0,0,0,0,0], # 0: Idle
            [1,0,0,0,0,0,0], # 1: Forward
            [0,0,1,0,0,0,0], # 2: Turn Left
            [0,0,0,1,0,0,0], # 3: Turn Right
            [0,0,0,0,1,0,0], # 4: Strafe Left
            [0,0,0,0,0,1,0], # 5: Strafe Right
            [0,0,0,0,0,0,1], # 6: Attack
            [1,0,0,0,0,0,1], # 7: Forward + Attack
            [0,0,0,0,1,0,1], # 8: Strafe Left + Attack
            [0,0,0,0,0,1,1], # 9: Strafe Right + Attack
            [0,0,1,0,0,0,1], # 10: Turn Left + Attack
            [0,0,0,1,0,0,1], # 11: Turn Right + Attack
        ]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

        self.health_idx = self._get_var_idx(vzd.GameVariable.HEALTH)
        self.kills_idx = self._get_var_idx(vzd.GameVariable.FRAGCOUNT)
        self.ammo_idx = self._get_var_idx(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        
        self.prev_health = 100
        self.prev_kills = 0
        self.prev_ammo = 0
        self._update_prev_vars()

    def _get_var_idx(self, variable):
        try:
            available = self.game.get_available_game_variables()
            for i, v in enumerate(available):
                if v == variable:
                    return i
        except Exception:
            pass
        return None

    def _get_player_data(self, var):
        if isinstance(var, np.ndarray):
            return int(var.item(0))
        else:
            return int(var)

    def _update_prev_vars(self):
        state = self.game.get_state()
        if state:
            vars = state.game_variables
            if self.health_idx is not None: self.prev_health = self._get_player_data(vars[self.health_idx])
            if self.kills_idx is not None: self.prev_kills = self._get_player_data(vars[self.kills_idx])
            if self.ammo_idx is not None: self.prev_ammo = self._get_player_data(vars[self.ammo_idx])

    def step(self, action_idx):
        self.game.make_action(self.actions[action_idx], 4)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        reward = 0.0
        
        if state:
            vars = state.game_variables
            
            curr_health = self._get_player_data(vars[self.health_idx])
            curr_kills = self._get_player_data(vars[self.kills_idx])
            curr_ammo = self._get_player_data(vars[self.ammo_idx])

            # CÁLCULO DE RECOMPENSAS BASADO EN PAPERS
            kill_delta = curr_kills - self.prev_kills
            health_delta = curr_health - self.prev_health
            ammo_delta = curr_ammo - self.prev_ammo

            reward_kill = kill_delta * 100
            reward_health = health_delta * 1.0
            reward_ammo = ammo_delta * 0.1
            reward_living = 0.01

            reward = reward_kill + reward_health + reward_ammo + reward_living
            
            # Penalización extra al morir para que sea una señal fuerte de fracaso
            if done and curr_health <= 0:
                reward -= 100

            # Actualizar variables
            self.prev_health = curr_health
            self.prev_kills = curr_kills
            self.prev_ammo = curr_ammo
            
            obs = cv2.resize(state.screen_buffer, (84, 84))
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            # No hay información de estado, no se da recompensa ni castigo
            reward = 0

        return obs, reward, done, False, {}
    
    def reset(self, seed=None, options=None):
        self.game.new_episode()
        self._update_prev_vars()
        state = self.game.get_state()
        
        if state:
            obs = cv2.resize(state.screen_buffer, (84, 84))
            return obs, {}
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            return obs, {}