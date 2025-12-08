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
        
        # 12 ACCIONES COMBINADAS (Igual que PPO)
        self.actions = [
            [0,0,0,0,0,0,0], # 0: Idle
            [1,0,0,0,0,0,0], # 1: Forward
            [0,0,1,0,0,0,0], # 2: Turn Left
            [0,0,0,1,0,0,0], # 3: Turn Right
            [0,0,0,0,1,0,0], # 4: Strafe Left
            [0,0,0,0,0,1,0], # 5: Strafe Right
            [0,0,0,0,0,0,1], # 6: Attack
            [1,0,0,0,0,0,1], # 7: Forward + Attack (CRÍTICO)
            [0,0,0,0,1,0,1], # 8: Strafe Left + Attack
            [0,0,0,0,0,1,1], # 9: Strafe Right + Attack
            [0,0,1,0,0,0,1], # 10: Turn Left + Attack
            [0,0,0,1,0,0,1], # 11: Turn Right + Attack
        ]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

        # Detectar índices de variables dinámicamente
        self.damage_idx = self._get_var_idx(vzd.GameVariable.DAMAGECOUNT)
        self.kill_idx = self._get_var_idx(vzd.GameVariable.KILLCOUNT)
        self.ammo_idx = self._get_var_idx(vzd.GameVariable.AMMO2)
        self.health_idx = self._get_var_idx(vzd.GameVariable.HEALTH)
        
        # Inicializar previos
        self.prev_damage = 0
        self.prev_kill = 0
        self.prev_ammo = 0
        self.prev_health = 100

        # Resetear para llenar los previos correctamente
        self._update_prev_vars()

    def _get_var_idx(self, variable):
        try:
            # Busca si la variable está disponible en el juego actual
            available = self.game.get_available_game_variables()
            for i, v in enumerate(available):
                if v == variable:
                    return i
        except Exception:
            pass
        return None

    def _update_prev_vars(self):
        state = self.game.get_state()
        if state:
            vars = state.game_variables
            if self.damage_idx is not None: self.prev_damage = vars[self.damage_idx]
            if self.kill_idx is not None: self.prev_kill = vars[self.kill_idx]
            if self.ammo_idx is not None: self.prev_ammo = vars[self.ammo_idx]
            if self.health_idx is not None: self.prev_health = vars[self.health_idx]

    def step(self, action_idx):
        game_reward = self.game.make_action(self.actions[action_idx], 4)
        
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        reward = game_reward * 0.05
        
        if state:
            # Procesar imagen
            screen = cv2.resize(state.screen_buffer, (84, 84))
            obs = screen
            
            # Calcular recompensa PPO-Style
            vars = state.game_variables
            
            # 1. Daño infligido
            if self.damage_idx is not None:
                curr_dmg = vars[self.damage_idx]
                if curr_dmg > self.prev_damage:
                    reward += (curr_dmg - self.prev_damage) * 0.01
                self.prev_damage = curr_dmg
                
            # 2. Enemigos matados
            if self.kill_idx is not None:
                curr_kill = vars[self.kill_idx]
                if curr_kill > self.prev_kill:
                    reward += (curr_kill - self.prev_kill) * 5.0
                self.prev_kill = curr_kill
                
            # 3. Gasto de munición (Penalización pequeña)
            if self.ammo_idx is not None:
                curr_ammo = vars[self.ammo_idx]
                if curr_ammo < self.prev_ammo:
                    reward -= (self.prev_ammo - curr_ammo) * 0.01
                self.prev_ammo = curr_ammo

            if self.health_idx is not None:
                curr_health = vars[self.health_idx]
                if curr_health < self.prev_health:
                    # -0.2 por cada punto de vida perdido
                    reward -= (self.prev_health - curr_health) * 0.2
                self.prev_health = curr_health
                
        else:
            obs = np.zeros((84, 84), dtype=np.uint8)
        
        # Penalización por existir (para que se de prisa)
        if not done:
            reward -= 0.001

        if done:
            if self.prev_health <= 0: # O usa una lógica basada en el reward del juego
                reward -= 10.0 # Castigo extra por fracasar muriendo

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        self._update_prev_vars()
        
        state = self.game.get_state()
        if state:
            return cv2.resize(state.screen_buffer, (84, 84)), {}
        return np.zeros((84, 84), dtype=np.uint8), {}