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
        
        # El espacio de acciones que definiste
        self.actions = [
            # Tu lista de acciones...
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

        # --- Obtener índices de variables de juego ---
        self.health_idx = self._get_var_idx(vzd.GameVariable.HEALTH)
        self.kills_idx = self._get_var_idx(vzd.GameVariable.FRAGCOUNT)
        # AÑADIDO: Rastreador de armadura
        self.armor_idx = self._get_var_idx(vzd.GameVariable.ARMOR)
        
        # --- Variables de estado previo ---
        self.prev_health = 100
        self.prev_kills = 0
        self.prev_armor = 0 # La armadura inicial suele ser 0
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

    def _update_prev_vars(self):
        """Actualiza todas las variables de estado previas (vida, kills, armadura)."""
        state = self.game.get_state()
        if state:
            vars = state.game_variables
            if self.health_idx is not None: self.prev_health = vars[self.health_idx]
            if self.kills_idx is not None: self.prev_kills = vars[self.kills_idx]
            if self.armor_idx is not None: self.prev_armor = vars[self.armor_idx]

    def step(self, action_idx):
        # Ejecutar la acción en el juego
        self.game.make_action(self.actions[action_idx], 4)
        
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        # --- NUEVO CÁLCULO DE RECOMPENSAS SEGÚN TUS ESPECIFICACIONES ---
        reward = 0.0
        
        if state:
            vars = state.game_variables
            
            # 1. Recompensa por vivir
            reward += 0.05
            
            # Obtener variables actuales
            curr_health = vars[self.health_idx] if self.health_idx is not None else 0
            curr_kills = vars[self.kills_idx] if self.kills_idx is not None else 0
            curr_armor = vars[self.armor_idx] if self.armor_idx is not None else 0
            
            # 2. Recompensa/Penalización por cambio de vida
            health_delta = curr_health - self.prev_health
            if health_delta > 0:
                reward += 10.0  # Curarse
            elif health_delta < 0:
                reward -= 1.0   # Recibir daño
            
            # 3. Recompensa por conseguir armadura
            armor_delta = curr_armor - self.prev_armor
            if armor_delta > 0:
                reward += 5.0
            
            # 4. Recompensa por matar
            kill_delta = curr_kills - self.prev_kills
            if kill_delta > 0:
                reward += 20.0 * kill_delta
            
            # 5. Penalización por disparar
            # CORREGIDO: El botón de ataque es el de índice 6 en tu lista
            if self.actions[action_idx][6] == 1:
                reward -= 0.05
                
            # Actualizar variables para el próximo frame
            self.prev_health = curr_health
            self.prev_kills = curr_kills
            self.prev_armor = curr_armor
            
            # Procesar la observación (pantalla del juego)
            screen = cv2.resize(state.screen_buffer, (84, 84))
            obs = screen
        else:
            # Si el estado es nulo (el agente murió)
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            # 6. Penalización por morir
            reward -= 10.0

        return obs, reward, done, False, {}
    
    def reset(self, seed=None, options=None):
        self.game.new_episode()
        self._update_prev_vars() # Actualiza los contadores al inicio
        state = self.game.get_state()
        if state:
            return cv2.resize(state.screen_buffer, (84, 84)), {}
        return np.zeros((84, 84), dtype=np.uint8), {}