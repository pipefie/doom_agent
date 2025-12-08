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
        
        # MANTENEMOS LAS 12 ACCIONES para compatibilidad con la red neuronal del Nivel 3
        # Aunque disparar no sirva de nada aquí, la red espera 12 salidas.
        self.actions = [
            # --- Acciones Simples ---
            [0,0,0], # 0: Idle
            [1,0,0], # 1: Avanzar (Forward)
            [0,1,0], # 2: Girar Izquierda
            [0,0,1], # 3: Girar Derecha
            
            # --- Acciones Combinadas (CRÍTICAS para este nivel) ---
            # Permiten correr y girar a la vez, o correr en diagonal.
            
            [1,1,0], # 4: Avanzar + Girar Izq (Curva rápida)
            [1,0,1], # 5: Avanzar + Girar Der (Curva rápida)

        ]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

        self.health_idx = self._get_var_idx(vzd.GameVariable.HEALTH)
        self.prev_health = 100
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
        state = self.game.get_state()
        if state:
            vars = state.game_variables
            if self.health_idx is not None: self.prev_health = vars[self.health_idx]

    def step(self, action_idx):
            # Ejecutar acción (4 tics por frame es estándar)
            self.game.make_action(self.actions[action_idx], 4)
            
            state = self.game.get_state()
            done = self.game.is_episode_finished()
            
            # --- CÁLCULO DE RECOMPENSA ---
            reward = 0.0
            
            if state:
                # Variables actuales
                vars = state.game_variables
                curr_health = vars[self.health_idx] if self.health_idx is not None else 0
                
                # 1. Recompensa por vivir (incentivo base)
                reward += 0.05 
                
                # 2. Diferencia de salud
                health_delta = curr_health - self.prev_health
                
                if health_delta > 0:
                    # ¡Recogió un Medkit! (+1.0 por cada kit aprox)
                    reward += 1.0
                # Actualizar previo
                self.prev_health = curr_health
                
                # Procesar imagen
                screen = cv2.resize(state.screen_buffer, (84, 84))
                obs = screen
            else:
                obs = np.zeros((84, 84), dtype=np.uint8)
                # Penalización extra por morir
                reward -= 5.0 

            return obs, reward, done, False, {}
    
    def reset(self, seed=None, options=None):
        self.game.new_episode()
        self._update_prev_vars()
        state = self.game.get_state()
        if state:
            return cv2.resize(state.screen_buffer, (84, 84)), {}
        return np.zeros((84, 84), dtype=np.uint8), {}