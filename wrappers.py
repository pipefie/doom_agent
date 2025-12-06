# wrappers.py
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
        
        # --- CAMBIO IMPORTANTE: Detección dinámica de acciones ---
        # Esto cuenta cuántos botones hay en el .cfg (ej: 3, 5, 7...)
        self.num_actions = self.game.get_available_buttons_size()
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Observación (H, W, Canales se manejan con FrameStack luego)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def step(self, action):
        # Crear un array de ceros del tamaño correcto de botones
        buttons = [0] * self.num_actions
        # Activar el botón seleccionado por la red neuronal
        buttons[action] = 1
        
        # Frame skip de 4
        reward = self.game.make_action(buttons, 4)
        
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        if state:
            # Procesar imagen a escala de grises y resize
            screen = state.screen_buffer
            screen = cv2.resize(screen, (84, 84))
            obs = screen
        else:
            obs = np.zeros((84, 84), dtype=np.uint8)
        
        info = {}
        # gymnasium requiere devolver: obs, reward, terminated, truncated, info
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state()
        
        if state:
            screen = state.screen_buffer
            screen = cv2.resize(screen, (84, 84))
            obs = screen
        else:
            obs = np.zeros((84, 84), dtype=np.uint8)
            
        return obs, {}