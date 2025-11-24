import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class VizDoomGym(gym.Env):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.game.init()
        
        # Action Space (3 botones: Left, Right, Shoot)
        self.action_space = spaces.Discrete(3) 
        
        # Definimos espacio (H, W). FrameStack añadirá la profundidad (4) después.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def step(self, action):
        buttons = [0] * 3
        buttons[action] = 1
        
        reward = self.game.make_action(buttons, 4)
        
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        if state:
            # Viene como (H, W) porque pusimos GRAY8 en make_env
            screen = state.screen_buffer
            screen = cv2.resize(screen, (84, 84))
            obs = screen
        else:
            obs = np.zeros((84, 84), dtype=np.uint8)
        
        info = {}
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state()
        screen = state.screen_buffer
        screen = cv2.resize(screen, (84, 84))
        # Devolvemos (84, 84) limpio
        obs = screen 
        return obs, {}
