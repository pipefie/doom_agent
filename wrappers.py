import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class VizDoomGym(gym.Env):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.game.init()
        
        # Define Action Space (3 buttons: Left, Right, Shoot)
        self.action_space = spaces.Discrete(3) 
        
        # Define Observation Space (C, H, W) for PyTorch
        # Resizing to 84x84 is standard for DQN/PPO
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )

    def step(self, action):
        # Convert discrete action index to one-hot buttons
        buttons = [0] * 3
        buttons[action] = 1
        
        # Make action and skip 4 frames (standard RL practice)
        reward = self.game.make_action(buttons, 4)
        
        # Get state
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        if state:
            # Grab screen buffer
            screen = state.screen_buffer
            # Resize and Grayscale
            screen = cv2.resize(screen, (84, 84))
            # Add channel dimension (1, 84, 84)
            obs = np.expand_dims(screen, axis=0)
        else:
            obs = np.zeros((1, 84, 84), dtype=np.uint8)
        
        # Handle "info" dict (useful for debugging)
        info = {}
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state()
        screen = state.screen_buffer
        screen = cv2.resize(screen, (84, 84))
        obs = np.expand_dims(screen, axis=0)
        return obs, {}
