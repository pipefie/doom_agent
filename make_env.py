import vizdoom as vzd
from wrappers import VizDoomGym

def make_env(scenario_path):
    def thunk():
        game = vzd.DoomGame()
        game.load_config(scenario_path)
        # Important: Set window to invisible for training speed
        game.set_window_visible(False) 
        game.set_mode(vzd.Mode.PLAYER)
        env = VizDoomGym(game)
        # Standard wrappers for stability
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Stack 4 frames so agent sees motion
        env = gym.wrappers.FrameStack(env, 4) 
        return env
    return thunk


