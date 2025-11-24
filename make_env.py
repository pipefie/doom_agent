import gymnasium as gym
import vizdoom as vzd
from wrappers import VizDoomGym

def make_env(scenario_path):
    def thunk():
        game = vzd.DoomGame()
        game.load_config(scenario_path)
        
        # 1. Sin ventana para velocidad
        game.set_window_visible(False) 
        
        # 2. MODO JUGADOR
        game.set_mode(vzd.Mode.PLAYER)
        
        # 3. CRÍTICO: Forzar formato GRIS (elimina el canal de color extra)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        
        # 4. Apagar audio para evitar errores de Linux/Pipewire
        game.set_sound_enabled(False)
        
        env = VizDoomGym(game)
        
        # Wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Usamos la versión moderna de Gymnasium
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        
        return env
    return thunk
