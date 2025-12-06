# play_dqn_level2.py
import time
import gymnasium as gym
import vizdoom as vzd
import torch
import torch.nn as nn
import numpy as np
from wrappers import VizDoomGym  # Asegúrate de tener wrappers.py

# --- 1. DEFINICIÓN DE LA RED (Debe ser IDÉNTICA a train_dqn.py) ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
        # Misma estructura exacta que en train_dqn.py
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU(),
            # Esta es la capa "network.9" que daba error antes
            layer_init(nn.Linear(512, action_space_size), std=0.01),
        )

    def forward(self, x):
        return self.network(x / 255.0)

# --- 2. CONFIGURACIÓN DEL ENTORNO VISUAL ---
def make_visual_env(scenario_path):
    game = vzd.DoomGame()
    game.load_config(scenario_path)
    game.set_window_visible(True) # ¡Ventana visible!
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_sound_enabled(False)
    
    env = VizDoomGym(game)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env

# --- 3. BUCLE DE JUEGO ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando modelo en: {device}")

    # Crear entorno
    try:
        env = make_visual_env("defend_the_center.cfg")
    except Exception as e:
        print(f"Error al crear entorno: {e}")
        exit()
    
    # Inicializar la Red DQN
    model = QNetwork(env.action_space.n).to(device)
    
    # Cargar Pesos
    model_path = "doom_dqn_level2.pth"
    try:
        print(f"Intentando cargar {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("¡Modelo cargado con éxito!")
    except FileNotFoundError:
        print(f"ERROR: No se encuentra '{model_path}'. Asegúrate de haber entrenado primero.")
        env.close()
        exit()
    except RuntimeError as e:
        print(f"ERROR de estructura del modelo: {e}")
        env.close()
        exit()

    model.eval() # Modo evaluación

    num_episodes = 20
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"--- Episodio {episode + 1} ---")
        
        while not done:
            # Preparar observación: (1, 4, 84, 84)
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Obtener Q-values
                q_values = model(obs_tensor)
                # Elegir la acción con el Q-value más alto (Argmax)
                action = torch.argmax(q_values, dim=1).item()
            
            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            
            # Pausa para ver el juego a velocidad humana
            time.sleep(0.05) 
            
        print(f"Puntuación final: {total_reward}")
        time.sleep(1)

    print("Fin de la demostración.")
    env.close()
