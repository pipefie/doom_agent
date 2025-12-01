# play_ppo.py
import time
import gymnasium as gym
import vizdoom as vzd
import torch
import torch.nn as nn
import numpy as np
import cv2
from wrappers import VizDoomGym

# --- 1. DEFINICIÓN DEL AGENTE (Idéntica a train.py) ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
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
        )
        self.actor = layer_init(nn.Linear(512, action_space_size), std=0.01)
        
        # --- AÑADIDO: El Critic es necesario para cargar el archivo,
        # aunque no lo usemos para jugar ---
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_action(self, x):
        # x viene como (1, 4, 84, 84)
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        # Elegimos la acción con mayor probabilidad (determinista)
        return torch.argmax(logits, dim=1)

# --- 2. CONFIGURACIÓN DEL ENTORNO VISUAL ---
def make_visual_env(scenario_path):
    game = vzd.DoomGame()
    game.load_config(scenario_path)
    
    # AQUI ESTÁ LA MAGIA: VENTANA VISIBLE
    game.set_window_visible(True) 
    
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8) # Importante mantener GRAY8
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
        env = make_visual_env("basic.cfg")
    except Exception as e:
        print(f"Error al crear entorno: {e}")
        exit()
    
    # Inicializar Agente
    agent = Agent(env.action_space.n).to(device)
    
    # Cargar Pesos
    model_path = "doom_agent_model.pth"
    try:
        print(f"Intentando cargar {model_path}...")
        agent.load_state_dict(torch.load(model_path, map_location=device))
        print("¡Modelo cargado con éxito!")
    except FileNotFoundError:
        print(f"ERROR: No se encuentra '{model_path}'. Asegúrate de haber entrenado primero.")
        env.close()
        exit()
    except RuntimeError as e:
        print(f"ERROR de estructura del modelo: {e}")
        env.close()
        exit()

    agent.eval() # Modo evaluación (desactiva dropout, etc)

    num_episodes = 10
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"--- Episodio {episode + 1} ---")
        
        while not done:
            # El entorno devuelve numpy, convertimos a Tensor y añadimos dimensión de batch
            # obs shape: (4, 84, 84) -> input shape: (1, 4, 84, 84)
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = agent.get_action(obs_tensor)
            
            # Ejecutar acción
            # .item() convierte el tensor de 1 elemento a un entero de Python
            obs, reward, terminated, truncated, info = env.step(action.item())
            
            done = terminated or truncated
            total_reward += reward
            
            # Pequeña pausa para que veas lo que pasa (sino va muy rápido)
            time.sleep(0.05) 
            
        print(f"Puntuación final: {total_reward}")
        time.sleep(1) # Pausa entre partidas

    print("Fin de la demostración.")
    env.close()
