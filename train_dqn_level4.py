# train_dqn_level4.py
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Importamos TU función make_env
from make_env import make_env

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "doom_dqn"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    # Configuración del escenario
    scenario_path: str = "configs/health_gathering_supreme.cfg"

    # Partimos con la base de lo aprendido en el segundo modo de juego del Doom
    load_model: str = "doom_dqn_level3.pth"
    use_pretrained: bool = True

    total_timesteps: int = 800000
    learning_rate: float = 2.5e-4
    
    # --- Argumentos específicos de DQN ---
    num_envs: int = 1  # DQN suele funcionar mejor con 1 env, pero soporta vectorizado
    buffer_size: int = 25000  # Memoria de repetición
    gamma: float = 0.99
    tau: float = 1.0  # 1.0 = Hard update (copia total), < 1.0 = Soft update
    target_network_frequency: int = 1000  # Cada cuánto actualizamos la red objetivo
    batch_size: int = 32
    start_e: float = 0.6       # Epsilon inicial (exploración 60%)
    end_e: float = 0.05        # Epsilon final (5%)
    exploration_fraction: float = 0.6  # % del entrenamiento para bajar epsilon
    learning_starts: int = 10000  # Pasos aleatorios antes de empezar a entrenar
    train_frequency: int = 4      # Entrenar cada X pasos

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# --- 1. Red Neuronal Q-Network (Solo un cabezal) ---
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Misma CNN que usabas en PPO
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
            # Salida: Un valor Q para cada acción posible
            layer_init(nn.Linear(512, env.single_action_space.n), std=0.01),
        )

    def forward(self, x):
        return self.network(x / 255.0)

# --- 2. Replay Buffer Simple ---
class ReplayBuffer:
    def __init__(self, buffer_size, obs_shape, action_shape, device):
        self.obs = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32).to("cpu")
        self.next_obs = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32).to("cpu")
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.long).to("cpu")
        self.rewards = torch.zeros((buffer_size), dtype=torch.float32).to("cpu")
        self.dones = torch.zeros((buffer_size), dtype=torch.float32).to("cpu")
        self.pos = 0
        self.size = 0
        self.buffer_size = buffer_size
        self.device = device

    def add(self, obs, next_obs, action, reward, done):
        # Maneja inserción de múltiples entornos si num_envs > 1
        n_entries = obs.shape[0]
        for i in range(n_entries):
            idx = (self.pos + i) % self.buffer_size
            self.obs[idx] = torch.tensor(obs[i])
            self.next_obs[idx] = torch.tensor(next_obs[i])
            self.actions[idx] = torch.tensor(action[i])
            self.rewards[idx] = torch.tensor(reward[i])
            self.dones[idx] = torch.tensor(done[i])
        
        self.pos = (self.pos + n_entries) % self.buffer_size
        self.size = min(self.size + n_entries, self.buffer_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs].to(self.device),
            self.actions[idxs].to(self.device),
            self.rewards[idxs].to(self.device),
            self.next_obs[idxs].to(self.device),
            self.dones[idxs].to(self.device),
        )

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"Doom_DQN_{args.exp_name}_{args.seed}_{int(time.time())}"
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Configuración de entornos
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.scenario_path) for i in range(args.num_envs)]
    )

    # Inicialización de Redes (Q-Network y Target Network)
    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)

    if args.use_pretrained and os.path.exists(args.load_model): 
        print(f"Cargando modelo pre-entrenado desde: {args.load_model}") 
        try: 
            # 1. Cargar el archivo
            pretrained_dict = torch.load(args.load_model, map_location=device)
            
            # 2. Obtener el estado actual de la nueva red (Level 3)
            model_dict = q_network.state_dict()
            
            # 3. Filtrar: Quedarnos solo con las capas que tienen el mismo tamaño
            # Esto eliminará la última capa lineal si el número de acciones cambió
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            
            # 4. Actualizar y cargar
            model_dict.update(pretrained_dict)
            q_network.load_state_dict(model_dict)
            
            print(f"¡Pesos cargados! (Se han ignorado las capas con diferente tamaño de acciones)") 
        except Exception as e:
            print(f"Error crítico cargando pesos: {e}") 

    target_network.load_state_dict(q_network.state_dict()) # Copia inicial

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    
    # Inicializar Buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        device
    )

    obs, _ = envs.reset(seed=args.seed)
    
    print(f"¡Iniciando entrenamiento DQN por {args.total_timesteps} pasos!")
    start_time = time.time()

    # Bucle paso a paso (diferente a PPO)
    for global_step in range(args.total_timesteps):
        
        # --- 3. Estrategia Epsilon-Greedy ---
        epsilon = args.start_e - (args.start_e - args.end_e) * (global_step / (args.total_timesteps * args.exploration_fraction))
        epsilon = max(epsilon, args.end_e) # No bajar del mínimo

        # Selección de acción
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Paso del entorno
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Logging
        if "episode" in infos:
            env_dones = infos.get("_episode", [False] * args.num_envs)
            for i, done in enumerate(env_dones):
                if done:
                    print(f"Global Step={global_step}, Reward={infos['episode']['r'][i]:.2f}, Epsilon={epsilon:.3f}")
                    writer.add_scalar("charts/episodic_return", infos['episode']['r'][i], global_step)
                    writer.add_scalar("charts/episodic_length", infos['episode']['l'][i], global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)

        # Guardar en Buffer
        # Manejo correcto de "final observation" para truncamientos
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        dones = np.logical_or(terminations, truncations)
        rb.add(obs, real_next_obs, actions, rewards, dones)

        obs = next_obs

        # --- 4. Entrenamiento (Si tenemos suficientes datos) ---
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                # Desempaquetar datos
                # b_obs: (Batch, 4, 84, 84)
                # b_actions: (Batch)
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = data

                with torch.no_grad():
                    # Calcular el valor objetivo (Target Q)
                    # Q_target = r + gamma * max(Q_target_net(s'))
                    target_max, _ = target_network(b_next_obs).max(dim=1)
                    td_target = b_rewards + args.gamma * target_max * (1 - b_dones)
                
                # Calcular el valor actual (Current Q)
                old_val = q_network(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze()
                
                # Loss (MSE)
                loss = F.mse_loss(old_val, td_target)

                # Optimizar
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

            # --- 5. Actualizar Target Network ---
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    # Guardado final
    print("Guardando modelo DQN...")
    torch.save(q_network.state_dict(), "doom_dqn_level4.pth")
    envs.close()
    writer.close()
