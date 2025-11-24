import gymnasium as gym
import vizdoom.gymnasium_wrapper # Necesario para registrar los entornos
import time

ENV_NAME = "VizdoomCorridor-v0"

try:
    env = gym.make(ENV_NAME, render_mode='human')

    # Iniciar el primer episodio
    obs_dict, info = env.reset()
    print(obs_dict)
    
    # El array "gamevariables" solo tiene UN elemento. Lo extraemos.
    # Este único valor en "Corridor" es la Vida (AMMO).
    game_variables = obs_dict["gamevariables"]
    vida = int(game_variables[0]) # <-- Accedemos solo al primer elemento

    print("Iniciando simulación...")
    print(f"Espacio de acciones: {env.action_space}")
    print(f"Valores iniciales: Vida={vida}")

    # Bucle principal de la simulación
    for i in range(5000):
        action = env.action_space.sample()
        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        # Actualizamos la variable del juego en cada paso
        game_variables = obs_dict["gamevariables"]
        vida = int(game_variables[0]) # <-- Accedemos solo al primer elemento
        
        # Imprime la información actualizada
        if i % 20 == 0:
            # La vida y el blindaje no se proporcionan en las variables de este mapa.
            print(f"Paso {i}: Recompensa={reward:.2f}, Munición=N/A, Vida={vida}, Blindaje=N/A")

        # Si el episodio termina, se reinicia
        if terminated or truncated:
            print("Episodio terminado. Reiniciando...")
            obs_dict, info = env.reset()
            game_variables = obs_dict["gamevariables"]
            vida = int(game_variables[0]) # <-- Accedemos solo al primer elemento

        time.sleep(0.02)

finally:
    if 'env' in locals():
        env.close()
        print("Entorno cerrado.")
