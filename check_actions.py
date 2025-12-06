# check_actions.py
import vizdoom as vzd

def check_actions(scenario_path):
    print(f"--- Analizando: {scenario_path} ---")
    game = vzd.DoomGame()
    
    try:
        game.load_config(scenario_path)
    except Exception as e:
        print(f"Error cargando el config: {e}")
        return

    # Inicializar para cargar todas las variables
    game.init()

    # Obtener botones disponibles
    available_buttons = game.get_available_buttons()
    
    print(f"Número total de acciones posibles: {len(available_buttons)}")
    print("Lista de botones activados:")
    for button in available_buttons:
        print(f" - {button}")
    
    game.close()

if __name__ == "__main__":
    # Comprobamos el nivel 3
    check_actions("deadly_corridor.cfg")