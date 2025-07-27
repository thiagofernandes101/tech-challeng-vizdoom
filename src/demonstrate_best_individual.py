import vizdoom as vzd
import numpy as np
import time
from itertools import product

# --- Configuração do ViZDoom ---
SCENARIO_PATH = "deadly_corridor.cfg"  # Nome do arquivo de configuração do cenário
GENOME_LENGTH = 3000  # Número máximo de ações por episódio

def generate_action_space(game):
    # ... (código inalterado)
    buttons = game.get_available_buttons()
    button_indices = {button.name: i for i, button in enumerate(buttons)}
    num_buttons = len(buttons)
    conflict_groups = [
        {'MOVE_FORWARD', 'MOVE_BACKWARD'},
        {'TURN_LEFT', 'TURN_RIGHT'},
        {'MOVE_LEFT', 'MOVE_RIGHT'}
    ]
    options = []
    independent_buttons = set(button_indices.keys())
    for group in conflict_groups:
        group_options = [()]
        for button_name in group:
            if button_name in button_indices:
                group_options.append((button_indices[button_name],))
                independent_buttons.discard(button_name)
        options.append(group_options)
    for button_name in independent_buttons:
        options.append([(), (button_indices[button_name],)])
    action_combinations = list(product(*options))
    final_actions = []
    for combo in action_combinations:
        action_list = [0] * num_buttons
        for button_set in combo:
            for button_index in button_set:
                action_list[button_index] = 1
        if any(action_list):
            final_actions.append(action_list)
    print(f"Espaço de ações gerado com {len(final_actions)} ações complexas.")
    return final_actions

# Função para inicializar o jogo
def initialize_game():
    print("Inicializando ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)

    # Roda o jogo com a janela visível para demonstração
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
    game.set_doom_skill(1)
    game.init()

    # Define as ações possíveis que o agente pode tomar
    actions = generate_action_space(game)
    return game, actions

# Função para demonstrar o melhor indivíduo
def demonstrate_best_individual():
    # Carrega o genoma do melhor indivíduo salvo
    best_genome = np.load('best_genome.npy')

    # Inicializa o jogo e as ações possíveis
    game, actions = initialize_game()

    print("Iniciando demonstração do melhor indivíduo...")
    game.new_episode()

    for action_index in best_genome:
        if game.is_episode_finished():
            break

        # Executa a ação correspondente ao índice
        action_to_perform = actions[action_index]
        game.make_action(action_to_perform)

        # Aguarda um pequeno intervalo para visualização
        time.sleep(1.0 / 60.0)

    print("Demonstração concluída.")
    game.close()

if __name__ == "__main__":
    while True:
        demonstrate_best_individual()
