import vizdoom as vzd
import numpy as np
import time
from itertools import product

# --- Configuração do ViZDoom ---
SCENARIO_PATH = "deadly_corridor.cfg"  # Nome do arquivo de configuração do cenário
GENOME_LENGTH = 500  # Número máximo de ações por episódio

# Função para inicializar o jogo
def initialize_game():
    print("Inicializando ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)

    # Roda o jogo com a janela visível para demonstração
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_available_buttons([
        vzd.Button.ATTACK,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT
    ])
    game.init()

    # Define as ações possíveis que o agente pode tomar
    num_buttons = game.get_available_buttons_size()
    actions = [list(p) for p in product([0, 1], repeat=num_buttons)]
    actions.remove([0] * num_buttons)  # Remove a ação "não fazer nada"

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
        time.sleep(1.0 / 35.0)

    print("Demonstração concluída.")
    game.close()

if __name__ == "__main__":
    demonstrate_best_individual()
