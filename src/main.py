import os
import neat
import numpy as np
import itertools as it
import vizdoom as vzd

from Game import Game
from StateProcessorLstm import StateProcessor

# --- 1. Initialization ---
game = Game.initialize_doom("deadly_corridor.cfg", render_window=False)

# Define as ações possíveis
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

# Cria o processador de estado (reutilizado do código anterior)
processor = StateProcessor(goal_name="GreenArmor", enemy_names=["Zombieman", "Imp"])

# --- 2. A Função de Avaliação (O Coração da Integração com NEAT) ---

def eval_genomes(genomes, config):
    """
    Roda uma simulação para cada genoma na população para avaliar sua aptidão.
    A aptidão (fitness) é a recompensa total acumulada no episódio.
    """
    for genome_id, genome in genomes:
        # Cria a rede neural a partir do genoma
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Inicia um novo episódio para este genoma
        game.new_episode()
        genome.fitness = 0  # Inicia a aptidão como zero

        while not game.is_episode_finished():
            raw_state = game.get_state()
            # Supondo que as primeiras 2 game_variables são a posição X e Y
            player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
            player_health = game.get_game_variable(vzd.GameVariable.HEALTH)

            # Processa o estado para criar o vetor de entrada
            state_vector = processor.process(raw_state, player_pos, player_health)

            # Alimenta a rede com o estado para obter as saídas
            output = net.activate(state_vector)

            # --- Traduz a saída da rede em ações do jogo ---
            # output[0]: Controle de virada (-1.0 a 1.0)
            # output[1]: Controle de movimento (-1.0 a 1.0)
            # output[2]: Controle de tiro (> 0.5 para atirar)
            
            turn_left = output[0] < -0.5
            turn_right = output[0] > 0.5
            move_forward = output[1] > 0.5
            attack = output[2] > 0.5
            
            # Os outros botões (como move_backward, strafe) são mantidos como 0
            action = [turn_left, turn_right, attack, move_forward, False, False, False]
            
            reward = game.make_action(action)
            genome.fitness += reward

        print(f"Genome ID: {genome_id} Fitness: {genome.fitness}")


# --- 3. O Orquestrador Principal do NEAT ---
def run(config_file):
    """
    Configura e executa o algoritmo NEAT.
    """
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Cria a população
    p = neat.Population(config)

    # Adiciona reporters para mostrar o progresso no console
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5)) # Salva um checkpoint a cada 5 gerações

    # Executa a evolução por até 100 gerações, usando nossa função de avaliação
    winner = p.run(eval_genomes, 100)

    # Mostra o melhor genoma encontrado
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.txt')
    run(config_path)
    game.close()