import os
import neat
import numpy as np
import itertools as it
import vizdoom as vzd

from game import Game
from reporter import SaveBestGenomeReporter
from state_processor import StateProcessor

# --- 1. Initialization ---
SCENARIO_FILE = "deadly_corridor.cfg"
game = Game.initialize_doom(SCENARIO_FILE, render_window=False)

# Define as ações possíveis
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

# Cria o processador de estado (reutilizado do código anterior)
processor = StateProcessor(
    goal_name="GreenArmor", 
    enemy_names=["Zombieman", "ShotgunGuy", "HellKnight", "MarineChainsawVzd", "BaronBall", "Demon", "ChaingunGuy"])

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
        is_shooting = False 

        while not game.is_episode_finished():
            raw_state = game.get_state()
            # Supondo que as primeiras 2 game_variables são a posição X e Y
            player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
            player_health = game.get_game_variable(vzd.GameVariable.HEALTH)
            player_angle = game.get_game_variable(vzd.GameVariable.ANGLE)
            player_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)

            # Processa o estado para criar o vetor de entrada
            state_vector = processor.process(raw_state, player_pos, player_angle, player_health, player_ammo, is_shooting)

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
            is_shooting = attack
            
            reward = game.make_action(action)

            # --- LÓGICA DE RECOMPENSA CUSTOMIZADA ---
            custom_reward = 0.0
            
            # Extrai informações relevantes do vetor de estado
            enemy_is_present = state_vector[3]
            enemy_distance_normalized = state_vector[4] # Distância já normalizada
            crosshair_on_enemy = state_vector[6]

            # 1. Penalidade por inação em combate
            # Se um inimigo está visível e perto, penalize a passividade
            if enemy_is_present > 0 and enemy_distance_normalized > 0.4 and not is_shooting:
                custom_reward -= 0.5 

            # 2. Penalidade por desperdício de munição
            # Se atirou, mas não na direção do inimigo
            if is_shooting and not crosshair_on_enemy:
                custom_reward -= 1.0

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
    p.add_reporter(neat.Checkpointer(1)) # Salva um checkpoint a cada 5 gerações

    save_reporter = SaveBestGenomeReporter(SCENARIO_FILE, processor, 'src/genomes/gen')
    p.add_reporter(save_reporter)

    # Executa a evolução por até 100 gerações, usando nossa função de avaliação
    winner = p.run(eval_genomes, 100)

    # Mostra o melhor genoma encontrado
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.txt')
    run(config_path)
    game.close()