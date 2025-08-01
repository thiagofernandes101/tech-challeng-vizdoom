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
game = Game.initialize_doom(SCENARIO_FILE, render_window=True)

# Define as ações possíveis
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

# Cria o processador de estado (reutilizado do código anterior)
processor = StateProcessor(
    goal_name="GreenArmor", 
    enemy_names=["Zombieman", "ShotgunGuy", "HellKnight", "MarineChainsawVzd", "BaronBall", "Demon", "ChaingunGuy"])

# --- 2. A Função de Avaliação (O Coração da Integração com NEAT) ---

def eval_genomes(genomes, config):
    button_map = {button.name: i for i, button in enumerate(game.get_available_buttons())}
    action_template = [False] * game.get_available_buttons_size()

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game.new_episode()
        genome.fitness = 0

        is_shooting = False

        while not game.is_episode_finished():
            raw_state = game.get_state()
            if raw_state is None: break

            # Coleta de todas as variáveis de estado necessárias
            player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
            player_angle = game.get_game_variable(vzd.GameVariable.ANGLE)
            player_health = game.get_game_variable(vzd.GameVariable.HEALTH)
            ammo_count = game.get_game_variable(vzd.GameVariable.AMMO2)

            state_vector = processor.process(raw_state, player_pos, player_angle, player_health, ammo_count, is_shooting)
            
            output = net.activate(state_vector)
            
            action = list(action_template)
            do_turn_left = output[0] > 0.5
            do_turn_right = output[1] > 0.5
            do_move_forward = output[2] > 0.5
            do_attack = output[3] > 0.7
            
            action[button_map['TURN_LEFT']] = do_turn_left
            action[button_map['TURN_RIGHT']] = do_turn_right
            action[button_map['MOVE_FORWARD']] = do_move_forward
            action[button_map['ATTACK']] = do_attack

            # action = [turn_left, turn_right, attack, move_forward, False, False, False]
            is_shooting = do_attack

            reward = game.make_action(action)
            
            # --- LÓGICA DE RECOMPENSA CUSTOMIZADA ATUALIZADA ---
            custom_reward = 0.0
            
            enemy_is_present = state_vector[3]
            enemy_distance_normalized = state_vector[4]
            crosshair_on_enemy = state_vector[6]

            # 1. Penalidade por Inação (AUMENTADA)
            # Aumentamos a penalidade para tornar a passividade mais custosa.
            if enemy_is_present > 0 and enemy_distance_normalized > 0.4 and not is_shooting:
                custom_reward -= 1.0  # Era -0.5, agora é -1.0

            # 2. Penalidade por Desperdício de Munição (MANTIDA)
            if is_shooting and not crosshair_on_enemy:
                custom_reward -= 1.0

            # 3. BÔNUS DE AGRESSÃO PRECISA (NOVO)
            # Recompensa o agente por atirar ENQUANTO a mira está no inimigo.
            # Isso cria um forte incentivo para mirar e manter o fogo.
            if is_shooting and crosshair_on_enemy:
                custom_reward += 2.0 # Um bônus significativo por cada tiro preciso.
            
            # Verifica se o episódio terminou
            is_finished = game.is_episode_finished()
            if is_finished:
                # BÔNUS DE VITÓRIA COM COMBATE
                total_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                player_health = game.get_game_variable(vzd.GameVariable.HEALTH)
                
                # Se o agente sobreviveu E matou alguém, dê um grande bônus
                if player_health > 0 and total_kills > 0:
                    genome.fitness += 200.0 # Bônus substancial por vencer lutando

            genome.fitness += (reward + custom_reward)

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