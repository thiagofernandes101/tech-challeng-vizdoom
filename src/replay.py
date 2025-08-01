import os
import sys
import neat
import time
import pickle
import vizdoom as vzd
import numpy as np

from game import Game
from state_processor import StateProcessor

def replay_genome(config_path: str, genome_path: str):
    """
    Carrega um genoma salvo (.pkl) e o exibe jogando uma partida no ViZDoom.
    """
    print(f"Carregando genoma de: {genome_path}")

    # Carrega a configuração do NEAT, necessária para reconstruir a rede
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Carrega o objeto do genoma salvo
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    print("\nGenoma carregado com sucesso:")
    print(genome)

    # Cria a rede neural (o "cérebro") a partir do genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Inicializa o ambiente ViZDoom com a janela visível
    game = Game.initialize_doom("deadly_corridor.cfg", render_window=True)
    game.set_mode(vzd.Mode.SPECTATOR) # Modo espectador para assistir

    processor = StateProcessor(goal_name="GreenArmor", enemy_names=["Zombieman", "Imp"])
    
    print("\nIniciando apresentação...")
    
    for i in range(10): # Roda 3 episódios para demonstração
        game.new_episode()
        total_reward = 0.0
        is_shooting = False 
        
        while not game.is_episode_finished():
            raw_state = game.get_state()
            
            player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
            player_health = game.get_game_variable(vzd.GameVariable.HEALTH)
            player_angle = game.get_game_variable(vzd.GameVariable.ANGLE)
            player_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
            state_vector = processor.process(raw_state, player_pos, player_angle, player_health, player_ammo, is_shooting)
            output = net.activate(state_vector)

            turn_left = output[0] < -0.5
            turn_right = output[0] > 0.5
            move_forward = output[1] > 0.5
            attack = output[2] > 0.5
            action = [turn_left, turn_right, attack, move_forward, False, False, False]

            game.make_action(action)

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

            total_reward += game.get_last_reward()
            time.sleep(1.0 / 35.0) # Aproximadamente 35 FPS
        
        print(f"Episódio {i + 1} finalizado. Recompensa total: {total_reward:.2f}")
        time.sleep(2) # Pausa entre os episódios

    game.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python src/Replay.py <caminho_para_o_genoma.pkl>")
        sys.exit(1)
        
    genome_file = sys.argv[1]
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.txt')
    
    if not os.path.exists(genome_file):
        print(f"Erro: Arquivo de genoma '{genome_file}' não encontrado.")
        sys.exit(1)

    replay_genome(config_path, genome_file)