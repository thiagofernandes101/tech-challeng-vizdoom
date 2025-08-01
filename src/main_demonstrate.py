import vizdoom as vzd
import numpy as np
import time
import math
import os
import glob
import logging
from models.NeuralNetwork import NeuralNetwork

# ==============================================================================
# LOGGER
# ==============================================================================
def setup_logger(name, log_file, level=logging.INFO):
    """Configura um logger para exibir mensagens no console e em um arquivo."""
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger

demo_logger = setup_logger('demo_logger', 'demonstration_log.txt')

# ==============================================================================
# 1. PARÂMETROS GLOBAIS (SINCRONIZADOS COM main.py)
# ==============================================================================
SCENARIO_PATH = "deadly_corridor.cfg"
FRAME_SKIP = 4

# ATUALIZADO: Parâmetros para a arquitetura com LSTM
NN_INPUT_SIZE = 8       # Tamanho do "Vetor de Estado de Nível"
LSTM_HIDDEN_SIZE = 128   # Tamanho da memória da LSTM
NN_HIDDEN_SIZE = LSTM_HIDDEN_SIZE # Para consistência
W_KILL_BONUS = 750.0

ENEMY_NAMES = {'Zombieman', 'ShotgunGuy'}
ENEMY_THREAT_LEVELS = { 'Zombieman': 1.0, 'ShotgunGuy': 3.0 }
ARMOR_POSITION = np.array([1312.0, 0.0])
ZONE_BOUNDS = [ (0, 450), (451, 900), (901, 1400) ]
MAX_AMMO = 52
TARGET_KILLS_PER_ZONE = 2

# ==============================================================================
# 2. FUNÇÕES AUXILIARES (IDÊNTICAS A main.py)
# ==============================================================================
def generate_action_space(game):
    buttons = game.get_available_buttons()
    button_indices = {button.name: i for i, button in enumerate(buttons)}
    num_buttons = len(buttons)

    action_templates = [
        {'MOVE_FORWARD'}, {'MOVE_BACKWARD'}, {'TURN_RIGHT'}, {'TURN_LEFT'},
        {'MOVE_RIGHT'}, {'MOVE_LEFT'}, {'ATTACK'}, {'ATTACK', 'MOVE_BACKWARD'},
        {'ATTACK', 'MOVE_RIGHT'}, {'ATTACK', 'MOVE_LEFT'},
    ]

    final_actions = []
    for template in action_templates:
        action_list = [0] * num_buttons
        for button_name in template:
            if button_name in button_indices:
                action_list[button_indices[button_name]] = 1
        final_actions.append(action_list)

    demo_logger.info(f"ESPAÇO DE AÇÕES TÁTICO FINAL GERADO: {len(final_actions)} ações.")
    return final_actions, button_indices

def initialize_game_for_visualization():
    """Inicializa o jogo em modo de visualização para o jogador."""
    demo_logger.info("Inicializando ViZDoom para visualização...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_doom_skill(1)
    game.init()
    return game

def get_current_zone(player_pos):
    """Determina em qual zona de combate o jogador está."""
    player_x = player_pos[0]
    for i, (x_min, x_max) in enumerate(ZONE_BOUNDS):
        if x_min <= player_x <= x_max:
            return i
    return -1

def get_entities_and_zones(state, player_pos):
    enemies_by_zone = {i: [] for i in range(len(ZONE_BOUNDS))}
    all_enemies = []
    if state and state.labels:
        for label in state.labels:
            if label.object_name in ENEMY_NAMES:
                enemy_pos = np.array([label.object_position_x, label.object_position_y])
                enemy_zone = get_current_zone(enemy_pos)
                enemy_data = {'pos': enemy_pos, 'name': label.object_name, 'id': label.object_id}
                all_enemies.append(enemy_data)
                if enemy_zone != -1:
                    enemies_by_zone[enemy_zone].append(enemy_data)
    return all_enemies, enemies_by_zone

# ==============================================================================
# 3. FUNÇÃO DE VISUALIZAÇÃO (ATUALIZADA PARA LSTM)
# ==============================================================================
def run_visualization(game, agent_nn, actions):
    """Executa um episódio de demonstração usando o genoma treinado."""
    demo_logger.info("Iniciando visualização...")

    game.new_episode()
    cleared_zones = [False] * len(ZONE_BOUNDS)
    zone_kill_counts = [0] * len(ZONE_BOUNDS)
    last_kill_count = 0

    # NOVO: Inicializa a memória da LSTM
    hidden_state = agent_nn.init_hidden()
    
    max_ticks = 2100
    for _ in range(max_ticks):
        if game.is_episode_finished(): break
        state = game.get_state()
        if state is None: continue
        
        # --- Coleta de dados brutos do jogo ---
        player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
        player_angle = game.get_game_variable(vzd.GameVariable.ANGLE)
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        current_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        all_enemies, enemies_by_zone = get_entities_and_zones(state, player_pos)
        
        # --- Lógica de Combate e Navegação (simplificada) ---
        is_combat_mode = False
        closest_enemy_dist = float('inf')
        current_target = None
        if all_enemies:
            for enemy in all_enemies:
                dist = np.linalg.norm(enemy['pos'] - player_pos)
                if dist < closest_enemy_dist:
                    closest_enemy_dist = dist
                    current_target = enemy
            if closest_enemy_dist < 600:
                 is_combat_mode = True

        # ==================================================================
        # NOVO: Construção do "Vetor de Estado de Nível" (nn_input)
        # ==================================================================
        nn_input = []

        # --- Bloco 1: Estado Global e Tático do Jogador (8 entradas) ---
        dist_ao_alvo_norm = 0.0
        angulo_ao_alvo_norm = 0.0
        if current_target:
            delta_alvo = current_target['pos'] - player_pos
            dist_ao_alvo_norm = np.linalg.norm(delta_alvo) / 1000.0 # Normaliza
            target_angle_alvo = math.degrees(math.atan2(delta_alvo[1], delta_alvo[0]))
            angulo_ao_alvo_norm = ((player_angle - target_angle_alvo + 180) % 360 - 180) / 180.0

        delta_obj_final = ARMOR_POSITION - player_pos
        dist_obj_final_norm = np.linalg.norm(delta_obj_final) / 1500.0 # Normaliza
        target_angle_obj_final = math.degrees(math.atan2(delta_obj_final[1], delta_obj_final[0]))
        angulo_obj_final_norm = ((player_angle - target_angle_obj_final + 180) % 360 - 180) / 180.0

        nn_input.extend([
            current_health / 100.0,
            current_ammo / MAX_AMMO,
            1.0 if is_combat_mode else 0.0,
            len(all_enemies) / 6.0, # Contagem total de inimigos visíveis (normalizada)
            dist_ao_alvo_norm,
            angulo_ao_alvo_norm,
            dist_obj_final_norm,
            angulo_obj_final_norm,
        ])

        nn_input = np.array(nn_input)
        # ==================================================================

        # NOVO: A chamada para a rede agora gerencia o estado da memória
        action_scores, hidden_state = agent_nn.forward(nn_input, hidden_state)
        action_index = np.argmax(action_scores)
        game.make_action(actions[action_index], FRAME_SKIP)

        time.sleep(0.028) # Pausa para melhor visualização

    # Log do final do episódio
    demo_logger.info(f"Episódio finalizado. Zonas Limpas: {cleared_zones}")
    demo_logger.info(f"Total de Kills: {game.get_game_variable(vzd.GameVariable.KILLCOUNT)}")
    demo_logger.info(f"Vida Restante: {game.get_game_variable(vzd.GameVariable.HEALTH)}")

# ==============================================================================
# 4. BLOCO PRINCIPAL DE EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    list_of_files = glob.glob('best_genome_gen_*.npy')
    if not list_of_files:
        demo_logger.info("ERRO: Nenhum arquivo de genoma salvo ('best_genome_gen_*.npy') foi encontrado.")
        exit()
        
    latest_file = max(list_of_files, key=os.path.getctime)
    demo_logger.info(f"\n{'='*20}\nCarregando o melhor genoma de: {latest_file}\n{'='*20}")
    
    best_genome = np.load(latest_file)

    game = initialize_game_for_visualization()
    possible_actions, _ = generate_action_space(game)
    
    NN_OUTPUT_SIZE = len(possible_actions)
    
    # ATUALIZADO: Configuração da rede para LSTM
    nn_config = (NN_INPUT_SIZE, LSTM_HIDDEN_SIZE, NN_OUTPUT_SIZE)
    agent_network = NeuralNetwork(*nn_config)
    
    try:
        agent_network.set_weights_from_flat(best_genome)
    except ValueError as e:
        demo_logger.error(f"ERRO DE INCOMPATIBILIDADE DE GENOMA: {e}")
        demo_logger.error("Isso geralmente significa que a arquitetura da rede (NN_INPUT_SIZE, LSTM_HIDDEN_SIZE) no script de demonstração não corresponde à do script de treinamento que salvou o genoma.")
        game.close()
        exit()
    
    try:
        while True:
            run_visualization(game, agent_network, possible_actions)
            demo_logger.info("\nReiniciando em 3 segundos...")
            time.sleep(3)
    except Exception as e:
        demo_logger.error(f"Ocorreu um erro durante a visualização: {e}")
    finally:
        game.close()