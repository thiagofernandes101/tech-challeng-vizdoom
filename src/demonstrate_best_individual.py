import vizdoom as vzd
import numpy as np
import time
import math
import os
import glob
import logging
from itertools import product
from NeuralNetwork import NeuralNetwork

# ==============================================================================
# LOGGER
# ==============================================================================
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(log_file, mode='a')
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
NN_INPUT_SIZE = 11
NN_HIDDEN_SIZE = 25

ENEMY_NAMES = {'Zombieman', 'ShotgunGuy'}
ENEMY_THREAT_LEVELS = { 'Zombieman': 1.0, 'ShotgunGuy': 3.0 }
ARMOR_POSITION = np.array([1312.0, 0.0])
ZONE_BOUNDS = [ (0, 450), (451, 900), (901, 1400) ]
MAX_AMMO = 52
LOW_HEALTH_THRESHOLD = 40.0

# ==============================================================================
# 2. FUNÇÕES AUXILIARES (IDÊNTICAS A main.py)
# ==============================================================================
def generate_action_space(game):
    buttons = game.get_available_buttons()
    button_indices = {button.name: i for i, button in enumerate(buttons)}
    num_buttons = len(buttons)
    conflict_groups = [
        {'MOVE_FORWARD', 'MOVE_BACKWARD'}, {'TURN_LEFT', 'TURN_RIGHT'}, {'MOVE_LEFT', 'MOVE_RIGHT'}
    ]
    independent_buttons = set(button_indices.keys())
    for group in conflict_groups:
        for button_name in group:
            independent_buttons.discard(button_name)
    options = []
    for group in conflict_groups:
        group_options = [()]
        for button_name in group:
            if button_name in button_indices:
                group_options.append((button_indices[button_name],))
        options.append(group_options)
    for button_name in independent_buttons:
        if button_name in button_indices:
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
    demo_logger.info(f"ESPAÇO DE AÇÕES TÁTICO GERADO: {len(final_actions)} ações.")
    return final_actions, button_indices

def initialize_game_for_visualization():
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
    player_x = player_pos[0]
    for i, (x_min, x_max) in enumerate(ZONE_BOUNDS):
        if x_min <= player_x <= x_max:
            return i
    return -1

def get_entities_and_zones(state, player_pos):
    enemies_by_zone = {i: [] for i in range(len(ZONE_BOUNDS))}
    enemies_aiming_count = 0
    all_enemies = []
    if state and state.labels:
        for label in state.labels:
            if label.object_name in ENEMY_NAMES:
                enemy_pos = np.array([label.object_position_x, label.object_position_y])
                enemy_zone = get_current_zone(enemy_pos)
                enemy_data = {'pos': enemy_pos, 'name': label.object_name}
                all_enemies.append(enemy_data)
                if enemy_zone != -1:
                    enemies_by_zone[enemy_zone].append(enemy_data)
                vec_enemy_to_player = player_pos - enemy_pos
                angle_enemy_to_player = math.degrees(math.atan2(vec_enemy_to_player[1], vec_enemy_to_player[0]))
                angle_diff = (label.object_angle - angle_enemy_to_player + 180) % 360 - 180
                if abs(angle_diff) < 15.0:
                    enemies_aiming_count += 1
    return all_enemies, enemies_by_zone, enemies_aiming_count

# ==============================================================================
# 3. FUNÇÃO DE VISUALIZAÇÃO
# ==============================================================================
def run_visualization(game, agent_nn, actions):
    demo_logger.info("Iniciando visualização...")
    game.new_episode()
    cleared_zones = [False] * len(ZONE_BOUNDS)

    while not game.is_episode_finished():
        state = game.get_state()
        if state is None: continue
        
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        current_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
        all_enemies, enemies_by_zone, enemies_aiming_count = get_entities_and_zones(state, player_pos)
        is_being_aimed_at = 1.0 if enemies_aiming_count > 0 else 0.0
        
        player_zone = get_current_zone(player_pos)
        is_combat_mode = len(all_enemies) > 0
        is_low_health = 1.0 if current_health < LOW_HEALTH_THRESHOLD else 0.0
        
        # A decisão do objetivo tático agora ocorre ANTES de qualquer atualização de 'cleared_zones' neste ciclo.
        tactical_objective = ARMOR_POSITION
        active_threats_in_zone = []
        if player_zone != -1:
            active_threats_in_zone = enemies_by_zone[player_zone]

        if is_low_health and player_zone > 0 and active_threats_in_zone:
            prev_zone_x_min, prev_zone_x_max = ZONE_BOUNDS[player_zone - 1]
            tactical_objective = np.array([(prev_zone_x_min + prev_zone_x_max) / 2, 0.0])
        elif player_zone != -1 and not cleared_zones[player_zone] and active_threats_in_zone:
            highest_priority_score = -1; best_target = None
            for enemy in active_threats_in_zone:
                dist = np.linalg.norm(enemy['pos'] - player_pos)
                threat = ENEMY_THREAT_LEVELS.get(enemy['name'], 1.0)
                priority = threat / (dist + 1e-6)
                if priority > highest_priority_score:
                    highest_priority_score = priority; best_target = enemy
            if best_target: tactical_objective = best_target['pos']
        else:
            next_zone_found = False
            for i in range(len(ZONE_BOUNDS)):
                if not cleared_zones[i]:
                    zone_x_min, zone_x_max = ZONE_BOUNDS[i]
                    tactical_objective = np.array([(zone_x_min + zone_x_max) / 2, 0.0])
                    next_zone_found = True
                    break
            if not next_zone_found: tactical_objective = ARMOR_POSITION

        dist_to_objective, angle_to_objective = 0.0, 0.0
        if tactical_objective is not None:
            delta = tactical_objective - player_pos
            dist_to_objective = np.linalg.norm(delta)
            target_angle = math.degrees(math.atan2(delta[1], delta[0]))
            angle_diff = (game.get_game_variable(vzd.GameVariable.ANGLE) - target_angle + 180) % 360 - 180
            angle_to_objective = angle_diff / 180.0
        
        nn_input = [
            current_health / 100.0, 
            current_ammo / MAX_AMMO, 
            1.0 if is_combat_mode else 0.0,
            dist_to_objective / 1000.0, 
            angle_to_objective,
            float(cleared_zones[0]), 
            float(cleared_zones[1]), 
            float(cleared_zones[2]),
            is_being_aimed_at, 
            len(active_threats_in_zone) / 2.0, 
            is_low_health
        ]

        action_scores = agent_nn.forward(nn_input)
        action_index = np.argmax(action_scores)
        
        action = actions[action_index]
        game.make_action(action, FRAME_SKIP)
        
        # A atualização de 'cleared_zones' agora ocorre DEPOIS da ação, espelhando o treino.
        if player_zone != -1 and not cleared_zones[player_zone] and not enemies_by_zone[player_zone]:
             cleared_zones[player_zone] = True
        
        time.sleep(0.028)

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
    nn_config = (NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE)
    agent_network = NeuralNetwork(*nn_config)
    
    try:
        agent_network.set_weights_from_flat(best_genome)
    except ValueError as e:
        demo_logger.error(f"ERRO DE INCOMPATIBILIDADE DE GENOMA: {e}")
        demo_logger.error("Isso geralmente significa que a arquitetura da rede (NN_INPUT/HIDDEN/OUTPUT_SIZE) no script de demonstração não corresponde à do script de treinamento que salvou o genoma.")
        game.close()
        exit()
    
    try:
        v_counter = 1
        while v_counter <= 100:
            run_visualization(game, agent_network, possible_actions)
            demo_logger.info("\nReiniciando em 3 segundos...")
            time.sleep(3)
            v_counter += 1
    except Exception as e:
        demo_logger.error(f"Ocorreu um erro durante a visualização: {e}")
    finally:
        game.close()