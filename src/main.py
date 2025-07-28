import vizdoom as vzd
import numpy as np
import random
import time
import math
import logging
from NeuralNetwork import NeuralNetwork

# ==============================================================================
# LOGGER
# ==============================================================================
def setup_logger(name, log_file, level=logging.INFO):
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

training_logger = setup_logger('training_logger', 'training_log.txt')

# ==============================================================================
# 1. PARÂMETROS GLOBAIS (FILOSOFIA "SOBREVIVENTE PRIORITÁRIO")
# ==============================================================================
POPULATION_SIZE = 150
SCENARIO_PATH = "deadly_corridor.cfg"
FRAME_SKIP = 4
NN_INPUT_SIZE = 9
NN_HIDDEN_SIZE = 16

ENEMY_NAMES = {'Zombieman', 'ShotgunGuy'}
ENEMY_THREAT_LEVELS = { 'Zombieman': 1.0, 'ShotgunGuy': 3.0 }
ARMOR_POSITION = np.array([1312.0, 0.0])
ZONE_BOUNDS = [ (0, 450), (451, 900), (901, 1400) ]
MAX_AMMO = 52

DEATH_PENALTY_BASE = -6000.0
DEATH_PENALTY_ZONE_REDUCTION = 1500.0
LEVEL_COMPLETION_BONUS = 15000.0
W_ARMOR_PICKUP_ALONE = 500.0
W_LEVEL_INCOMPLETE_PENALTY = -3000.0
W_ZONE_CLEAR_BONUSES = [ 2000.0, 3000.0, 4000.0 ]
W_PROGRESS_TOWARDS_OBJECTIVE = 150.0
W_COMBAT_SURVIVAL_BONUS = 2.0

W_DAMAGE_DEALT_BONUS = 30.0
W_AMMO_EFFICIENCY_BONUS = 200.0
W_AMMO_CONSERVED_BONUS = 15.0

W_HEALTH_PRESERVATION_BONUS = 25.0
W_COMBAT_DAMAGE_PENALTY = -25.0

W_BEING_AIMED_AT_PENALTY = -30.0
W_TIME_PENALTY = -1.0
W_WASTED_SHOT_PENALTY = -20.0
W_STAGNATION_PENALTY = -100.0
STAGNATION_TICKS_THRESHOLD = 40
STAGNATION_DISTANCE_THRESHOLD = 15.0

# --- Parâmetros Genéticos (Com Mutação Decrescente) ---
ELITISM_COUNT = 2
TOURNAMENT_SIZE = 3
STAGNATION_LIMIT = 80

MAX_MUTATION_RATE = 0.10  # Começa com alta exploração
MIN_MUTATION_RATE = 0.01  # Termina com baixo refinamento
MUTATION_DECAY_GENERATIONS = 150 # Número de gerações para a taxa cair do máximo ao mínimo

# ==============================================================================
# 2. FUNÇÕES AUXILIARES E DE AVALIAÇÃO
# ==============================================================================
def generate_action_space(game):
    actions = [
        [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]
    ]
    buttons = game.get_available_buttons()
    button_indices = {button.name: i for i, button in enumerate(buttons)}
    button_map = { 'ATTACK': button_indices.get('ATTACK'), 'MOVE_FORWARD': button_indices.get('MOVE_FORWARD') }
    training_logger.info(f"ESPAÇO DE AÇÕES SIMPLIFICADO: {len(actions)} ações.")
    return actions, button_map

def initialize_game():
    training_logger.info("Inicializando instância única do ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_doom_skill(3)
    game.init()
    _, button_map = generate_action_space(game)
    return game, button_map

def create_initial_population(genome_length):
    return [{"genome": (np.random.rand(genome_length) * 2 - 1) * 0.5, "fitness": None} for _ in range(POPULATION_SIZE)]

def get_current_zone(player_pos):
    player_x = player_pos[0]
    for i, (x_min, x_max) in enumerate(ZONE_BOUNDS):
        if x_min <= player_x <= x_max:
            return i
    return -1

def get_entities_and_zones(state, player_pos):
    enemies_by_zone = {i: [] for i in range(len(ZONE_BOUNDS))}
    is_targeted = 0.0
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
                    is_targeted = 1.0
    return all_enemies, enemies_by_zone, is_targeted

def evaluate_individual(genome, game, actions, b_map, nn_config):
    input_size, hidden_size, output_size = nn_config
    agent_nn = NeuralNetwork(input_size, hidden_size, output_size)
    agent_nn.set_weights_from_flat(genome)
    game.new_episode()
    last_health = game.get_game_variable(vzd.GameVariable.HEALTH)
    last_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
    total_reward = 0.0
    cleared_zones = [False] * len(ZONE_BOUNDS)
    last_dist_to_objective = None
    max_ticks = 2100 
    for _ in range(max_ticks):
        if game.is_episode_finished(): break
        state = game.get_state()
        if state is None: continue
        player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
        all_enemies, enemies_by_zone, is_being_aimed_at = get_entities_and_zones(state, player_pos)
        player_zone = get_current_zone(player_pos)
        is_combat_mode = len(all_enemies) > 0
        tactical_objective = ARMOR_POSITION
        if player_zone != -1 and not cleared_zones[player_zone] and enemies_by_zone[player_zone]:
            highest_priority_score = -1; best_target = None
            for enemy in enemies_by_zone[player_zone]:
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
            last_health / 100.0, last_ammo / 52.0, 1.0 if is_combat_mode else 0.0,
            dist_to_objective / 1000.0, angle_to_objective,
            float(cleared_zones[0]), float(cleared_zones[1]), float(cleared_zones[2]),
            is_being_aimed_at
        ]
        action_scores = agent_nn.forward(nn_input)
        action_index = np.argmax(action_scores)
        total_reward += W_TIME_PENALTY
        damage_dealt_before_action = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        if last_dist_to_objective is not None:
            progress = last_dist_to_objective - dist_to_objective
            total_reward += W_PROGRESS_TOWARDS_OBJECTIVE * progress
        last_dist_to_objective = dist_to_objective
        if is_being_aimed_at > 0: total_reward += W_BEING_AIMED_AT_PENALTY
        if is_combat_mode: total_reward += W_COMBAT_SURVIVAL_BONUS
        if player_zone != -1 and not cleared_zones[player_zone] and not enemies_by_zone[player_zone]:
            cleared_zones[player_zone] = True
            bonus = W_ZONE_CLEAR_BONUSES[player_zone]
            total_reward += bonus
            training_logger.info(f"  -> ZONA {player_zone + 1} LIMPA! Bônus: +{bonus}")
        game.make_action(actions[action_index], FRAME_SKIP)
        damage_delta = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT) - damage_dealt_before_action
        if damage_delta > 0: total_reward += W_DAMAGE_DEALT_BONUS * damage_delta
        current_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        if current_ammo < last_ammo and damage_delta > 0:
            total_reward += W_AMMO_EFFICIENCY_BONUS
        elif current_ammo < last_ammo and damage_delta == 0:
             urgency_multiplier = 1 + (MAX_AMMO - current_ammo) / MAX_AMMO
             total_reward += W_WASTED_SHOT_PENALTY * urgency_multiplier
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        health_lost = last_health - current_health
        if health_lost > 0:
            urgency_multiplier = 1 + (100 - current_health) / 100.0
            total_reward += (W_COMBAT_DAMAGE_PENALTY * health_lost) * urgency_multiplier
        last_health = current_health
        last_ammo = current_ammo
    if game.is_player_dead():
        num_zones_cleared = sum(cleared_zones)
        death_penalty_final = DEATH_PENALTY_BASE + (num_zones_cleared * DEATH_PENALTY_ZONE_REDUCTION)
        total_reward += death_penalty_final
        training_logger.info(f"  -> Agente morreu após limpar {num_zones_cleared} zonas. Penalidade: {death_penalty_final:.2f}")
    elif game.is_episode_finished():
        final_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        final_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        total_reward += W_HEALTH_PRESERVATION_BONUS * final_health
        total_reward += W_AMMO_CONSERVED_BONUS * final_ammo
        if all(cleared_zones):
            training_logger.info(f"  -> MISSÃO CUMPRIDA! Bônus de conclusão concedido.")
            total_reward += LEVEL_COMPLETION_BONUS
        else:
            if np.linalg.norm(player_pos - ARMOR_POSITION) < 50:
                 total_reward += W_ARMOR_PICKUP_ALONE
            training_logger.info(f"  -> MISSÃO INCOMPLETA! Penalidade aplicada.")
            total_reward += W_LEVEL_INCOMPLETE_PENALTY
    return total_reward

# ==============================================================================
# 3. ALGORITMO GENÉTICO
# ==============================================================================
def tournament_selection(population):
    return max(random.sample(population, TOURNAMENT_SIZE), key=lambda x: x['fitness'])

def two_point_crossover(p1_genome, p2_genome):
    genome_len = len(p1_genome);
    if genome_len < 3: return p1_genome.copy(), p2_genome.copy()
    p1 = random.randint(1, genome_len - 1); p2 = random.randint(1, genome_len - 1)
    start_point, end_point = min(p1, p2), max(p1, p2)
    if start_point == end_point:
        if end_point < genome_len - 1: end_point += 1
        else: start_point -=1
    p1_list, p2_list = list(p1_genome), list(p2_genome)
    child1_genome = np.array(p1_list[:start_point] + p2_list[start_point:end_point] + p1_list[end_point:])
    child2_genome = np.array(p2_list[:start_point] + p1_list[start_point:end_point] + p2_list[end_point:])
    return child1_genome, child2_genome

def mutate(genome, rate):
    mutated_genome = genome.copy()
    for i in range(len(mutated_genome)):
        if random.random() < rate: mutated_genome[i] += random.gauss(0, 0.2)
    return mutated_genome
    
# ALTERADO: A função generate_new_population não precisa mais da lógica de hipermutação
def generate_new_population(old_pop, mut_rate):
    sorted_pop = sorted(old_pop, key=lambda x: x['fitness'], reverse=True)
    new_pop = [sorted_pop[i] for i in range(ELITISM_COUNT)]
    while len(new_pop) < POPULATION_SIZE:
        p1, p2 = tournament_selection(sorted_pop), tournament_selection(sorted_pop)
        c1_g, c2_g = two_point_crossover(p1['genome'], p2['genome'])
        new_pop.append({'genome': mutate(c1_g, mut_rate), 'fitness': None})
        if len(new_pop) < POPULATION_SIZE:
            new_pop.append({'genome': mutate(c2_g, mut_rate), 'fitness': None})
    return new_pop
    
# ==============================================================================
# 4. LOOP PRINCIPAL DE EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    game_instance, button_map_for_rewards = initialize_game()
    possible_actions, _ = generate_action_space(game_instance)
    num_possible_actions = len(possible_actions)
    training_logger.info(f"Número de ações possíveis detectado: {num_possible_actions}")
    NN_OUTPUT_SIZE = num_possible_actions
    nn_config = (NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE)
    temp_nn = NeuralNetwork(*nn_config)
    genome_length = temp_nn.total_weights
    training_logger.info(f"Arquitetura da Rede: {NN_INPUT_SIZE} -> {NN_HIDDEN_SIZE} -> {NN_OUTPUT_SIZE}")
    training_logger.info(f"Tamanho do Genoma (Pesos): {genome_length}")
    
    current_population = create_initial_population(genome_length)
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    generation = 0
    
    while True:
        training_logger.info(f"\n{'='*20} GERAÇÃO {generation} {'='*20}")

        # ALTERADO: Lógica de mutação decrescente
        decay_progress = min(1.0, generation / MUTATION_DECAY_GENERATIONS)
        current_mutation_rate = MAX_MUTATION_RATE - (MAX_MUTATION_RATE - MIN_MUTATION_RATE) * decay_progress
        training_logger.info(f"Taxa de Mutação para esta Geração: {current_mutation_rate:.4f}")

        start_time = time.time()
        for ind in current_population:
            if ind['fitness'] is None:
                ind['fitness'] = evaluate_individual(ind['genome'], game_instance, possible_actions, button_map_for_rewards, nn_config)
        
        eval_time = time.time() - start_time
        training_logger.info(f"Avaliação de {POPULATION_SIZE} indivíduos concluída em {eval_time:.2f}s.")
        
        sorted_population = sorted(current_population, key=lambda x: x['fitness'], reverse=True)
        current_best_fitness = sorted_population[0]['fitness']
        training_logger.info(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")
        
        if current_best_fitness > best_fitness_overall + 1.0: # Um limiar de melhoria mínimo
            best_fitness_overall = current_best_fitness
            generations_without_improvement = 0
            training_logger.info(f"✨ Nova melhoria significativa! Melhor fitness geral: {best_fitness_overall:.2f}")
            np.save(f'best_genome_gen_{generation}.npy', sorted_population[0]['genome'])
        else:
            generations_without_improvement += 1
            
        training_logger.info(f"Gerações estagnadas: {generations_without_improvement}/{STAGNATION_LIMIT}")
        
        if generations_without_improvement >= STAGNATION_LIMIT:
            training_logger.info(f"CRITÉRIO DE PARADA ATINGIDO: Limite de estagnação.")
            break
            
        current_population = generate_new_population(sorted_population, current_mutation_rate)
        generation += 1
        
    training_logger.info("\n" + "="*50 + "\nEvolução finalizada.\n" + "="*50)
    game_instance.close()