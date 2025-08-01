import vizdoom as vzd
import numpy as np
import random
import time
import math
import logging
from itertools import product
from models.NeuralNetwork import NeuralNetwork

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
# 1. PARÂMETROS GLOBAIS (FILOSOFIA "CONSCIÊNCIA ESTRATÉGICA")
# ==============================================================================
POPULATION_SIZE = 300
SCENARIO_PATH = "deadly_corridor.cfg"
FRAME_SKIP = 4

# ALTERADO: Parâmetros para a nova arquitetura com LSTM
NN_INPUT_SIZE = 8       # Tamanho do nosso novo "Vetor de Estado de Nível"
LSTM_HIDDEN_SIZE = 128   # Tamanho da memória da LSTM
NN_HIDDEN_SIZE = LSTM_HIDDEN_SIZE # Para consistência de nome, se necessário em algum lugar

IDEAL_COMBAT_DISTANCE = 450.0
MIN_COMBAT_DISTANCE = 150.0

ENEMY_NAMES = {'Zombieman', 'ShotgunGuy'}
ENEMY_THREAT_LEVELS = { 'Zombieman': 1.0, 'ShotgunGuy': 3.0 }
ARMOR_POSITION = np.array([1312.0, 0.0])
ZONE_BOUNDS = [ (0, 450), (451, 900), (901, 1400) ]
MAX_AMMO = 52
LOW_HEALTH_THRESHOLD = 40.0

TARGET_KILLS_PER_ZONE = 2

# NOVO: Parâmetros para o sistema de "Target Lock"
TARGET_LOCK_DURATION_TICKS = 60 # Quantos ticks o agente fica focado em um alvo

DEATH_PENALTY_BASE = -7000.0 
DEATH_PENALTY_ZONE_REDUCTION = 1500.0
W_SURVIVAL_BONUS = 3000.0
LEVEL_COMPLETION_BONUS = 15000.0
W_ARMOR_PICKUP_ALONE = 500.0
W_LEVEL_INCOMPLETE_PENALTY = -3000.0
W_ZONE_CLEAR_BONUSES = [ 2000.0, 3000.0, 4000.0 ]
W_PROGRESS_TOWARDS_OBJECTIVE = 150.0
W_COMBAT_SURVIVAL_BONUS = 3.0
W_HEALTH_PRESERVATION_BONUS = 35.0
W_DAMAGE_DEALT_BONUS = 50.0
W_AMMO_EFFICIENCY_BONUS = 200.0
W_AMMO_CONSERVED_BONUS = 15.0
W_COMBAT_DAMAGE_PENALTY = -60.0
W_WASTED_SHOT_PENALTY = -30.0
W_BEING_AIMED_AT_PENALTY = -30.0
W_MULTIPLE_THREATS_PENALTY = -40.0
W_TARGET_ISOLATION_BONUS = 25.0
W_TIME_PENALTY = -1.0
W_STAGNATION_PENALTY = -100.0
STAGNATION_TICKS_THRESHOLD = 40
STAGNATION_DISTANCE_THRESHOLD = 15.0
W_AIM_BONUS = 40.0
W_KILL_BONUS = 750.0
W_NAVIGATION_ORIENTATION_BONUS = 60.0
W_POSITIONING_BONUS = 40.0 # Reduzido, pois o objetivo de recuo é mais diretivo

ELITISM_PERCENTAGE = 0.10
ELITISM_COUNT = int(POPULATION_SIZE * ELITISM_PERCENTAGE)
W_EXPLORATION_BONUS = 0.1

TOURNAMENT_SIZE = 5
STAGNATION_LIMIT = 1000
IMPROVEMENT_THRESHOLD = 0.05
MAX_MUTATION_RATE = 0.10
MIN_MUTATION_RATE = 0.02
MUTATION_DECAY_GENERATIONS = 800

ADAPTIVE_MUTATION_THRESHOLD = 25

# ==============================================================================
# 2. FUNÇÕES AUXILIARES E DE AVALIAÇÃO
# ==============================================================================
# ALTERADO: Espaço de ações AGORA INCLUI o movimento para trás (essencial para recuar)
def generate_action_space(game):
    buttons = game.get_available_buttons()
    button_indices = {button.name: i for i, button in enumerate(buttons)}
    num_buttons = len(buttons)

    action_templates = [
        # Movimentos Básicos
        {'MOVE_FORWARD'},
        {'MOVE_BACKWARD'}, # ESSENCIAL PARA RECUAR
        {'TURN_RIGHT'},
        {'TURN_LEFT'},
        {'MOVE_RIGHT'},    # Strafe Direita
        {'MOVE_LEFT'},     # Strafe Esquerda
        # Ações de Combate
        {'ATTACK'},
        {'ATTACK', 'MOVE_BACKWARD'},
        {'ATTACK', 'MOVE_RIGHT'},
        {'ATTACK', 'MOVE_LEFT'},
    ]

    final_actions = []
    for template in action_templates:
        action_list = [0] * num_buttons
        for button_name in template:
            if button_name in button_indices:
                index = button_indices[button_name]
                action_list[index] = 1
        final_actions.append(action_list)

    training_logger.info(f"ESPAÇO DE AÇÕES TÁTICO FINAL GERADO: {len(final_actions)} ações.")
    return final_actions, button_indices

def initialize_game():
    training_logger.info("Inicializando instância única do ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_doom_skill(1)
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

def get_label_entity(state, label_name: str):
    if state and state.labels:
        for label in state.labels:
            if label.object_name == label_name:
                return {
                    'pos': np.array([label.object_position_x, label.object_position_y]),
                    'name': label.object_name,
                    'id': label.object_id
                }

def get_all_enemies(state):
    all_enemies = []
    for enemy in ENEMY_NAMES:
        enemy_data = get_label_entity(state, enemy)
        if enemy_data:
            all_enemies.append(enemy_data)
    return all_enemies
    # if state and state.labels:
    #     for label in state.labels:
    #         if label.object_name in ENEMY_NAMES:
    #             enemy_pos = np.array([label.object_position_x, label.object_position_y])
    #             enemy_zone = get_current_zone(enemy_pos)
    #             # Adiciona ID para rastreamento no Target Lock
    #             enemy_data = {'pos': enemy_pos, 'name': label.object_name, 'id': label.object_id} 
    #             all_enemies.append(enemy_data)
    #             if enemy_zone != -1:
    #                 enemies_by_zone[enemy_zone].append(enemy_data)
    # return all_enemies, enemies_by_zone

def evaluate_individual(genome, game, actions, b_map, nn_config, individual_index):
    input_size, lstm_hidden_size, output_size = nn_config
    agent_nn = NeuralNetwork(input_size, lstm_hidden_size, output_size)
    agent_nn.set_weights_from_flat(genome)
    
    # --- Parâmetros da nova filosofia "Exterminador" ---
    W_EXPLORATION_BONUS = 0.5 # Aumentamos um pouco, pois só se aplica em momentos seguros
    W_IGNORE_ENEMY_PENALTY = -2.0 # Penalidade por avançar com inimigos na tela

    game.new_episode()
    training_logger.info(f"->Individuo {individual_index}:")
    
    last_health = game.get_game_variable(vzd.GameVariable.HEALTH)
    last_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
    total_reward = 0.0
    last_kill_count = 0
    max_position_x = 0.0
    
    hidden_state = agent_nn.init_hidden()

    max_ticks = 2100
    for tick in range(max_ticks):
        if game.is_episode_finished(): break
        state = game.get_state()
        if state is None: continue
        
        player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
        player_angle = game.get_game_variable(vzd.GameVariable.ANGLE)
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        current_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        all_enemies = get_all_enemies(state) # Não precisamos mais de 'enemies_by_zone'
        
        # REVISADO: Lógica de recompensa e penalidade de exploração
        progress_made = player_pos[0] - max_position_x
        if progress_made > 0:
            if not all_enemies: # Só recompensa o avanço se não houver inimigos
                total_reward += W_EXPLORATION_BONUS * progress_made
            else: # Penaliza o avanço se houver inimigos na tela
                total_reward += W_IGNORE_ENEMY_PENALTY * progress_made
        max_position_x = player_pos[0]
        
        is_combat_mode = bool(all_enemies)
        current_target = None
        if is_combat_mode:
            # Encontra o inimigo mais próximo como alvo principal
            closest_enemy_dist = float('inf')
            for enemy in all_enemies:
                dist = np.linalg.norm(enemy['pos'] - player_pos)
                if dist < closest_enemy_dist:
                    closest_enemy_dist = dist
                    current_target = enemy

        # ==================================================================
        # REVISADO: Construção do Vetor de Entrada Simplificado
        # Removemos a complexidade das zonas para focar no combate
        # NN_INPUT_SIZE agora precisará ser ajustado para 8
        # ==================================================================
        nn_input = []
        
        dist_ao_alvo_norm = 0.0
        angulo_ao_alvo_norm = 0.0
        if current_target:
            delta_alvo = current_target['pos'] - player_pos
            dist_ao_alvo_norm = np.linalg.norm(delta_alvo) / 1000.0
            target_angle_alvo = math.degrees(math.atan2(delta_alvo[1], delta_alvo[0]))
            angulo_ao_alvo_norm = ((player_angle - target_angle_alvo + 180) % 360 - 180) / 180.0

        delta_obj_final = get_label_entity(state, 'GreenArmor')['pos'] - player_pos
        dist_obj_final_norm = np.linalg.norm(delta_obj_final) / 1500.0
        target_angle_obj_final = math.degrees(math.atan2(delta_obj_final[1], delta_obj_final[0]))
        angulo_obj_final_norm = ((player_angle - target_angle_obj_final + 180) % 360 - 180) / 180.0

        game_living_reward = game.get_living_reward()
        game_death_penalty = game.get_death_penalty()
        get_kill_reward = game.get_kill_reward()
        get_hit_taken_reward = game.get_hit_taken_reward()
        get_hit_reward = game.get_hit_reward()
        get_hit_taken_penalty = game.get_hit_taken_penalty()
        get_damage_made_reward = game.get_damage_made_reward()
        get_damage_taken_reward = game.get_damage_taken_reward()
        get_damage_taken_penalty = game.get_damage_taken_penalty()
        get_armor_reward = game.get_armor_reward()
        get_last_reward = game.get_last_reward()

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

        action_scores, hidden_state = agent_nn.forward(nn_input, hidden_state)
        action_index = np.argmax(action_scores)
        
        total_reward += W_TIME_PENALTY
        damage_dealt_before_action = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        game.make_action(actions[action_index], FRAME_SKIP)
        
        current_kill_count = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        if current_kill_count > last_kill_count:
            total_reward += (current_kill_count - last_kill_count) * W_KILL_BONUS
        last_kill_count = current_kill_count

        damage_delta = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT) - damage_dealt_before_action
        if damage_delta > 0: total_reward += W_DAMAGE_DEALT_BONUS * damage_delta
        
        ammo_after_action = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        if ammo_after_action < last_ammo and damage_delta == 0:
            total_reward += W_WASTED_SHOT_PENALTY
        
        health_lost = last_health - current_health
        if health_lost > 0:
            total_reward += (W_COMBAT_DAMAGE_PENALTY * health_lost)
        
        last_health = current_health
        last_ammo = ammo_after_action
    
    # --- Lógica de Fim de Episódio (Simplificada) ---
    final_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    final_health = game.get_game_variable(vzd.GameVariable.HEALTH)

    if game.is_player_dead():
        # A penalidade de morte agora é mais simples
        total_reward += DEATH_PENALTY_BASE + (final_kills * 500) # Recompensa um pouco mais por kills antes de morrer
        training_logger.info(f"  --> Agente morreu.")
    elif game.is_episode_finished():
        total_reward += W_SURVIVAL_BONUS
        total_reward += W_HEALTH_PRESERVATION_BONUS * final_health
        total_reward += W_AMMO_CONSERVED_BONUS * ammo_after_action
        # Bônus de conclusão só se matou a maioria dos inimigos
        if final_kills >= 5: # Total de 6 inimigos na fase
            training_logger.info(f"  --> MISSÃO CUMPRIDA! Bônus de conclusão concedido.")
            total_reward += LEVEL_COMPLETION_BONUS
    get_total_reward = game.get_total_reward()        
    training_logger.info(f"  --> Total de Kills: {final_kills}, Vida Restante: {final_health}")
    return total_reward

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

if __name__ == "__main__":
    game_instance, button_map_for_rewards = initialize_game()
    possible_actions, _ = generate_action_space(game_instance)
    num_possible_actions = len(possible_actions)
    training_logger.info(f"Número de ações possíveis detectado: {num_possible_actions}")
    NN_OUTPUT_SIZE = num_possible_actions

    nn_config = (NN_INPUT_SIZE, LSTM_HIDDEN_SIZE, NN_OUTPUT_SIZE)

    temp_nn = NeuralNetwork(*nn_config)
    genome_length = temp_nn.total_weights
    training_logger.info(f"Arquitetura da Rede: Entrada({NN_INPUT_SIZE}) -> LSTM({LSTM_HIDDEN_SIZE}) -> Saída({NN_OUTPUT_SIZE})")
    training_logger.info(f"Tamanho do Genoma (Pesos): {genome_length}")

    current_population = create_initial_population(genome_length)
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    generation = 0

    while True:
        training_logger.info(f"\n{'='*20} GERAÇÃO {generation} {'='*20}")
        
        # ALTERADO: Lógica de Mutação Adaptativa
        if generations_without_improvement > ADAPTIVE_MUTATION_THRESHOLD:
            # Aumenta a mutação para tentar "escapar" do ótimo local
            current_mutation_rate = MAX_MUTATION_RATE
            training_logger.info(f"Estagnação detectada! Resetando a mutação para o valor máximo: {current_mutation_rate:.4f}")
        else:
            # Lógica de decaimento normal
            decay_progress = min(1.0, generation / MUTATION_DECAY_GENERATIONS)
            current_mutation_rate = MAX_MUTATION_RATE - (MAX_MUTATION_RATE - MIN_MUTATION_RATE) * decay_progress
        
        training_logger.info(f"Taxa de Mutação para esta Geração: {current_mutation_rate:.4f}")
        start_time = time.time()
        
        for i, ind in enumerate(current_population):
            if ind['fitness'] is None:
                num_runs = 3  # Joga 3 vezes para testar a consistência
                fitness_scores = []
                training_logger.info(f"Avaliando indivíduo {i+1} em {num_runs} tentativas...")
                
                for run in range(num_runs):
                    score = evaluate_individual(ind['genome'], game_instance, possible_actions, button_map_for_rewards, nn_config, i + 1)
                    fitness_scores.append(score)
                
                # Usa a média do fitness. Indivíduos sortudos serão penalizados.
                ind['fitness'] = np.mean(fitness_scores)
                training_logger.info(f"  --> Fitness médio do indivíduo {i+1}: {ind['fitness']:.2f}")
                
        eval_time = time.time() - start_time
        training_logger.info(f"Avaliação de {POPULATION_SIZE} indivíduos concluída em {eval_time:.2f}s.")
        sorted_population = sorted(current_population, key=lambda x: x['fitness'], reverse=True)
        current_best_fitness = sorted_population[0]['fitness']
        training_logger.info(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")
        if current_best_fitness > best_fitness_overall + 1.0:
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