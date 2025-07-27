import vizdoom as vzd
import numpy as np
import random
import time
from itertools import product
import math

# ==============================================================================
# 1. PARÂMETROS DO ALGORITMO GENÉTICO E DO JOGO (REVISADOS)
# ==============================================================================
POPULATION_SIZE = 100
GENOME_LENGTH = 3000
SCENARIO_PATH = "deadly_corridor.cfg"
GOAL_POSITION = np.array([1312.0, 0.0])

# --- Avaliação e Consistência ---
NUM_EVAL_RUNS = 3  # AUMENTADO: Crucial para reduzir a sorte e obter um fitness mais estável.
W_INCONSISTENCY_PENALTY = 1.5

# --- Recompensas e Penalidades Globais ---
DEATH_PENALTY = -2000.0
LEVEL_COMPLETION_BONUS = 5000.0
W_TIME_PENALTY = -1.0  # Levemente ajustado, pois a estagnação já é penalizada.

# --- Recompensas e Penalidades de NAVEGAÇÃO ---
W_GOAL_PROGRESS = 200.0  # Mantido alto para incentivar o avanço.
W_STAGNATION_PENALTY = -75.0 # Aumentado para punir ainda mais a inação.
STAGNATION_TICKS_THRESHOLD = 40
STAGNATION_DISTANCE_THRESHOLD = 10.0

# --- Recompensas e Penalidades de COMBATE e MIRA (NOVOS E REFINADOS) ---
W_COMBAT_KILL_BONUS = 600.0 # Ligeiramente aumentado.
W_DAMAGE_DEALT_BONUS = 25.0
W_COMBAT_DAMAGE_PENALTY = -5.0
W_CROSSHAIR_ON_TARGET_BONUS = 30.0 # NOVO: Recompensa massiva por atirar com a mira no alvo.
W_WASTED_SHOT_PENALTY = -20.0      # NOVO: Penalidade por atirar sem inimigos na tela (em modo navegação).
W_AMMO_USAGE_PENALTY = -1.0        # NOVO: Pequeno custo para cada bala gasta.
LEVEL_COMPLETION_AMMO_BONUS_PER_BULLET = 5.0 # NOVO: Grande bônus por munição restante NO FINAL.

# --- Parâmetros dos Operadores Genéticos ---
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2

# --- Parâmetros de Critério de Parada e Mutação ---
MAX_GENERATIONS = 999999
STAGNATION_LIMIT = 100
IMPROVEMENT_THRESHOLD = 0.1
INITIAL_MUTATION_RATE = 0.02
BOOSTED_MUTATION_RATE = 0.05
HYPERMUTATION_RATE = 0.20
HYPERMUTATION_TRIGGER = int(STAGNATION_LIMIT * 0.75)
HYPERMUTATION_COUNT = 5

# ==============================================================================
# 2. FUNÇÕES (COM LÓGICA DE MIRA E MODOS DE COMPORTAMENTO)
# ==============================================================================

# ... (funções generate_action_space, initialize_game, create_initial_population, etc. inalteradas) ...
def generate_action_space(game):
    buttons = game.get_available_buttons()
    button_indices = {button.name: i for i, button in enumerate(buttons)}
    num_buttons = len(buttons)
    conflict_groups = [{'MOVE_FORWARD', 'MOVE_BACKWARD'}, {'TURN_LEFT', 'TURN_RIGHT'}, {'MOVE_LEFT', 'MOVE_RIGHT'}]
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

def initialize_game():
    print("Inicializando ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True) # Essencial para a lógica de mira
    game.set_doom_skill(1)
    game.init()
    actions = generate_action_space(game)
    return game, actions

def create_initial_population(num_actions):
    population = []
    for _ in range(POPULATION_SIZE):
        genome = [random.randint(0, num_actions - 1) for _ in range(GENOME_LENGTH)]
        population.append({"genome": genome, "fitness": 0.0})
    return population

def get_entities_from_state(state):
    player_pos = np.array([
        game_instance.get_game_variable(vzd.GameVariable.POSITION_X),
        game_instance.get_game_variable(vzd.GameVariable.POSITION_Y)
    ])
    enemies = []
    armor_pos = None
    known_enemy_names = {"Zombieman", "ShotgunGuy", "Imp", "Demon"}
    
    if state and state.labels:
        for label in state.labels:
            if label.object_name in known_enemy_names:
                enemies.append({
                    'pos': np.array([label.object_position_x, label.object_position_y]),
                    'name': label.object_name,
                    # NOVO: Coordenadas na tela para a lógica de mira
                    'x': label.x, 'y': label.y, 'width': label.width, 'height': label.height
                })
            elif label.object_name == "GreenArmor":
                armor_pos = np.array([label.object_position_x, label.object_position_y])
    return player_pos, enemies, armor_pos

def calculate_tactical_fitness(game, individual, actions):
    episode_scores = []
    buttons = game.get_available_buttons()
    attack_button_index = [i for i, b in enumerate(buttons) if b.name == 'ATTACK'][0]
    
    screen_width = game.get_screen_width()
    screen_height = game.get_screen_height()
    crosshair_pos = (screen_width / 2, screen_height / 2)

    for _ in range(NUM_EVAL_RUNS):
        game.new_episode()
        
        last_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        last_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        last_kills = 0
        last_distance_to_goal = None
        stagnation_tick_counter = 0
        stagnation_anchor_pos = None
        total_reward_for_episode = 0.0

        # NOVO: Variável de memória para a posição da armadura
        known_armor_pos = None

        for action_index in individual["genome"]:
            if game.is_episode_finished():
                break

            state = game.get_state()
            if state is None: continue

            player_pos, threats, armor_pos_from_label = get_entities_from_state(state)
            
            # ALTERADO: Lógica de memória para o alvo principal.
            # Se a armadura for vista, atualizamos nossa memória.
            if armor_pos_from_label is not None:
                known_armor_pos = armor_pos_from_label

            # O alvo é a posição memorizada da armadura ou, na sua ausência, o objetivo final estático.
            current_main_goal = known_armor_pos if known_armor_pos is not None else GOAL_POSITION
            
            in_combat_mode = len(threats) > 0
            total_reward_for_episode += W_TIME_PENALTY
            
            current_distance_to_goal = np.linalg.norm(current_main_goal - player_pos)
            
            # 1. Recompensa de Progresso
            if last_distance_to_goal is not None:
                progress_made = last_distance_to_goal - current_distance_to_goal
                progress_multiplier = 0.25 if in_combat_mode else 1.0
                total_reward_for_episode += W_GOAL_PROGRESS * progress_made * progress_multiplier
            last_distance_to_goal = current_distance_to_goal

            # 2. Penalidade de Estagnação
            if stagnation_anchor_pos is None: stagnation_anchor_pos = player_pos
            if np.linalg.norm(player_pos - stagnation_anchor_pos) < STAGNATION_DISTANCE_THRESHOLD:
                stagnation_tick_counter += 1
                if stagnation_tick_counter >= STAGNATION_TICKS_THRESHOLD:
                    total_reward_for_episode += W_STAGNATION_PENALTY
                    stagnation_tick_counter = 0
            else:
                stagnation_anchor_pos = player_pos
                stagnation_tick_counter = 0

            # ... (O restante da função permanece exatamente o mesmo) ...

            action_to_perform = actions[action_index]
            is_shooting = action_to_perform[attack_button_index] == 1
            
            current_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
            ammo_used = last_ammo - current_ammo
            if ammo_used > 0:
                total_reward_for_episode += W_AMMO_USAGE_PENALTY * ammo_used
            last_ammo = current_ammo

            if is_shooting:
                if in_combat_mode:
                    crosshair_on_target = False
                    for enemy in threats:
                        if (enemy['x'] <= crosshair_pos[0] <= enemy['x'] + enemy['width'] and
                            enemy['y'] <= crosshair_pos[1] <= enemy['y'] + enemy['height']):
                            crosshair_on_target = True
                            break
                    if crosshair_on_target:
                        total_reward_for_episode += W_CROSSHAIR_ON_TARGET_BONUS
                else:
                    total_reward_for_episode += W_WASTED_SHOT_PENALTY

            if in_combat_mode:
                # Usar get_total_reward() é mais robusto para dano, pois é o delta desde a última ação.
                damage_dealt_reward = game.get_total_reward()
                total_reward_for_episode += W_DAMAGE_DEALT_BONUS * damage_dealt_reward
                
                health_lost = last_health - game.get_game_variable(vzd.GameVariable.HEALTH)
                if health_lost > 0:
                    total_reward_for_episode += W_COMBAT_DAMAGE_PENALTY * health_lost
            
            game.make_action(action_to_perform)

            last_health = game.get_game_variable(vzd.GameVariable.HEALTH)
            current_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            if current_kills > last_kills:
                total_reward_for_episode += W_COMBAT_KILL_BONUS * (current_kills - last_kills)
                last_kills = current_kills

            if game.is_episode_finished():
                if game.is_player_dead():
                    total_reward_for_episode += DEATH_PENALTY
                else:
                    # Verifica se o jogador chegou perto do objetivo final físico
                    final_player_pos_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
                    # Usamos o GOAL_POSITION estático aqui, pois é a condição de vitória do mapa.
                    if abs(final_player_pos_x - GOAL_POSITION[0]) < 100:
                        total_reward_for_episode += LEVEL_COMPLETION_BONUS
                        final_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
                        total_reward_for_episode += final_ammo * LEVEL_COMPLETION_AMMO_BONUS_PER_BULLET
                break

        if not game.is_episode_finished() and game.is_player_dead():
            total_reward_for_episode += DEATH_PENALTY

        episode_scores.append(total_reward_for_episode)

    mean_score = np.mean(episode_scores) if episode_scores else 0.0
    std_dev = np.std(episode_scores) if len(episode_scores) > 1 else 0.0
    final_fitness = mean_score - (W_INCONSISTENCY_PENALTY * std_dev)
    
    return final_fitness

# ... (generate_new_population e o __main__ permanecem os mesmos da resposta anterior)
def tournament_selection(population):
    tournament_competitors = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament_competitors, key=lambda x: x['fitness'])

def two_point_crossover(parent1_genome, parent2_genome):
    assert len(parent1_genome) == len(parent2_genome)
    genome_len = len(parent1_genome)
    if genome_len < 3:
        return parent1_genome, parent2_genome
    p1 = random.randint(1, genome_len - 2)
    p2 = random.randint(p1 + 1, genome_len - 1)
    child1_genome = parent1_genome[:p1] + parent2_genome[p1:p2] + parent1_genome[p2:]
    child2_genome = parent2_genome[:p1] + parent1_genome[p1:p2] + parent2_genome[p2:]
    return child1_genome, child2_genome

def mutate(genome, num_actions, mutation_rate):
    mutated_genome = []
    for gene in genome:
        if random.random() < mutation_rate:
            mutated_genome.append(random.randint(0, num_actions - 1))
        else:
            mutated_genome.append(gene)
    return mutated_genome

def generate_new_population(old_population, num_actions, mutation_rate, gen_without_improvement):
    sorted_old_population = sorted(old_population, key=lambda x: x['fitness'], reverse=True)
    new_population = []
    for i in range(ELITISM_COUNT):
        new_population.append(sorted_old_population[i])
    use_hypermutation = gen_without_improvement >= HYPERMUTATION_TRIGGER
    if use_hypermutation:
        print("🔥 Ativando Hipermutação para alguns indivíduos!")
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(sorted_old_population)
        parent2 = tournament_selection(sorted_old_population)
        child1_genome, child2_genome = two_point_crossover(parent1['genome'], parent2['genome'])
        current_child_mutation_rate = mutation_rate
        if use_hypermutation and len(new_population) < ELITISM_COUNT + HYPERMUTATION_COUNT:
            current_child_mutation_rate = HYPERMUTATION_RATE
        mutated_child1_genome = mutate(child1_genome, num_actions, current_child_mutation_rate)
        mutated_child2_genome = mutate(child2_genome, num_actions, current_child_mutation_rate)
        new_population.append({'genome': mutated_child1_genome, 'fitness': 0.0})
        if len(new_population) < POPULATION_SIZE:
            new_population.append({'genome': mutated_child2_genome, 'fitness': 0.0})
    return new_population


if __name__ == "__main__":
    game_instance, possible_actions = initialize_game()
    num_possible_actions = len(possible_actions)
    print(f"\nCriando população inicial com {POPULATION_SIZE} indivíduos...")
    current_population = create_initial_population(num_possible_actions)
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    current_mutation_rate = INITIAL_MUTATION_RATE
    for generation in range(MAX_GENERATIONS):
        print(f"\n{'='*20} GERAÇÃO {generation} {'='*20}")
        # print(f"Avaliando {len(current_population)} indivíduos (Taxa de Mutação: {current_mutation_rate}, Avaliações: {NUM_EVAL_RUNS})...")
        start_time_eval = time.time()
        for i, individual in enumerate(current_population):
            # A reavaliação de indivíduos de elite pode ser útil com a nova lógica de fitness
            # if individual['fitness'] == 0.0:
            fitness = calculate_tactical_fitness(game_instance, individual, possible_actions)
            individual["fitness"] = fitness
            # print(f"  Indivíduo {i+1}/{POPULATION_SIZE} avaliado. Fitness: {fitness:.2f}", end='\r')
        eval_time = time.time() - start_time_eval
        print(f"\nAvaliação concluída em {eval_time:.2f}s.")
        sorted_population = sorted(current_population, key=lambda x: x['fitness'], reverse=True)
        current_best_fitness = sorted_population[0]['fitness']
        print(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")
        if current_best_fitness > best_fitness_overall + IMPROVEMENT_THRESHOLD:
            best_fitness_overall = current_best_fitness
            generations_without_improvement = 0
            current_mutation_rate = INITIAL_MUTATION_RATE
            print(f"✨ Nova melhoria significativa encontrada! Melhor fitness geral: {best_fitness_overall:.2f}")
            np.save(f'best_genome_{generation}.npy', sorted_population[0]['genome'])
        else:
            generations_without_improvement += 1
            print(f"Sem melhoria significativa. Gerações estagnadas: {generations_without_improvement}/{STAGNATION_LIMIT}")
        if generations_without_improvement > STAGNATION_LIMIT / 2 and current_mutation_rate == INITIAL_MUTATION_RATE:
            print(f"⚠️ Estagnação detectada! Aumentando a mutação para {BOOSTED_MUTATION_RATE}.")
            current_mutation_rate = BOOSTED_MUTATION_RATE
        if generations_without_improvement >= STAGNATION_LIMIT:
            print(f"CRITÉRIO DE PARADA ATINGIDO: Ausência de melhoria por {STAGNATION_LIMIT} gerações.")
            break
        print("Gerando a próxima população...")
        current_population = generate_new_population(sorted_population, num_possible_actions, current_mutation_rate, generations_without_improvement)
    print("\n" + "="*50)
    print("Evolução finalizada.")
    game_instance.close()
    print("Instância do jogo finalizada. Processo concluído.")