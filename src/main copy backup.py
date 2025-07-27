import vizdoom as vzd
import numpy as np
import random
import time
from itertools import product

# ==============================================================================
# 1. PARÂMETROS DO ALGORITMO GENÉTICO E DO JOGO (AJUSTADOS)
# ==============================================================================

POPULATION_SIZE = 200
GENOME_LENGTH = 3000  # Aumentado para dar mais tempo para estratégias

# --- Pesos da Função de Fitness (REFINADOS) ---
W_PROGRESS_X = 15.0
W_SURVIVAL_TIME = 1.0
W_KILL_SCORE = 300.0
W_HEALTH_BONUS = 3.0
W_AMMO_BONUS = 0.1
W_DAMAGE_PENALTY = -10.0  # Novo: penalidade direta por tomar dano
DEATH_PENALTY = -2000.0
LEVEL_COMPLETION_BONUS = 5000.0
W_SAFE_PROGRESS = 25.0
W_UNSAFE_PROGRESS_PENALTY = -15.0
W_WASTED_SHOT_PENALTY = -2.0
W_SMART_SHOT_BONUS = 5.0
W_BACKWARD_PENALTY = -1.0
W_IDLE_PENALTY = -0.5
W_EVASION_BONUS = 7.0
W_COMBAT_DAMAGE_PENALTY = -15.0
NUM_EVAL_RUNS = 3                       # Número de partidas para avaliar cada indivíduo
W_INCONSISTENCY_PENALTY = 1.5           # Multiplicador da penalidade por desvio padrão. PUNE A INCONSISTÊNCIA.
W_CAUTIOUS_PROGRESS_MULTIPLIER = 0.1    # Reduz a recompensa de progresso para 10% quando inimigos estão na tela, mas longe.

GOAL_POSITION = np.array([1312.0, 0.0])  # Posição (X, Y) da GreenArmor
COMBAT_PROXIMITY_THRESHOLD = 300.0      # Distância para um inimigo ativar o "Modo de Combate"
W_GOAL_PROGRESS = 2.0                   # Recompensa por reduzir a distância até o objetivo
W_COMBAT_KILL_BONUS = 400.0             # Bônus massivo por conseguir um abate EM MODO DE COMBATE

# --- Parâmetros dos Operadores Genéticos (CORRIGIDOS) ---
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.02  # <-- MUDANÇA CRÍTICA
ELITISM_COUNT = 1

# --- Parâmetros de Critério de Parada (AJUSTADOS) ---
MAX_GENERATIONS = 999999
STAGNATION_LIMIT = 50 # <-- MUDANÇA CRÍTICA
IMPROVEMENT_THRESHOLD = 0.1
INITIAL_MUTATION_RATE = 0.02
BOOSTED_MUTATION_RATE = 0.05

SCENARIO_PATH = "deadly_corridor.cfg"

# ==============================================================================
# 2. FUNÇÕES (COM FITNESS MELHORADA)
# ==============================================================================

def initialize_game():
    print("Inicializando ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_available_buttons([
        vzd.Button.ATTACK,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
    ])
    game.init()
    num_buttons = game.get_available_buttons_size()
    actions = [
        [1, 0, 0, 0, 0],  # 0: Atirar
        [0, 1, 0, 0, 0],  # 1: Mover para frente
        [0, 0, 1, 0, 0],  # 2: Mover para trás
        [0, 0, 0, 1, 0],  # 3: Virar para esquerda
        [0, 0, 0, 0, 1],  # 4: Virar para direita
        [1, 1, 0, 0, 0],  # 5: Atirar e Mover para frente (Ação de assalto)
    ]
    if [0] * num_buttons in actions:
        actions.remove([0] * num_buttons)
    print(f"Cardápio de ações definido com {len(actions)} movimentos possíveis.")
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
    if state and state.labels:
        for label in state.labels:
            if label.object_name not in ["DoomPlayer", "GreenArmor", "Stimpack"]:
                enemies.append(np.array([label.object_position_x, label.object_position_y]))
    return player_pos, enemies

def calculate_robust_fitness(game, individual, actions):
    """
    Calcula um fitness ROBUSTO avaliando o indivíduo em MÚLTIPLAS PARTIDAS.
    Recompensa o desempenho médio e penaliza a inconsistência (alto desvio padrão).
    Usa uma máquina de 3 estados para forçar o engajamento com inimigos.
    """
    episode_scores = []
    for _ in range(NUM_EVAL_RUNS):
        game.new_episode()
        
        # Rastreamento para uma única partida
        last_dist_to_goal = np.linalg.norm(GOAL_POSITION - np.array([0.0, 0.0]))
        last_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        last_kills = 0
        total_reward_for_episode = 0.0

        for action_index in individual["genome"]:
            if game.is_episode_finished():
                break

            state = game.get_state()
            player_pos, enemies = get_entities_from_state(state)
            
            # --- MÁQUINA DE 3 ESTADOS ---
            min_enemy_dist = float('inf')
            agent_state = "CLEAR_NAVIGATION"
            if enemies:
                distances = [np.linalg.norm(player_pos - enemy_pos) for enemy_pos in enemies]
                min_enemy_dist = min(distances)
                agent_state = "COMBAT" if min_enemy_dist < COMBAT_PROXIMITY_THRESHOLD else "CAUTIOUS_NAVIGATION"

            # --- LÓGICA DE RECOMPENSA BASEADA NO ESTADO ---
            current_dist_to_goal = np.linalg.norm(GOAL_POSITION - player_pos)
            distance_reduction = last_dist_to_goal - current_dist_to_goal

            if agent_state == "CLEAR_NAVIGATION":
                total_reward_for_episode += W_GOAL_PROGRESS * distance_reduction
                if actions[action_index][0] == 1: total_reward_for_episode += W_WASTED_SHOT_PENALTY
            
            elif agent_state == "CAUTIOUS_NAVIGATION":
                # Progresso é drasticamente menos recompensado. FOQUE NOS INIMIGOS!
                total_reward_for_episode += (W_GOAL_PROGRESS * W_CAUTIOUS_PROGRESS_MULTIPLIER) * distance_reduction
            
            elif agent_state == "COMBAT":
                health_lost = last_health - game.get_game_variable(vzd.GameVariable.HEALTH)
                if health_lost <= 0: total_reward_for_episode += W_EVASION_BONUS
                else: total_reward_for_episode += W_COMBAT_DAMAGE_PENALTY * health_lost
                
                current_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                if current_kills > last_kills: total_reward_for_episode += W_COMBAT_KILL_BONUS
            
            # Executa a ação
            game.make_action(actions[action_index])
            
            # Atualiza variáveis
            last_dist_to_goal = current_dist_to_goal
            last_health = game.get_game_variable(vzd.GameVariable.HEALTH)
            last_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)

        # --- FIM DA PARTIDA: Cálculo do score do episódio ---
        player_pos, _ = get_entities_from_state(game.get_state())
        if not game.is_player_dead() and np.linalg.norm(GOAL_POSITION - player_pos) < 50:
            total_reward_for_episode += LEVEL_COMPLETION_BONUS
        
        if game.is_player_dead():
            total_reward_for_episode += DEATH_PENALTY
        
        episode_scores.append(total_reward_for_episode)

    # --- FIM DA AVALIAÇÃO: Cálculo do fitness final (média - penalidade por inconsistência) ---
    mean_score = np.mean(episode_scores)
    std_dev = np.std(episode_scores)
    
    # A fórmula final: recompensa a média, PUNE a inconsistência.
    final_fitness = mean_score - (W_INCONSISTENCY_PENALTY * std_dev)
    
    return final_fitness

def tournament_selection(population):
    tournament_competitors = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament_competitors, key=lambda x: x['fitness'])

def two_point_crossover(parent1_genome, parent2_genome):
    assert len(parent1_genome) == len(parent2_genome)
    genome_len = len(parent1_genome)
    
    # Garante que os dois pontos de crossover sejam diferentes
    p1 = random.randint(1, genome_len - 2)
    p2 = random.randint(p1 + 1, genome_len - 1)

    child1_genome = parent1_genome[:p1] + parent2_genome[p1:p2] + parent1_genome[p2:]
    child2_genome = parent2_genome[:p1] + parent1_genome[p1:p2] + parent2_genome[p2:]
    
    return child1_genome, child2_genome

def mutate(genome, num_actions, mutation_rate): # Adicionado parâmetro
    mutated_genome = []
    for gene in genome:
        if random.random() < mutation_rate: # Usa o parâmetro
            mutated_genome.append(random.randint(0, num_actions - 1))
        else:
            mutated_genome.append(gene)
    return mutated_genome

def generate_new_population(old_population, num_actions, mutation_rate):
    sorted_old_population = sorted(old_population, key=lambda x: x['fitness'], reverse=True)
    new_population = []
    for i in range(ELITISM_COUNT):
        new_population.append(sorted_old_population[i])
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(sorted_old_population)
        parent2 = tournament_selection(sorted_old_population)
        child1_genome, child2_genome = two_point_crossover(parent1['genome'], parent2['genome'])
        
        # Passa a taxa para a função de mutação
        mutated_child1_genome = mutate(child1_genome, num_actions, mutation_rate) 
        mutated_child2_genome = mutate(child2_genome, num_actions, mutation_rate)
        
        new_population.append({'genome': mutated_child1_genome, 'fitness': 0.0})
        if len(new_population) < POPULATION_SIZE:
            new_population.append({'genome': mutated_child2_genome, 'fitness': 0.0})
    return new_population

# ==============================================================================
# 3. SCRIPT PRINCIPAL
# ==============================================================================
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
        print(f"Avaliando {len(current_population)} indivíduos...")
        start_time_eval = time.time()
        for individual in current_population:
            if individual['fitness'] == 0.0:
                fitness = calculate_robust_fitness(game_instance, individual, possible_actions)
                individual["fitness"] = fitness
        
        eval_time = time.time() - start_time_eval
        print(f"Avaliação concluída em {eval_time:.2f}s.")

        sorted_population = sorted(current_population, key=lambda x: x['fitness'], reverse=True)
        current_best_fitness = sorted_population[0]['fitness']
        
        print(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")

        if current_best_fitness > best_fitness_overall + IMPROVEMENT_THRESHOLD:
            best_fitness_overall = current_best_fitness
            generations_without_improvement = 0
            current_mutation_rate = INITIAL_MUTATION_RATE
            print(f"✨ Nova melhoria significativa encontrada! Melhor fitness geral: {best_fitness_overall:.2f}")
            np.save('best_genome.npy', sorted_population[0]['genome'])
        else:
            generations_without_improvement += 1
            print(f"Sem melhoria significativa. Gerações estagnadas: {generations_without_improvement}/{STAGNATION_LIMIT}")

        if generations_without_improvement > STAGNATION_LIMIT / 2:
             print(f"⚠️  Estagnação detectada! Aumentando a mutação para {BOOSTED_MUTATION_RATE}.")
             current_mutation_rate = BOOSTED_MUTATION_RATE

        if generations_without_improvement >= STAGNATION_LIMIT:
            print(f"CRITÉRIO DE PARADA ATINGIDO: Ausência de melhoria por {STAGNATION_LIMIT} gerações.")
            break
        
        print("Gerando a próxima população...")
        current_population = generate_new_population(sorted_population, num_possible_actions, current_mutation_rate)

    print("\n" + "="*50)
    print("Evolução finalizada.")
    game_instance.close()
    print("Instância do jogo finalizada. Processo concluído.")