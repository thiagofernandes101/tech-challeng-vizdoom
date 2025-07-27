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
    actions = [list(p) for p in product([0, 1], repeat=num_buttons)]
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

def calculate_fitness(game, individual, actions):
    """
    Calcula o fitness recompensando o progresso SEGURO.
    Penaliza o avanço imprudente para forçar o agente a lidar com inimigos primeiro.
    """
    game.new_episode()
    
    # Existe um delay antes do personagem começar a executar as ações, ou seja alguns dos primeiros movimentos são ignorados
    for _ in range(7):
        game.advance_action()
    
    # Variáveis para rastreamento passo a passo
    last_health = game.get_game_variable(vzd.GameVariable.HEALTH)
    last_pos_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
    total_progress_reward = 0.0

    for action_index in individual["genome"]:
        if game.is_episode_finished():
            break
            
        game.make_action(actions[action_index])
        
        # --- Lógica de Progresso Condicional ---
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        current_pos_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
        
        delta_pos_x = current_pos_x - last_pos_x
        health_lost = last_health - current_health

        if delta_pos_x > 0:  # O agente tentou avançar
            if health_lost <= 0:  # Avançou com segurança (sem perder vida)
                total_progress_reward += W_SAFE_PROGRESS * delta_pos_x
            else:  # Avançou de forma imprudente (enquanto tomava dano)
                total_progress_reward += W_UNSAFE_PROGRESS_PENALTY * delta_pos_x
        
        # Atualiza as variáveis de rastreamento para o próximo passo
        last_health = current_health
        last_pos_x = current_pos_x

    # --- Cálculo Final do Fitness ---
    steps_taken = game.get_episode_time()
    kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    final_health = game.get_game_variable(vzd.GameVariable.HEALTH)
    ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)

    if game.is_player_dead():
        # A penalidade por progresso imprudente já foi aplicada, então o fitness reflete isso
        fitness_score = DEATH_PENALTY + total_progress_reward + (W_SURVIVAL_TIME * steps_taken)
        return fitness_score
    else:
        performance_bonus = (W_KILL_SCORE * kills) + \
                            (W_HEALTH_BONUS * final_health) + \
                            (W_AMMO_BONUS * ammo)
        
        completion_bonus = LEVEL_COMPLETION_BONUS if game.is_episode_finished() else 0.0
        
        # O fitness final agora usa a recompensa de progresso calculada dinamicamente
        fitness_score = total_progress_reward + performance_bonus + completion_bonus + (W_SURVIVAL_TIME * steps_taken)
        return fitness_score

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
                fitness = calculate_fitness(game_instance, individual, possible_actions)
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