import vizdoom as vzd
import numpy as np
import random
import time
import json
import os
from itertools import product
from vizdoom import GameVariable
import matplotlib
matplotlib.use('TkAgg')  # Força backend interativo para exibir gráficos
import matplotlib.pyplot as plt


## VALIDAR OS PESOS, PENSAR EM ALGO PARA DESESTAGNAR, AVALIAR A DISTANCIA E A DIREÇAO PERCORRIDA E AUMENTAR A ALEATORIEDADE CONFORME AUMENTA A ESTAGNACAO

## REMOVER O ANDAR EM CIRCULOS (ultimas mudanças)

# ==============================================================================
# 1. PARÂMETROS DO ALGORITMO GENÉTICO E DO JOGO
# ==============================================================================

# --- Parâmetros da População ---
POPULATION_SIZE = 100  # Tamanho da população (100 indivíduos)
GENOME_LENGTH = 3000   # Número máximo de ações por episódio (duração da "vida" do agente)

## ARRUMAR O PONTO DE PASSOS PRA FRENTE E AUMENTAR A PENALIDADE POR TIRO ERRADO -> avaliar se estao funcionando corretamente no calculo

# --- Pesos da Função de Fitness (Ajuste estes valores para guiar a evolução!) ---
# O objetivo é maximizar essa pontuação
W_KILLS    = 200.0    # Valorize muito matar inimigos
W_HEALTH   = 1.0      # Valorize um pouco a vida restante
W_AMMO     = 0.2      # Valorize pouco a munição restante
W_SHOT     = -2.0     # Penalize pouco por tiro (precisa atirar para matar)
W_STEPS    = 0.0      # Não penalize por andar (deixe o agente explorar)
W_FORWARD  = 10.0     # Incentive fortemente andar para frente
W_REWARD   = 0.10     # Deixe como está
W_BACKWARD = -5.0     # Penalize andar para trás, mas menos que antes
W_CIRCLE   = -1.0     # Penalize girar em círculo, mas menos
W_SIDE     = -2.0     # Penalize andar para os lados, mas menos

# --- Parâmetros dos Operadores Genéticos ---
TOURNAMENT_SIZE = 3     # Número de indivíduos que competem em cada torneio de seleção
MUTATION_RATE = 0.35    # Probabilidade de um gene sofrer mutação
ELITISM_COUNT = max(2, POPULATION_SIZE // 20)  # 5% da população, mínimo 2

# --- Parâmetros de Critério de Parada ---
MAX_GENERATIONS = 300
STAGNATION_LIMIT = 100
IMPROVEMENT_THRESHOLD = 0.1 # A melhoria mínima no fitness para ser considerada "significativa"

# --- Configuração do ViZDoom ---
SCENARIO_PATH = "deadly_corridor.cfg"  # Nome do arquivo de configuração do cenário

# --- Configuração de Salvamento ---
SAVE_DIR = "saved_individuals"  # Diretório para salvar os melhores indivíduos
BEST_INDIVIDUAL_FILE = "best_individual.json"  # Arquivo do melhor indivíduo

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def initialize_game(headless=True):
    """
    Cria e configura a instância do jogo ViZDoom.
    
    Args:
        headless (bool): Se True, executa sem janela (para treinamento rápido)
                        Se False, mostra a janela (para demonstração)
    """
    print("Inicializando ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)
    
    # Configura visibilidade da janela baseado no parâmetro
    game.set_window_visible(not headless)
    
    # Se não for headless, adiciona delay para visualização
    if not headless:
        game.set_ticrate(35)  # 35 FPS para visualização mais suave
    
    game.set_mode(vzd.Mode.PLAYER)  # type: ignore
    game.set_available_buttons([
        vzd.Button.ATTACK,  # type: ignore
        vzd.Button.MOVE_LEFT,  # type: ignore
        vzd.Button.MOVE_RIGHT,  # type: ignore
        vzd.Button.MOVE_FORWARD,  # type: ignore
        vzd.Button.MOVE_BACKWARD,  # type: ignore
        vzd.Button.TURN_LEFT,  # type: ignore
        vzd.Button.TURN_RIGHT,  # type: ignore
        vzd.Button.MOVE_UP,  # type: ignore
        vzd.Button.MOVE_DOWN  # type: ignore
    ])
    game.init()
    
    # Crie o "cardápio" de ações possíveis.
    # Cada ação é uma lista de 0s e 1s com o mesmo tamanho do número de botões.
    
    num_buttons = game.get_available_buttons_size()
    
    # Gera todas as combinações possíveis de botões (0 ou 1 para cada)
    # product([0, 1], repeat=num_buttons) -> (0,0,0,0), (0,0,0,1), (0,0,1,0), ...
    actions = [list(p) for p in product([0, 1], repeat=num_buttons)]
    
    # Opcional: Remova a ação "não fazer nada" se não for desejada
    actions.remove([0] * num_buttons)
    # --- define pares conflitantes ---
    conflicts = [
        (1, 2),  # left vs right
        (3, 4),  # forward vs backward
        (5, 6),  # turn left vs turn right
        (7, 8)   # move up vs move down
    ]

    filtered = []
    for a in actions:
        # pula se apertou ambos de qualquer par conflitante
        if any(a[i] and a[j] for i, j in conflicts):
            continue
        # opcional: limitar a no máximo 2 botões de cada vez
        if sum(a) > 2:
            continue
        filtered.append(a)

    actions = filtered
    print(f"Ações filtradas: de {len(actions)} para {len(filtered)} possíveis movimentos.")
    print(f"Cardápio de ações definido com {len(actions)} movimentos possíveis.")
    
    return game, actions

def save_individual(individual, filename=None):
    """
    Salva um indivíduo em arquivo JSON.
    
    Args:
        individual (dict): Dicionário contendo 'genome' e 'fitness'
        filename (str): Nome do arquivo. Se None, usa nome padrão com timestamp
    """
    # Cria diretório se não existir
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"individual_fitness_{individual['fitness']:.2f}_{timestamp}.json"
    
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Prepara dados para salvar
    data_to_save = {
        'genome': individual['genome'],
        'fitness': individual['fitness'],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'population_size': POPULATION_SIZE,
            'genome_length': GENOME_LENGTH,
            'w_kills': W_KILLS,
            'w_health': W_HEALTH,
            'w_ammo': W_AMMO,
            'w_steps': W_STEPS
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Indivíduo salvo em: {filepath}")
    return filepath

def load_individual(filepath):
    """
    Carrega um indivíduo de arquivo JSON.
    
    Args:
        filepath (str): Caminho para o arquivo JSON
        
    Returns:
        dict: Dicionário contendo 'genome' e 'fitness'
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    individual = {
        'genome': data['genome'],
        'fitness': data['fitness']
    }
    
    print(f"Indivíduo carregado de: {filepath}")
    print(f"Fitness: {individual['fitness']:.2f}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    
    return individual

def demonstrate_individual(individual, actions, num_episodes=3):
    """
    Demonstra um indivíduo executando no jogo com tela visível.
    
    Args:
        individual (dict): Indivíduo a ser demonstrado
        actions (list): Lista de ações possíveis
        num_episodes (int): Número de episódios para demonstrar
    """
    print(f"\n{'='*50}")
    print(f"DEMONSTRAÇÃO DO MELHOR INDIVÍDUO")
    print(f"Fitness: {individual['fitness']:.2f}")
    print(f"Episódios: {num_episodes}")
    print(f"{'='*50}")
    
    # Inicializa jogo com tela visível
    game, _ = initialize_game(headless=False)
    
    total_kills = 0
    total_health = 0
    total_ammo = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        print(f"\n--- Episódio {episode + 1}/{num_episodes} ---")
        
        game.new_episode()
        episode_steps = 0
        
        # Executa cada ação do genoma
        for action_index in individual["genome"]:
            if game.is_episode_finished():
                break
            
            action_to_perform = actions[action_index]
            game.make_action(action_to_perform)
            episode_steps += 1
            
            # Pequena pausa para visualização
            time.sleep(0.05)
        
        # Coleta estatísticas do episódio
        kills = game.get_game_variable(GameVariable.KILLCOUNT)  # type: ignore
        health = game.get_game_variable(GameVariable.HEALTH)  # type: ignore
        ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)  # type: ignore
        
        total_kills += kills
        total_health += health
        total_ammo += ammo
        total_steps += episode_steps
        
        print(f"Episódio {episode + 1}: Kills={kills}, Health={health}, Ammo={ammo}, Steps={episode_steps}")
        
        # Pausa entre episódios
        if episode < num_episodes - 1:
            print("Pausa de 3 segundos antes do próximo episódio...")
            time.sleep(3)
    
    # Estatísticas finais
    avg_kills = total_kills / num_episodes
    avg_health = total_health / num_episodes
    avg_ammo = total_ammo / num_episodes
    avg_steps = total_steps / num_episodes
    
    print(f"\n{'='*50}")
    print(f"ESTATÍSTICAS FINAIS (média de {num_episodes} episódios):")
    print(f"Kills por episódio: {avg_kills:.1f}")
    print(f"Health por episódio: {avg_health:.1f}")
    print(f"Ammo por episódio: {avg_ammo:.1f}")
    print(f"Steps por episódio: {avg_steps:.1f}")
    print(f"{'='*50}")
    
    game.close()

def list_saved_individuals():
    """
    Lista todos os indivíduos salvos no diretório.
    
    Returns:
        list: Lista de caminhos para os arquivos salvos
    """
    if not os.path.exists(SAVE_DIR):
        print(f"Diretório {SAVE_DIR} não existe.")
        return []
    
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.json')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True)
    
    print(f"\nIndivíduos salvos em '{SAVE_DIR}':")
    for i, filename in enumerate(files):
        filepath = os.path.join(SAVE_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            fitness = data.get('fitness', 'N/A')
            timestamp = data.get('timestamp', 'N/A')
            print(f"{i+1}. {filename}")
            print(f"   Fitness: {fitness}")
            print(f"   Timestamp: {timestamp}")
        except:
            print(f"{i+1}. {filename} (erro ao ler)")
    
    return [os.path.join(SAVE_DIR, f) for f in files]

def create_initial_population(num_actions):
    """Cria a população inicial com genomas aleatórios."""
    population = []
    for _ in range(POPULATION_SIZE):
        # Genoma é uma lista de índices de ações aleatórias
        genome = [random.randint(0, num_actions - 1) for _ in range(GENOME_LENGTH)]
        individual = {
            "genome": genome,
            "fitness": 0.0  # Fitness inicial
        }
        population.append(individual)
    return population

def calculate_fitness(game, individual, actions):
    game.new_episode()
    steps = 0
    total_reward = 0.0
    forward_count = 0
    stuck_counter = 0
    circle_count = 0
    last_turn = None
    turn_repeat = 0
    last_move = None
    last_lateral = None
    side_count = 0

    prev_x = game.get_game_variable(GameVariable.POSITION_X)
    prev_y = game.get_game_variable(GameVariable.POSITION_Y)
    start_x, start_y = prev_x, prev_y

    ammo_start = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
    prev_kills = game.get_game_variable(GameVariable.KILLCOUNT)

    for step, gene_idx in enumerate(individual["genome"]):
        if game.is_episode_finished():
            break
        action = actions[gene_idx]
        reward = game.make_action(action, 1)
        total_reward += reward
        steps += 1
        if action[3] == 1:  # MOVE_FORWARD
            forward_count += 1
            if last_move == 'backward':
                pass
            last_move = 'forward'
        elif action[4] == 1:  # MOVE_BACKWARD
            if last_move == 'forward':
                pass
            last_move = 'backward'
        else:
            last_move = None
        # Penalidade para girar em círculos
        if action[5] == 1:  # TURN_LEFT
            if last_turn == 'left':
                turn_repeat += 1
            else:
                turn_repeat = 1
            last_turn = 'left'
        elif action[6] == 1:  # TURN_RIGHT
            if last_turn == 'right':
                turn_repeat += 1
            else:
                turn_repeat = 1
            last_turn = 'right'
        else:
            last_turn = None
            turn_repeat = 0
        if turn_repeat > 2:
            circle_count += 1
        # Penalidade para ficar parado
        x = game.get_game_variable(GameVariable.POSITION_X)
        y = game.get_game_variable(GameVariable.POSITION_Y)
        move_dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        if move_dist < 1e-3:
            stuck_counter += 1
        prev_x, prev_y = x, y

    kills  = int(game.get_game_variable(GameVariable.KILLCOUNT))
    health = int(game.get_game_variable(GameVariable.HEALTH))
    ammo   = int(game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO))
    steps  = game.get_episode_time()

    # Detectar se chegou ao final (ajuste GOAL_X/GOAL_Y conforme necessário)
    x = game.get_game_variable(GameVariable.POSITION_X)
    y = game.get_game_variable(GameVariable.POSITION_Y)
    GOAL_X, GOAL_Y = 100.0, 50.0  # <-- AJUSTE para o final do seu mapa!
    reached_goal = (np.sqrt((x - GOAL_X)**2 + (y - GOAL_Y)**2) < 5.0)

    # Valorize só os primeiros 4 kills (para não farmar respawn)
    kills_capped = min(kills, 4)
    victory = (kills_capped == 4) and reached_goal

    fitness = 0.0

    # Penalidade forte por morrer
    if health <= 0:
        fitness -= 2000
    else:
        fitness += 4 * health  # Aumenta peso da vida restante

    # Valorize matar até 4 inimigos
    fitness += 2000 * kills_capped
    if reached_goal:
        fitness += 1000  # Bônus por chegar ao final, mesmo sem matar todos
    if victory:
        fitness += 10000  # Vitória total!
    fitness += 1 * ammo    # Bônus pequeno por munição restante

    # Penalidade por stuck
    fitness -= 15 * stuck_counter
    # Penalidade por girar em círculo
    fitness -= 5 * circle_count

    # Bônus progressivo por aproximação do goal
    dist_to_goal = np.sqrt((x - GOAL_X)**2 + (y - GOAL_Y)**2)
    fitness += max(0, 500 - dist_to_goal * 5)

    return fitness, kills_capped, health, ammo, reached_goal, victory, stuck_counter, circle_count, steps

def uniform_crossover(parent1_genome, parent2_genome):
    """
    Realiza o crossover uniforme entre dois genomas.
    """
    assert len(parent1_genome) == len(parent2_genome)
    child1, child2 = [], []
    for g1, g2 in zip(parent1_genome, parent2_genome):
        if random.random() < 0.5:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)
    return child1, child2

def two_point_crossover(parent1_genome, parent2_genome):
    """
    Realiza o crossover de dois pontos entre os genomas de dois pais.
    """
    assert len(parent1_genome) == len(parent2_genome)
    genome_len = len(parent1_genome)
    if genome_len < 2:
        return parent1_genome[:], parent2_genome[:]
    point1 = random.randint(0, genome_len - 2)
    point2 = random.randint(point1 + 1, genome_len - 1)
    child1 = parent1_genome[:point1] + parent2_genome[point1:point2] + parent1_genome[point2:]
    child2 = parent2_genome[:point1] + parent1_genome[point1:point2] + parent2_genome[point2:]
    return child1, child2

def mutate(genome, num_actions):
    """
    Aplica mutação a um genoma com base na MUTATION_RATE.
    """
    mutated_genome = []
    for gene in genome:
        if random.random() < MUTATION_RATE:
            # Se a mutação ocorrer, substitui o gene por uma nova ação aleatória
            mutated_genome.append(random.randint(0, num_actions - 1))
        else:
            # Caso contrário, mantém o gene original
            mutated_genome.append(gene)
    return mutated_genome

def generate_new_population(old_population, num_actions, generation):
    """
    Gera uma nova população completa usando elitismo, seleção, crossover e mutação.
    Inclui introdução de indivíduos aleatórios a cada geração.
    Usa crossover alternado (uniforme e dois pontos).
    """
    sorted_old_population = sorted(old_population, key=lambda x: x['fitness'], reverse=True)
    new_population = []
    for i in range(ELITISM_COUNT):
        new_population.append(sorted_old_population[i])
    # Introduz 10% de indivíduos aleatórios a cada geração
    num_random = max(1, POPULATION_SIZE // 10)
    for _ in range(num_random):
        genome = [random.randint(0, num_actions - 1) for _ in range(GENOME_LENGTH)]
        new_population.append({'genome': genome, 'fitness': 0.0})
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(sorted_old_population)
        parent2 = tournament_selection(sorted_old_population)
        # Alterna entre crossover uniforme e dois pontos
        if random.random() < 0.5:
            child1_genome, child2_genome = uniform_crossover(parent1['genome'], parent2['genome'])
        else:
            child1_genome, child2_genome = two_point_crossover(parent1['genome'], parent2['genome'])
        mutated_child1_genome = mutate(child1_genome, num_actions)
        mutated_child2_genome = mutate(child2_genome, num_actions)
        new_population.append({'genome': mutated_child1_genome, 'fitness': 0.0})
        if len(new_population) < POPULATION_SIZE:
            new_population.append({'genome': mutated_child2_genome, 'fitness': 0.0})
    return new_population

# ==============================================================================
# 3. SCRIPT PRINCIPAL DE AVALIAÇÃO
# ==============================================================================

if __name__ == "__main__":
    print("🎯 ALGORITMO GENÉTICO - TREINAMENTO")
    print("="*50)
    print("Este script executa o treinamento do algoritmo genético.")
    print("Para demonstrar os agentes treinados, use: python demo.py")
    print("="*50)
    
    # --- INICIALIZAÇÃO GERAL ---
    game_instance, possible_actions = initialize_game(headless=True)
    num_possible_actions = len(possible_actions)
    
    print(f"\nCriando população inicial com {POPULATION_SIZE} indivíduos...")
    current_population = create_initial_population(num_possible_actions)
    
    # Variáveis para controlar o critério de parada
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    fitness_history = [] # Para guardar o histórico de fitness de cada geração
    avg_fitness_history = []
    min_fitness_history = []

    # --- SETUP DOS GRÁFICOS EM TEMPO REAL (MÚLTIPLOS SUBPLOTS) ---
    plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    # Fitness
    line_max, = axs[0,0].plot([], [], label='Fitness Máximo')
    line_avg, = axs[0,0].plot([], [], label='Fitness Médio')
    line_min, = axs[0,0].plot([], [], label='Fitness Mínimo')
    axs[0,0].set_title('Fitness')
    axs[0,0].set_xlabel('Geração')
    axs[0,0].set_ylabel('Fitness')
    axs[0,0].legend()
    axs[0,0].grid()
    # Kills
    line_kills_avg, = axs[0,1].plot([], [], label='Kills Médios')
    line_kills_max, = axs[0,1].plot([], [], label='Kills Máximos')
    axs[0,1].set_title('Kills')
    axs[0,1].set_xlabel('Geração')
    axs[0,1].set_ylabel('Kills')
    axs[0,1].legend()
    axs[0,1].grid()
    # Health
    line_health_avg, = axs[0,2].plot([], [], label='Health Médio')
    line_health_max, = axs[0,2].plot([], [], label='Health Máximo')
    axs[0,2].set_title('Health')
    axs[0,2].set_xlabel('Geração')
    axs[0,2].set_ylabel('Health')
    axs[0,2].legend()
    axs[0,2].grid()
    # Ammo
    line_ammo_avg, = axs[1,0].plot([], [], label='Ammo Médio')
    line_ammo_max, = axs[1,0].plot([], [], label='Ammo Máximo')
    axs[1,0].set_title('Ammo')
    axs[1,0].set_xlabel('Geração')
    axs[1,0].set_ylabel('Ammo')
    axs[1,0].legend()
    axs[1,0].grid()
    # % Goal e % Vitória
    line_goal, = axs[1,1].plot([], [], label='% Goal')
    line_victory, = axs[1,1].plot([], [], label='% Vitória')
    axs[1,1].set_title('% Goal e % Vitória')
    axs[1,1].set_xlabel('Geração')
    axs[1,1].set_ylabel('% Indivíduos')
    axs[1,1].legend()
    axs[1,1].grid()
    # Stuck e Circle
    line_stuck, = axs[1,2].plot([], [], label='Stuck Médio')
    line_circle, = axs[1,2].plot([], [], label='Circle Médio')
    axs[1,2].set_title('Stuck e Circle')
    axs[1,2].set_xlabel('Geração')
    axs[1,2].set_ylabel('Contagem')
    axs[1,2].legend()
    axs[1,2].grid()
    plt.tight_layout()
    plt.show()

    # --- LOOP PRINCIPAL DE EVOLUÇÃO ---
    # Históricos para análise
    avg_kills_history = []
    max_kills_history = []
    avg_health_history = []
    max_health_history = []
    avg_ammo_history = []
    max_ammo_history = []
    pct_goal_history = []
    pct_victory_history = []
    avg_stuck_history = []
    avg_circle_history = []
    try:
        for generation in range(MAX_GENERATIONS):
            print(f"\n{'='*20} GERAÇÃO {generation} {'='*20}")
            print(f"Avaliando {len(current_population)} indivíduos...")
            start_time_eval = time.time()
            for i, individual in enumerate(current_population):
                if individual['fitness'] == 0.0:
                    fitness, kills, health, ammo, reached_goal, victory, stuck_counter, circle_count, steps = calculate_fitness(game_instance, individual, possible_actions)
                    individual["fitness"] = fitness
                    individual["kills"] = kills
                    individual["health"] = health
                    individual["ammo"] = ammo
                    individual["reached_goal"] = reached_goal
                    individual["victory"] = victory
                    individual["stuck_counter"] = stuck_counter
                    individual["circle_count"] = circle_count
                    individual["steps"] = steps
            eval_time = time.time() - start_time_eval
            print(f"Avaliação concluída em {eval_time:.2f}s.")
            sorted_population = sorted(current_population, key=lambda x: x['fitness'], reverse=True)
            current_best_fitness = sorted_population[0]['fitness']
            fitness_history.append(current_best_fitness)
            avg_fitness = np.mean([ind['fitness'] for ind in current_population])
            min_fitness = np.min([ind['fitness'] for ind in current_population])
            avg_fitness_history.append(avg_fitness)
            min_fitness_history.append(min_fitness)
            # --- NOVOS HISTÓRICOS ---
            avg_kills_history.append(np.mean([ind['kills'] for ind in current_population]))
            max_kills_history.append(np.max([ind['kills'] for ind in current_population]))
            avg_health_history.append(np.mean([ind['health'] for ind in current_population]))
            max_health_history.append(np.max([ind['health'] for ind in current_population]))
            avg_ammo_history.append(np.mean([ind['ammo'] for ind in current_population]))
            max_ammo_history.append(np.max([ind['ammo'] for ind in current_population]))
            pct_goal_history.append(100.0 * np.mean([ind['reached_goal'] for ind in current_population]))
            pct_victory_history.append(100.0 * np.mean([ind['victory'] for ind in current_population]))
            avg_stuck_history.append(np.mean([ind['stuck_counter'] for ind in current_population]))
            avg_circle_history.append(np.mean([ind['circle_count'] for ind in current_population]))
            # Atualiza gráficos em tempo real (todos os subplots)
            x_vals = range(len(fitness_history))
            # Fitness
            line_max.set_data(x_vals, fitness_history)
            line_avg.set_data(x_vals, avg_fitness_history)
            line_min.set_data(x_vals, min_fitness_history)
            axs[0,0].relim(); axs[0,0].autoscale_view()
            # Kills
            line_kills_avg.set_data(x_vals, avg_kills_history)
            line_kills_max.set_data(x_vals, max_kills_history)
            axs[0,1].relim(); axs[0,1].autoscale_view()
            # Health
            line_health_avg.set_data(x_vals, avg_health_history)
            line_health_max.set_data(x_vals, max_health_history)
            axs[0,2].relim(); axs[0,2].autoscale_view()
            # Ammo
            line_ammo_avg.set_data(x_vals, avg_ammo_history)
            line_ammo_max.set_data(x_vals, max_ammo_history)
            axs[1,0].relim(); axs[1,0].autoscale_view()
            # % Goal e % Vitória
            line_goal.set_data(x_vals, pct_goal_history)
            line_victory.set_data(x_vals, pct_victory_history)
            axs[1,1].relim(); axs[1,1].autoscale_view()
            # Stuck e Circle
            line_stuck.set_data(x_vals, avg_stuck_history)
            line_circle.set_data(x_vals, avg_circle_history)
            axs[1,2].relim(); axs[1,2].autoscale_view()
            fig.canvas.draw(); fig.canvas.flush_events();
            plt.pause(0.01)
            # Lógica do Critério de Parada
            if current_best_fitness > best_fitness_overall + IMPROVEMENT_THRESHOLD:
                best_fitness_overall = current_best_fitness
                generations_without_improvement = 0
                print(f"✨ Nova melhoria significativa encontrada! Melhor fitness geral: {best_fitness_overall:.2f}")
                save_individual(sorted_population[0])
            else:
                generations_without_improvement += 1
                print(f"Sem melhoria significativa. Gerações estagnadas: {generations_without_improvement}/{STAGNATION_LIMIT}")
            if generations_without_improvement >= STAGNATION_LIMIT:
                print(f"CRITÉRIO DE PARADA ATINGIDO: Ausência de melhoria por {STAGNATION_LIMIT} gerações.")
                break
            # --- Mutação adaptativa mais agressiva ---
            if generations_without_improvement > 100:
                MUTATION_RATE = min(0.7, MUTATION_RATE * 1.2)
            else:
                MUTATION_RATE = 0.2
            # --- GERAÇÃO DA PRÓXIMA POPULAÇÃO ---
            print("Gerando a próxima população...")
            current_population = generate_new_population(sorted_population, num_possible_actions, generation)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário! Finalizando e salvando resultados...")

    # --- FIM DA EVOLUÇÃO ---
    plt.ioff()  # Desliga modo interativo
    print("\n" + "="*50)
    print("Evolução finalizada.")
    final_best_individual = sorted(current_population, key=lambda x: x['fitness'], reverse=True)[0]
    print(f"Melhor fitness final alcançado: {final_best_individual['fitness']:.2f}")
    print(f"Executado por {len(fitness_history)} gerações.")
    save_individual(final_best_individual, "best_individual_final.json")
    game_instance.close()
    print("Instância do jogo finalizada. Processo concluído.")
    print("\n✅ Treinamento finalizado!")
    print("📁 Indivíduos salvos em: saved_individuals/")
    print("🎮 Para demonstrar os agentes, execute: python demo.py")
    # --- GRÁFICO DE PROGRESSO FINAL ---
    plt.figure(figsize=(10,6))
    plt.plot(fitness_history, label='Fitness Máximo')
    plt.plot(avg_fitness_history, label='Fitness Médio')
    plt.plot(min_fitness_history, label='Fitness Mínimo')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Progresso do Algoritmo Genético')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('fitness_progress.png')
    plt.show()
    # --- HISTOGRAMA DA ÚLTIMA GERAÇÃO ---
    plt.figure()
    fitness_values = [ind['fitness'] for ind in current_population]
    plt.hist(fitness_values, bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribuição do Fitness na Última Geração')
    plt.xlabel('Fitness')
    plt.ylabel('Número de Indivíduos')
    plt.tight_layout()
    plt.savefig('fitness_histogram.png')
    plt.show()
    # --- SCATTER PLOT FITNESS vs ÍNDICE ---
    plt.figure()
    plt.scatter(range(len(current_population)), fitness_values, color='red')
    plt.title('Fitness dos Indivíduos na Última Geração')
    plt.xlabel('Índice do Indivíduo')
    plt.ylabel('Fitness')
    plt.tight_layout()
    plt.savefig('fitness_scatter.png')
    plt.show()
    # --- GRÁFICOS AVANÇADOS DE COMPONENTES ---
    # Evolução dos componentes
    plt.figure(figsize=(10,6))
    plt.plot(avg_kills_history, label='Kills Médios')
    plt.plot(max_kills_history, label='Kills Máximos')
    plt.xlabel('Geração')
    plt.ylabel('Kills')
    plt.title('Evolução dos Kills por Geração')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('evolucao_kills.png')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(avg_health_history, label='Health Médio')
    plt.plot(max_health_history, label='Health Máximo')
    plt.xlabel('Geração')
    plt.ylabel('Health')
    plt.title('Evolução do Health por Geração')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('evolucao_health.png')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(avg_ammo_history, label='Ammo Médio')
    plt.plot(max_ammo_history, label='Ammo Máximo')
    plt.xlabel('Geração')
    plt.ylabel('Ammo')
    plt.title('Evolução do Ammo por Geração')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('evolucao_ammo.png')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(pct_goal_history, label='% Chegaram ao Goal')
    plt.plot(pct_victory_history, label='% Vitória Total')
    plt.xlabel('Geração')
    plt.ylabel('% Indivíduos')
    plt.title('Porcentagem de Goal e Vitória por Geração')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('evolucao_goal_victory.png')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(avg_stuck_history, label='Stuck Médio')
    plt.plot(avg_circle_history, label='Circle Médio')
    plt.xlabel('Geração')
    plt.ylabel('Contagem')
    plt.title('Stuck e Circle Count Médios por Geração')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('evolucao_stuck_circle.png')
    plt.show()

    # Boxplots dos componentes na última geração
    plt.figure(figsize=(10,6))
    plt.boxplot([ [ind['kills'] for ind in current_population],
                  [ind['health'] for ind in current_population],
                  [ind['ammo'] for ind in current_population] ],
                labels=['Kills', 'Health', 'Ammo'])
    plt.title('Boxplot dos Componentes na Última Geração')
    plt.tight_layout()
    plt.savefig('boxplot_componentes_ultima_geracao.png')
    plt.show()