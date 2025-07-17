import vizdoom as vzd
import numpy as np
import random
import time
from itertools import product

# ==============================================================================
# 1. PARÂMETROS DO ALGORITMO GENÉTICO E DO JOGO
# ==============================================================================

# --- Parâmetros da População ---
POPULATION_SIZE = 100  # Tamanho da população (100 indivíduos)
GENOME_LENGTH = 1000   # Número máximo de ações por episódio (duração da "vida" do agente)

# --- Pesos da Função de Fitness (Ajuste estes valores para guiar a evolução!) ---
# O objetivo é maximizar essa pontuação
W_KILLS = 150.0      # Peso para cada inimigo morto
W_HEALTH = 1.0       # Peso para cada ponto de vida restante
W_AMMO = 0.2         # Peso para cada ponto de munição restante
W_STEPS = -0.5       # Penalidade por passo dado (incentiva a terminar rápido)

# --- Parâmetros dos Operadores Genéticos ---
TOURNAMENT_SIZE = 3     # Número de indivíduos que competem em cada torneio de seleção
MUTATION_RATE = 0.5    # Probabilidade de um gene sofrer mutação (2%)
ELITISM_COUNT = 1       # Número de melhores indivíduos a serem passados diretamente para a próxima geração

# --- Parâmetros de Critério de Parada ---
MAX_GENERATIONS = 999999    # O número máximo de gerações que o algoritmo irá executar
STAGNATION_LIMIT = 10000    # N: O número de gerações sem melhoria antes de parar
IMPROVEMENT_THRESHOLD = 0.1 # A melhoria mínima no fitness para ser considerada "significativa"

# --- Configuração do ViZDoom ---
SCENARIO_PATH = "deadly_corridor.cfg"  # Nome do arquivo de configuração do cenário

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def initialize_game():
    """Cria e configura a instância do jogo ViZDoom."""
    print("Inicializando ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)
    
    # Roda o jogo em modo "background", sem janela, para máxima velocidade de treino
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_available_buttons([
        vzd.Button.ATTACK,
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.MOVE_UP,
        vzd.Button.MOVE_DOWN
    ])
    game.init()
    
    # Define as ações possíveis que o agente pode tomar
    # Cada ação é uma combinação de botões
    # Ex: [0, 0, 1] -> Atirar
    # actions = [random.randint(0, game.get_available_buttons_size() -1) for _ in range(GENOME_LENGTH)]
    # return game, actions
    
    # Crie o "cardápio" de ações possíveis.
    # Cada ação é uma lista de 0s e 1s com o mesmo tamanho do número de botões.
    
    num_buttons = game.get_available_buttons_size()
    
    # Gera todas as combinações possíveis de botões (0 ou 1 para cada)
    # product([0, 1], repeat=num_buttons) -> (0,0,0,0), (0,0,0,1), (0,0,1,0), ...
    actions = [list(p) for p in product([0, 1], repeat=num_buttons)]
    
    # Opcional: Remova a ação "não fazer nada" se não for desejada
    actions.remove([0] * num_buttons)
    
    print(f"Cardápio de ações definido com {len(actions)} movimentos possíveis.")
    
    return game, actions

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
    """
    Executa um episódio do Doom para um indivíduo e calcula seu fitness.
    Esta é a função mais importante e demorada do processo.
    """
    game.new_episode()
    
    # Executa cada ação (gene) do genoma do indivíduo
    for action_index in individual["genome"]:
        if game.is_episode_finished():
            break
        
        # Ação é a combinação de botões correspondente ao índice
        action_to_perform = actions[action_index]
        game.make_action(action_to_perform)  # Corrigido: Removido os colchetes extras

    # Coleta os resultados no final do episódio
    if game.is_episode_finished():
        # Se o episódio terminou (morreu ou completou), coletamos os dados finais
        kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        steps_taken = game.get_episode_time()
    else:
        # Se o episódio não terminou (atingiu o limite de ações), o agente está "vivo"
        # mas pode não ter feito nada útil.
        kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        steps_taken = GENOME_LENGTH

    # Aplica a fórmula de fitness com os pesos definidos
    fitness_score = (W_KILLS * kills) + \
                    (W_HEALTH * health) + \
                    (W_AMMO * ammo) + \
                    (W_STEPS * steps_taken)
                    
    return fitness_score

def tournament_selection(population):
    """
    Seleciona um indivíduo vencedor de um torneio.
    """
    # Seleciona 'TOURNAMENT_SIZE' indivíduos aleatórios da população
    tournament_competitors = random.sample(population, TOURNAMENT_SIZE)
    
    # O vencedor é aquele com o maior fitness
    winner = max(tournament_competitors, key=lambda x: x['fitness'])
    return winner

def one_point_crossover(parent1_genome, parent2_genome):
    """
    Realiza o crossover de um ponto entre os genomas de dois pais.
    """
    assert len(parent1_genome) == len(parent2_genome)
    genome_len = len(parent1_genome)
    
    # Escolhe um ponto de corte aleatório, exceto nas extremidades
    crossover_point = random.randint(1, genome_len - 1)
    
    # Cria os genomas dos filhos combinando as partes dos pais
    child1_genome = parent1_genome[:crossover_point] + parent2_genome[crossover_point:]
    child2_genome = parent2_genome[:crossover_point] + parent1_genome[crossover_point:]
    
    return child1_genome, child2_genome

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

def generate_new_population(old_population, num_actions):
    """
    Gera uma nova população completa usando elitismo, seleção, crossover e mutação.
    """
    # Primeiro, ordena a população antiga para encontrar o melhor indivíduo
    sorted_old_population = sorted(old_population, key=lambda x: x['fitness'], reverse=True)
    
    new_population = []
    
    # 1. Elitismo: Adiciona os melhores indivíduos diretamente à nova população
    # Regra 1 e 5: Manter o melhor indivíduo sem alterações.
    for i in range(ELITISM_COUNT):
        new_population.append(sorted_old_population[i])
        
    # 2. Geração dos Indivíduos Restantes (99, neste caso)
    # Regra 5: O restante da população é gerado pelo processo evolutivo.
    while len(new_population) < POPULATION_SIZE:
        # Seleção dos pais
        parent1 = tournament_selection(sorted_old_population)
        parent2 = tournament_selection(sorted_old_population)
        
        # Crossover para criar os filhos
        child1_genome, child2_genome = one_point_crossover(parent1['genome'], parent2['genome'])
        
        # Mutação dos filhos
        mutated_child1_genome = mutate(child1_genome, num_actions)
        mutated_child2_genome = mutate(child2_genome, num_actions)
        
        # Adiciona os novos filhos à população
        # O fitness é zerado, pois eles ainda não foram avaliados
        new_population.append({'genome': mutated_child1_genome, 'fitness': 0.0})
        if len(new_population) < POPULATION_SIZE:
            new_population.append({'genome': mutated_child2_genome, 'fitness': 0.0})
            
    return new_population

# ==============================================================================
# 3. SCRIPT PRINCIPAL DE AVALIAÇÃO
# ==============================================================================

if __name__ == "__main__":
    
    # --- INICIALIZAÇÃO GERAL ---
    game_instance, possible_actions = initialize_game()
    num_possible_actions = len(possible_actions)
    
    print(f"\nCriando população inicial com {POPULATION_SIZE} indivíduos...")
    current_population = create_initial_population(num_possible_actions)
    
    # Variáveis para controlar o critério de parada
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    fitness_history = [] # Para guardar o histórico de fitness de cada geração

    # --- LOOP PRINCIPAL DE EVOLUÇÃO ---
    for generation in range(MAX_GENERATIONS):
        print(f"\n{'='*20} GERAÇÃO {generation} {'='*20}")

        # 1. AVALIAÇÃO DA POPULAÇÃO ATUAL
        print(f"Avaliando {len(current_population)} indivíduos...")
        start_time_eval = time.time()
        # Nota: Só avaliamos os indivíduos que ainda não têm fitness (os gerados, não o elite)
        for i, individual in enumerate(current_population):
            if individual['fitness'] == 0.0: # Fitness 0.0 indica um novo indivíduo a ser avaliado
                 # Descomente a linha abaixo para ver o progresso da avaliação
                 # print(f"Avaliando indivíduo {i + 1}/{POPULATION_SIZE}...", end='\r')
                 fitness = calculate_fitness(game_instance, individual, possible_actions)
                 individual["fitness"] = fitness
        
        eval_time = time.time() - start_time_eval
        print(f"Avaliação concluída em {eval_time:.2f}s.")

        # 2. SELEÇÃO DE DADOS E VERIFICAÇÃO DE PARADA
        sorted_population = sorted(current_population, key=lambda x: x['fitness'], reverse=True)
        current_best_fitness = sorted_population[0]['fitness']
        fitness_history.append(current_best_fitness)
        
        print(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")

        # Lógica do Critério de Parada
        if current_best_fitness > best_fitness_overall + IMPROVEMENT_THRESHOLD:
            best_fitness_overall = current_best_fitness
            generations_without_improvement = 0
            print(f"✨ Nova melhoria significativa encontrada! Melhor fitness geral: {best_fitness_overall:.2f}")
            # Salvar o melhor indivíduo pode ser uma boa ideia aqui
            # np.save('best_genome.npy', sorted_population[0]['genome'])
        else:
            generations_without_improvement += 1
            print(f"Sem melhoria significativa. Gerações estagnadas: {generations_without_improvement}/{STAGNATION_LIMIT}")

        if generations_without_improvement >= STAGNATION_LIMIT:
            print(f"CRITÉRIO DE PARADA ATINGIDO: Ausência de melhoria por {STAGNATION_LIMIT} gerações.")
            break
        
        # 3. GERAÇÃO DA PRÓXIMA POPULAÇÃO
        print("Gerando a próxima população...")
        current_population = generate_new_population(sorted_population, num_possible_actions)

    # --- FIM DA EVOLUÇÃO ---
    print("\n" + "="*50)
    print("Evolução finalizada.")
    
    final_best_individual = sorted(current_population, key=lambda x: x['fitness'], reverse=True)[0]
    print(f"Melhor fitness final alcançado: {final_best_individual['fitness']:.2f}")
    print(f"Executado por {len(fitness_history)} gerações.")

    game_instance.close()
    print("Instância do jogo finalizada. Processo concluído.")