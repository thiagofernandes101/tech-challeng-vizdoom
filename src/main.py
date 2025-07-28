from typing import List
import vizdoom as vzd
import numpy as np
import random
import time
from itertools import product

from models.genome import Genome
from models.individual import Individual
from models.step_evaluation import StepEvaluation
from models.game_interface import GameInterface

# ==============================================================================
# 1. PARÂMETROS DO ALGORITMO GENÉTICO E DO JOGO
# ==============================================================================

# --- Parâmetros da População ---
MOVEMENT_ = 4500 
POPULATION_SIZE = 100  # Tamanho da população (100 indivíduos)

# --- Pesos da Função de Fitness (Ajuste estes valores para guiar a evolução!) ---
# O objetivo é maximizar essa pontuação
W_KILLS = 150.0      # Peso para cada inimigo morto
W_HEALTH = 1.0       # Peso para cada ponto de vida restante
W_AMMO = 0.2         # Peso para cada ponto de munição restante
ITEM_COUNT = 5.0
DAMEGE_COUNT = 10.0
DAMAGE_TAKEN = -0.5
MISSING_SHOT = -0.8
GAME_PROGRESS = 0.4

# --- Parâmetros dos Operadores Genéticos ---
TOURNAMENT_SIZE = 3     # Número de indivíduos que competem em cada torneio de seleção
MUTATION_RATE = 0.5    # Probabilidade de um gene sofrer mutação (2%)
ELITISM_COUNT = 5       # Número de melhores indivíduos a serem passados diretamente para a próxima geração

# --- Parâmetros de Critério de Parada ---
MAX_GENERATIONS = 999999    # O número máximo de gerações que o algoritmo irá executar
STAGNATION_LIMIT = 10000    # N: O número de gerações sem melhoria antes de parar
IMPROVEMENT_THRESHOLD = 0.1 # A melhoria mínima no fitness para ser considerada "significativa"

# --- Configuração do ViZDoom ---
SCENARIO_PATH = "deadly_corridor.cfg"  # Nome do arquivo de configuração do cenário

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def initialize_game() -> tuple[GameInterface, List[List[int]]]:
    """Cria e configura a instância do jogo ViZDoom."""
    print("Inicializando ViZDoom...")
    game_interface = GameInterface(SCENARIO_PATH, True)
    
    num_buttons = game_interface.get_avaliable_buttons_amount()
    
    # Gera todas as combinações possíveis de botões (0 ou 1 para cada)
    # product([0, 1], repeat=num_buttons) -> (0,0,0,0), (0,0,0,1), (0,0,1,0), ...
    actions = [list(p) for p in product([0, 1], repeat=num_buttons)]
    
    # Opcional: Remova a ação "não fazer nada" se não for desejada
    actions.remove([0] * num_buttons)
    
    print(f"Cardápio de ações definido com {len(actions)} movimentos possíveis.")
    
    return game_interface, actions

def create_initial_population(num_actions: int) -> List[Individual]:
    """Cria a população inicial com genomas aleatórios."""
    population = []
    for _ in range(POPULATION_SIZE):
        # Genoma é uma lista de índices de ações aleatórias
        genome = [Genome(random.randint(0, num_actions - 1)) for _ in range(GENOME_LENGTH)]
        individual = Individual(genome)
        population.append(individual)
    return population

def calculate_fitness(game_interface: GameInterface, individual: Individual, actions: List[List[int]]) -> tuple[int, List[int]]:
    """
    Executa um episódio do Doom para um indivíduo e calcula seu fitness.
    Esta é a função mais importante e demorada do processo.
    """
    step_results = []
    wrong_shot = 0
    episode_progress = 0
    game_interface.start_episode()

    # Executa cada ação (gene) do genoma do indivíduo
    for genome in individual.genomes:
        if game_interface.episode_is_finished():
            break

        action_to_perform = actions[genome.action_index]
        damege_count_before_action = game_interface.get_damage_count()
        step_evaluation = StepEvaluation(action_to_perform, 
                                        game_interface.get_kill_count(), 
                                        game_interface.get_current_healt(),
                                        game_interface.get_items_count(),
                                        damege_count_before_action
                                        )
        game_interface.make_action(action_to_perform)

        try:
            episode_progress = game_interface.get_current_y()
        except RuntimeError:
            pass

        step_evaluation.kills_after = game_interface.get_kill_count()
        step_evaluation.health_after = game_interface.get_current_healt()
        step_evaluation.items_after = game_interface.get_items_count()
        step_evaluation.damega_count_after = game_interface.get_damage_count()
        step_results.append(step_evaluation.step_results())

        if (damege_count_before_action == game_interface.get_damage_count() and action_to_perform[0] == 1):
            wrong_shot += 1

    fitness_score = (W_KILLS * game_interface.get_kill_count()) + \
                    (W_HEALTH * game_interface.get_current_healt()) + \
                    (W_AMMO * game_interface.get_selected_weapon_ammo()) + \
                    (ITEM_COUNT * game_interface.get_items_count()) +\
                    (DAMEGE_COUNT * game_interface.get_damage_count()) +\
                    (DAMAGE_TAKEN * game_interface.get_damage_taken()) +\
                    (MISSING_SHOT * wrong_shot) +\
                    (GAME_PROGRESS * episode_progress)
    


    return fitness_score, step_results + [0] * (GENOME_LENGTH - len(step_results))

def tournament_selection(population: List[Individual]) -> Individual:
    """
    Seleciona um indivíduo vencedor de um torneio.
    """
    # Seleciona 'TOURNAMENT_SIZE' indivíduos aleatórios da população
    tournament_competitors = random.sample(population, TOURNAMENT_SIZE)
    
    # O vencedor é aquele com o maior fitness
    winner = max(tournament_competitors, key=lambda x: x.fitness)
    return winner

def one_point_crossover(parent1_genome: List[Genome], parent2_genome: List[Genome])-> tuple[List[Genome], List[Genome]]:
    """
    Realiza o crossover de um ponto entre os genomas de dois pais.
    """
    assert len(parent1_genome) == len(parent2_genome)
    
    negative_gene_1 = [(i, g) for i, g in enumerate(parent1_genome) if g.action_side_effect < 0]
    negative_gene_2 = [(i, g) for i, g in enumerate(parent1_genome) if g.action_side_effect < 0]

    for (i_1, gene_1), (i_2, gene_2) in zip(negative_gene_1, negative_gene_2):
        parent1_genome[i_1] = Genome(gene_2.action_index)
        parent2_genome[i_2] = Genome(gene_1.action_index)

    
    return parent1_genome, parent2_genome

def mutate(genomes: List[Genome], num_actions: int) -> List[Genome]:
    """
    Aplica mutação a um genoma com base na MUTATION_RATE.
    """
    mutated_genome = []
    for gene in genomes:
        if (gene.action_side_effect < 0):
            new_action = random.randint(0, num_actions - 1)
            while new_action == gene.action_index:
                new_action = random.randint(0, num_actions - 1)
            mutated_genome.append(Genome(new_action))
        elif (gene.action_side_effect == 0):
            if random.random() < MUTATION_RATE:
                mutated_genome.append(Genome(random.randint(0, num_actions - 1)))
            else:
                mutated_genome.append(gene)
        else:
            mutated_genome.append(gene)
    return mutated_genome

def generate_new_population(old_population: List[Individual], num_actions: int, populate_stagnated: bool) -> List[Individual]:
    """
    Gera uma nova população completa usando elitismo, seleção, crossover e mutação.
    """
    # Primeiro, ordena a população antiga para encontrar o melhor indivíduo
    sorted_old_population = sorted(old_population, key=lambda x: x.fitness, reverse=True)
    
    new_population = []
    
    # 1. Elitismo: Adiciona os melhores indivíduos diretamente à nova população
    # Regra 1 e 5: Manter o melhor indivíduo sem alterações.
    new_population = sorted_old_population[:ELITISM_COUNT]

    if populate_stagnated:
        del sorted_old_population[-ELITISM_COUNT:]
    else:
        del sorted_old_population[:ELITISM_COUNT]
        del sorted_old_population[-ELITISM_COUNT:]
        
    # 2. Geração dos Indivíduos Restantes (99, neste caso)
    # Regra 5: O restante da população é gerado pelo processo evolutivo.
    while len(new_population) < POPULATION_SIZE:
        # Seleção dos pais
        parent1 = tournament_selection(sorted_old_population)
        parent2 = tournament_selection(sorted_old_population)
        
        # Crossover para criar os filhos ( trocando genes negativos entre eles)
        child1_genome, child2_genome = one_point_crossover(parent1.genomes, parent2.genomes)
        
        # Mutação dos filhos ( caso ainda tenham genes negativos)
        child1_genome = mutate(child1_genome, num_actions)
        
        child2_genome = mutate(child2_genome, num_actions)
        
        # Adiciona os novos filhos à população
        # O fitness é zerado, pois eles ainda não foram avaliados
        new_population.append(Individual(child1_genome))
        if len(new_population) < POPULATION_SIZE:
            new_population.append(Individual(child2_genome))
            
    return new_population


# ==============================================================================
# 3. SCRIPT PRINCIPAL DE AVALIAÇÃO
# ==============================================================================

if __name__ == "__main__":
    
    # --- INICIALIZAÇÃO GERAL ---
    game_interface, possible_actions = initialize_game()
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
            if individual.wainting_evaluated:
                # Descomente a linha abaixo para ver o progresso da avaliação
                # print(f"Avaliando indivíduo {i + 1}/{POPULATION_SIZE}...", end='\r')
                fitness, steps_evaluations = calculate_fitness(game_interface, individual, possible_actions)
                individual.fitness = fitness
                individual.evaluate_steps(steps_evaluations)
        
        eval_time = time.time() - start_time_eval
        print(f"Avaliação concluída em {eval_time:.2f}s.")

        # 2. SELEÇÃO DE DADOS E VERIFICAÇÃO DE PARADA
        sorted_population = sorted(current_population, key=lambda x: x.fitness, reverse=True)
        current_best_fitness = sorted_population[0].fitness
        fitness_history.append(current_best_fitness)
        
        print(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")

        # Lógica do Critério de Parada
        if current_best_fitness > best_fitness_overall + IMPROVEMENT_THRESHOLD:
            best_fitness_overall = current_best_fitness
            generations_without_improvement = 0
            print(f"✨ Nova melhoria significativa encontrada! Melhor fitness geral: {best_fitness_overall:.2f}")
            # Salvar o melhor indivíduo pode ser uma boa ideia aqui
            # np.save('best_genome.npy', sorted_population[0].genome.)
        else:
            generations_without_improvement += 1
            print(f"Sem melhoria significativa. Gerações estagnadas: {generations_without_improvement}/{STAGNATION_LIMIT}")

        if generations_without_improvement >= STAGNATION_LIMIT:
            print(f"CRITÉRIO DE PARADA ATINGIDO: Ausência de melhoria por {STAGNATION_LIMIT} gerações.")
            break
        
        # 3. GERAÇÃO DA PRÓXIMA POPULAÇÃO
        print("Gerando a próxima população...")
        current_population = generate_new_population(sorted_population, num_possible_actions, generations_without_improvement > 10)

    # --- FIM DA EVOLUÇÃO ---
    print("\n" + "="*50)
    print("Evolução finalizada.")
    
    final_best_individual = sorted(current_population, key=lambda x: x['fitness'], reverse=True)[0]
    print(f"Melhor fitness final alcançado: {final_best_individual['fitness']:.2f}")
    print(f"Executado por {len(fitness_history)} gerações.")

    game_interface.close()
    print("Instância do jogo finalizada. Processo concluído.")