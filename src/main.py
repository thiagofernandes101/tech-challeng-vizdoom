from itertools import product
import random
import time
from typing import List
from models.game_interface import GameInterface
from models.individual import Individual
from models.movement import Movement
from models.genome import Genome
from utils.genetic import Genetic

MOVEMENT_LIMIT = 4500 


SCENARIO_PATH = "deadly_corridor.cfg"
POPULATION_SIZE = 100 

W_KILLS = 150.0
W_HEALTH = 1.0
W_AMMO = 0.2
ITEM_COUNT = 5.0
DAMEGE_COUNT = 10.0
DAMAGE_TAKEN = -0.5
MISSING_SHOT = -0.8
GAME_PROGRESS = 0.4


MAX_GENERATIONS = 999999    # O número máximo de gerações que o algoritmo irá executar
STAGNATION_LIMIT = 10000    # N: O número de gerações sem melhoria antes de parar
IMPROVEMENT_THRESHOLD = 0.1 # A melhoria mínima no fitness para ser considerada "significativa"

CONVERGENCE = 10

ELITISM_COUNT = 3
TOURNAMENT_SIZE = 3 
MUTATION_RATE = 0.5 

def initialize_game() -> GameInterface:
    """Cria e configura a instância do jogo ViZDoom."""
    print("Inicializando ViZDoom...")
    return GameInterface(SCENARIO_PATH, True)

def all_valid_moviments() -> list[Movement]:
    movimentos_validos = []
    
    for comando in product([0, 1], repeat=9):
        move = Movement.from_list(list(comando))

        if move.no_action():
            continue
        if move.move_left and move.move_right:
            continue
        if move.move_forward and move.move_backward:
            continue
        if move.turn_left and move.turn_right:
            continue

        movimentos_validos.append(move)
    
    print(f"Cardápio de ações definido com {len(movimentos_validos)} movimentos possíveis.")
    return movimentos_validos

def create_initial_population(limit_per_individual: int, possible_movements: List[Movement]) -> List[Individual]:
    population: list[Individual] = []
    for _ in range(POPULATION_SIZE):
        movements = random.choices(possible_movements, k=limit_per_individual)
        population.append(Individual([Genome(movement) for movement in movements]))
    return population

def calculate_fitness(game_interface: GameInterface, individuals: List[Individual]) -> List[Individual]:
    for individual in individuals:
        if individual.evaluated:
            continue
        game_interface.start_episode()
        for genome in individual.genomes:
            if game_interface.episode_is_finished():
                individual.fitness = game_interface.get_fitness()
                break
            step_evaluation = game_interface.make_action(genome.movement)
            genome.movement_side_effect = step_evaluation.step_results()
    return individuals

if __name__ == "__main__":

    genetic = Genetic(ELITISM_COUNT, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE)
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    moviments = all_valid_moviments()
    game_interface = initialize_game()
    fitness_history: list[int] = []
    individuals = create_initial_population(MOVEMENT_LIMIT, moviments)
    for generation in range(MAX_GENERATIONS):
        print(f"\n{'='*20} GERAÇÃO {generation} {'='*20}")

        print(f"Avaliando {len(individuals)} indivíduos...")
        start_time_eval = time.time()
        population = calculate_fitness(game_interface, individuals)

        eval_time = time.time() - start_time_eval
        print(f"Avaliação concluída em {eval_time:.2f}s.")

        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)

        current_best_fitness = sorted_population[0].fitness
        fitness_history.append(current_best_fitness)
        print(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")

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

        print("Gerando a próxima população...")
        
        individuals = genetic.generate_new_population(sorted_population, moviments, generations_without_improvement > CONVERGENCE)

    print("\n" + "="*50)
    print("Evolução finalizada.")
    
    final_best_individual = sorted(individuals, key=lambda x: x.fitness, reverse=True)[0]
    print(f"Melhor fitness final alcançado: {final_best_individual['fitness']:.2f}")
    print(f"Executado por {len(fitness_history)} gerações.")

    game_interface.close()
    print("Instância do jogo finalizada. Processo concluído.")