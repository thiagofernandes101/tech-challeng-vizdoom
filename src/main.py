from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import time
from typing import List
from models.game_interface import GameInfo, GameInterface
from models.individual import Individual
from models.movement import Movement
from models.genome import Genome
import numpy as np
from models.game_element import GameElement
from models.individual_info import IndividualInfo
from models.result_manager import ResultManager
from utils.mapper import Mapper
from utils.simple_nn import SimpleNN
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

STAGNATION_LIMIT = 100
IMPROVEMENT_THRESHOLD = 0.1

CONVERGENCE = 10

ELITISM_COUNT = 3
TOURNAMENT_SIZE = 3 
MUTATION_RATE = 0.5 

NEURAL_INPUTS = 40
HIDDEN_SIZE = 222

def initialize_game() -> GameInterface:
    """Cria e configura a instância do jogo ViZDoom."""
    print("Inicializando ViZDoom...")
    return GameInterface(SCENARIO_PATH)

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

def generate_info_vector(game_interface: GameInterface, player: GameElement, elements: List[GameElement])-> np.ndarray:
    ELEMENT_TYPE_MAP = {
        'Enemy': 1,
        'Player': 2,
        'Colectable': 3,
        'Blood': 4,
        'Targer': 5,
    }
    episode_info_vector: list[float] = []
    episode_info_vector.append(game_interface.get_state_info(GameInfo.HEALTH) / 100)
    episode_info_vector.append(game_interface.get_state_info(GameInfo.ITEMS_COUNT) / 10)
    episode_info_vector.append(game_interface.get_state_info(GameInfo.DAMAGE_COUNT) / 100)
    episode_info_vector.append(game_interface.get_state_info(GameInfo.KILL_COUNT) / 10)
    episode_info_vector.append(game_interface.get_state_info(GameInfo.DAMAGE_TAKEN) / 100)
    episode_info_vector.append(game_interface.get_state_info(GameInfo.WEAPON_AMMO) / 100)

    episode_info_vector.append(game_interface.get_current_x() / 1000)
    episode_info_vector.append(game_interface.get_current_y() / 1000)
    episode_info_vector.append(player.angle / 1000)

    for element in elements:
        episode_info_vector.append(element.pos_x / 1000)
        episode_info_vector.append(element.pos_y / 1000)
        episode_info_vector.append(element.angle / 1000)
        element_class_name = type(element).__name__
        type_value = ELEMENT_TYPE_MAP.get(element_class_name, 0)
        episode_info_vector.append(type_value / 10)

    if len(episode_info_vector) < NEURAL_INPUTS:
        episode_info_vector += [0.0] * (NEURAL_INPUTS - len(episode_info_vector))

    return np.array(episode_info_vector).reshape(-1, 1)

def create_initial_population(game_interface: GameInterface, simple_nn: SimpleNN, valids_moves: list[Movement]) -> tuple[list[Individual], list[IndividualInfo]]:
    population: list[Individual] = []
    population_info: list[IndividualInfo] = []
    for _ in range(POPULATION_SIZE):
        game_interface.start_episode()
        individual = Individual()
        while not game_interface.episode_is_finished():
            elements, player = game_interface.get_visible_elements()
            input_vector = generate_info_vector(game_interface, player, elements)
            output = simple_nn.forward(input_vector)
            genome = Mapper.neural_output_to_moviment(output, valids_moves)
            genome.neural_output = output
            individual.inc_genome(genome)
            game_interface.make_action(genome.movement)
        individual_info = game_interface.individual_info()
        population_info.append(individual_info)
        individual.fitness = game_interface.get_fitness()
        population.append(individual)

    return population, population_info

def evaluate_mutated_individuals(game_interface: GameInterface, new_generation: list[Individual]) -> tuple[list[Individual], list[IndividualInfo]]:
    population_info: list[IndividualInfo] = []
    for individual in new_generation:
        game_interface.start_episode()
        for genome in individual.genomes:
            if game_interface.episode_is_finished():
                individual_info = game_interface.individual_info()
                population_info.append(individual_info)
                break
            game_interface.make_action(genome.movement)
        individual.fitness = game_interface.get_fitness()
    return new_generation, population_info

if __name__ == "__main__":

    rng = np.random.default_rng(seed=42)
    moviments = all_valid_moviments()
    genetic = Genetic(ELITISM_COUNT, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, moviments, rng)
    simple_nn = SimpleNN(NEURAL_INPUTS, HIDDEN_SIZE, 9, rng)
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    game_interface = initialize_game()

    fitness_history: list[int] = []
    
    print(f"Avaliando {POPULATION_SIZE} indivíduos da população Zero...")
    start_time_eval = time.time()
    populations_info: dict[int, IndividualInfo] = {}
    individuals, population_info = create_initial_population(game_interface, simple_nn, moviments)
    populations_info[0] = population_info
    eval_time = time.time() - start_time_eval
    print(f"Avaliação concluída em {eval_time:.2f}s.")

    sorted_population = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    current_best_fitness = sorted_population[0].fitness
    fitness_history.append(current_best_fitness)
    print(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")

    population = 1
    while True:
        if current_best_fitness > best_fitness_overall + IMPROVEMENT_THRESHOLD:
            best_fitness_overall = current_best_fitness
            generations_without_improvement = 0
            print(f"✨ Nova melhoria significativa encontrada! Melhor fitness geral: {best_fitness_overall:.2f}")
        else:
            generations_without_improvement += 1
            print(f"Sem melhoria significativa. Gerações estagnadas: {generations_without_improvement}/{STAGNATION_LIMIT}")

        if generations_without_improvement >= STAGNATION_LIMIT:
            print(f"CRITÉRIO DE PARADA ATINGIDO: Ausência de melhoria por {STAGNATION_LIMIT} gerações.")
            break
        individuals = genetic.generate_new_population(sorted_population)
        print(f"Avaliando {POPULATION_SIZE} indivíduos da população {population}...")
        start_time_eval = time.time()
        individuals, population_info = evaluate_mutated_individuals(game_interface, individuals)
        populations_info[population] = population_info
        eval_time = time.time() - start_time_eval
        population += 1
        sorted_population = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        current_best_fitness = sorted_population[0].fitness
        fitness_history.append(current_best_fitness)
        print(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")

    print('salvando resultados...')
    doc_dir = Path(__file__).resolve().parent.parent / 'docs'
    plot_dir = Path(__file__).resolve().parent.parent / 'plots'
    for key, value in populations_info.items():
        dict_info = [asdict(r) for r in value]
        doc_dir.mkdir(exist_ok=True)
        with open(doc_dir / f'results_gen_{key}.json', 'w', encoding='utf-8') as f:
            json.dump(dict_info, f, ensure_ascii=False, indent=4)
    print('gerando gráficos')
    result_manager = ResultManager(doc_dir, plot_dir, 'results_gen_*.json')
    result_manager.mean_fitness()
    result_manager.distance_vs_kill()
    result_manager.secondary_mean_evolutuion()
    print("\n" + "="*50)
    print("Evolução finalizada.")

    game_interface.close()
    print("Instância do jogo finalizada. Processo concluído.")
