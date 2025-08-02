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

STAGNATION_LIMIT = 1
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

def create_initial_population(simple_nn: SimpleNN, rng: np.random.Generator) -> list[Individual]:
    population: list[Individual] = []
    for _ in range(POPULATION_SIZE):
        individual = Individual()
        # CORREÇÃO: Use .genome, que agora está padronizado
        individual.genome = rng.standard_normal(simple_nn.get_weights().shape)
        population.append(individual)
    return population

def evaluate_population(game_interface: GameInterface, population: list[Individual], simple_nn: SimpleNN, valids_moves: list[Movement]) -> tuple[list[Individual], list[IndividualInfo]]:
    """
    Avalia cada indivíduo da população, atualiza seu fitness e coleta as métricas.
    """
    population_metrics = []
    for individual in population:
        # Configura a rede com o genoma (pesos) do indivíduo
        simple_nn.set_weights(individual.genome)
        
        game_interface.start_episode()
        while not game_interface.episode_is_finished():
            elements, player = game_interface.get_visible_elements()
            input_vector = generate_info_vector(game_interface, player, elements)
            output = simple_nn.forward(input_vector)
            movement = Mapper.neural_output_to_moviment(output, valids_moves).movement
            game_interface.make_action(movement)
        
        # Atualiza o fitness do indivíduo (a lista de população é modificada por referência)
        individual.fitness = game_interface.get_fitness()
        
        # Coleta as métricas detalhadas para este indivíduo
        population_metrics.append(game_interface.individual_info())
        
    # Retorna a população (com fitness atualizado) e as métricas coletadas
    return population, population_metrics

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    moviments = all_valid_moviments()
    genetic = Genetic(ELITISM_COUNT, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, moviments, rng)
    simple_nn = SimpleNN(NEURAL_INPUTS, HIDDEN_SIZE, 9, rng)
    game_interface = initialize_game()

    # Dicionário para armazenar o histórico de métricas de todas as gerações
    populations_metrics_history: dict[int, list[IndividualInfo]] = {}
    
    # 1. CRIE a população inicial
    print("Criando a população inicial...")
    individuals = create_initial_population(simple_nn, rng)
    
    # 2. AVALIE a população inicial (Geração 0)
    print("Avaliando a população inicial (Geração 0)...")
    start_time_eval = time.time()
    individuals, metrics = evaluate_population(game_interface, individuals, simple_nn, moviments)
    populations_metrics_history[0] = metrics
    eval_time = time.time() - start_time_eval
    print(f"Avaliação da Geração 0 concluída em {eval_time:.2f}s.")
    
    population_count = 1
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    
    # Loop de evolução
    while True:
        sorted_population = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        current_best_fitness = sorted_population[0].fitness
        print(f"Melhor Fitness da Geração {population_count - 1}: {current_best_fitness:.2f}")

        # Lógica de melhoria e critério de parada
        if current_best_fitness > best_fitness_overall + IMPROVEMENT_THRESHOLD:
            best_fitness_overall = current_best_fitness
            generations_without_improvement = 0
            print(f"✨ Nova melhoria! Melhor fitness geral: {best_fitness_overall:.2f}")
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= STAGNATION_LIMIT:
            print(f"CRITÉRIO DE PARADA ATINGIDO: Ausência de melhoria por {STAGNATION_LIMIT} gerações.")
            break

        # 3. GERE a nova população a partir da anterior
        individuals = genetic.generate_new_population(sorted_population)
        
        # 4. AVALIE a nova geração
        print(f"Avaliando {len(individuals)} indivíduos da Geração {population_count}...")
        start_time_eval = time.time()
        individuals, metrics = evaluate_population(game_interface, individuals, simple_nn, moviments)
        populations_metrics_history[population_count] = metrics
        eval_time = time.time() - start_time_eval
        print(f"Avaliação concluída em {eval_time:.2f}s.")
        
        population_count += 1
    
    # Lógica para salvar os resultados (como no seu código original)
    print('Salvando resultados...')
    for gen_num, metrics_list in populations_metrics_history.items():
        dict_info = [asdict(r) for r in metrics_list]
        # O caminho do diretório 'docs' um nível acima do 'src'
        doc_dir = Path(__file__).resolve().parent.parent / 'docs'
        doc_dir.mkdir(exist_ok=True)
        with open(doc_dir / f'results_gen_{gen_num}.json', 'w', encoding='utf-8') as f:
            json.dump(dict_info, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("Evolução finalizada.")
    game_interface.close()
    print("Instância do jogo finalizada. Processo concluído.")
