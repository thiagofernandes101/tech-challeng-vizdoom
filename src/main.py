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
from models.real_time_plot import RealTimePlot
from utils.mapper import Mapper
from utils.simple_nn import SimpleNN
from utils.genetic import Genetic

MOVEMENT_LIMIT = 4500 


SCENARIO_PATH = "deadly_corridor.cfg"
POPULATION_SIZE = 50 

W_KILLS = 150.0
W_HEALTH = 1.0
W_AMMO = 0.2
ITEM_COUNT = 5.0
DAMEGE_COUNT = 10.0
DAMAGE_TAKEN = -0.5
MISSING_SHOT = -0.8
GAME_PROGRESS = 0.4

STAGNATION_LIMIT = 4
IMPROVEMENT_THRESHOLD = 0.1

CONVERGENCE = 10

ELITISM_COUNT = 1
TOURNAMENT_SIZE = 3 
MUTATION_RATE = 0.5 

NEURAL_INPUTS = 40
HIDDEN_SIZE = 222

def initialize_game(show_screen: bool) -> GameInterface:
    """Cria e configura a instância do jogo ViZDoom."""
    print("Inicializando ViZDoom...")
    return GameInterface(SCENARIO_PATH, show_screen=show_screen)

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

    final_vector = episode_info_vector[:NEURAL_INPUTS]
    padding_needed = NEURAL_INPUTS - len(final_vector)
    
    if padding_needed > 0:
        final_vector += [0.0] * padding_needed

    return np.array(final_vector).reshape(-1, 1)

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
        individual.info = game_interface.individual_info()
        
    # Retorna a população (com fitness atualizado) e as métricas coletadas
    return population, population_metrics

if __name__ == "__main__":
    SHOW_GAME_SCREEN = True

    rng = np.random.default_rng(seed=42)
    moviments = all_valid_moviments()
    genetic = Genetic(ELITISM_COUNT, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, moviments, rng)
    simple_nn = SimpleNN(NEURAL_INPUTS, HIDDEN_SIZE, 9, rng)
    game_interface = initialize_game(show_screen=SHOW_GAME_SCREEN)

    plot = None
    if SHOW_GAME_SCREEN:
        print("Inicializando o gráfico de fitness em tempo real...")
        plot = RealTimePlot()

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
        if plot and not plot.handle_events():
            running = False
            continue

        sorted_population = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        current_best_fitness = sorted_population[0].fitness
        print(f"Melhor Fitness da Geração {population_count - 1}: {current_best_fitness:.2f}")

        if plot:
            plot.update_data(current_best_fitness)

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
        elite_individuals = sorted_population[:ELITISM_COUNT]
        elite_metrics = [game_interface.individual_info() for _ in elite_individuals]
        temporary_population = genetic.generate_new_population(sorted_population)
        children_to_evaluate = temporary_population[ELITISM_COUNT:]
        
        # 4. AVALIE a nova geração
        if children_to_evaluate:
            print(f"Avaliando {len(children_to_evaluate)} novos indivíduos da Geração {population_count}...")
            start_time_eval = time.time()
            
            # Renomeado 'metrics' para 'children_metrics' para maior clareza
            evaluated_children, children_metrics = evaluate_population(game_interface, children_to_evaluate, simple_nn, moviments)
            
            eval_time = time.time() - start_time_eval
            print(f"Avaliação concluída em {eval_time:.2f}s.")
        else:
            evaluated_children = []
            children_metrics = []

        # Extrai as métricas da elite, que já estão salvas em cada indivíduo da avaliação anterior.
        elite_metrics = [ind.info for ind in elite_individuals]

        # Combina as métricas da elite (da geração passada) com as dos filhos (da geração atual).
        all_metrics_for_this_generation = elite_metrics + children_metrics
        populations_metrics_history[population_count] = all_metrics_for_this_generation

        population_count += 1
    
    # Lógica para salvar os resultados (como no seu código original)
    print('Salvando resultados...')

    doc_dir = Path(__file__).resolve().parent.parent / 'docs'
    plot_dir = Path(__file__).resolve().parent.parent / 'plots'
    doc_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    for gen_num, metrics_list in populations_metrics_history.items():
        dict_info = [asdict(r) for r in metrics_list]
        doc_dir = Path(__file__).resolve().parent.parent / 'docs'
        doc_dir.mkdir(exist_ok=True)
        with open(doc_dir / f'results_gen_{gen_num}.json', 'w', encoding='utf-8') as f:
            json.dump(dict_info, f, ensure_ascii=False, indent=4)
    print('gerando gráficos')
    result_manager = ResultManager(doc_dir, plot_dir, 'results_gen_*.json')
    result_manager.mean_fitness()
    result_manager.distance_vs_kill()
    result_manager.secondary_mean_evolutuion()
    print("\n" + "="*50)
    print("Evolução finalizada.")

    if SHOW_GAME_SCREEN:
        input(f"Digite sair para fechar o jogo...")

    if plot:
        plot.close()

    game_interface.close()

    print("Instância do jogo finalizada. Processo concluído.")
