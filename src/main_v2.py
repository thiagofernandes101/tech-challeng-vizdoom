import time
from models.game_element import GameElement
from models.game_interface import GameInterface
from models.individual import Individual
from models.step_evaluation import StepEvaluation

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

def initialize_game() -> GameInterface:
    """Cria e configura a instância do jogo ViZDoom."""
    print("Inicializando ViZDoom...")
    return GameInterface(SCENARIO_PATH, True)

def calculate_fitness(game_interface: GameInterface) -> list[Individual]:
    population: list[Individual] = []
    for _ in range(POPULATION_SIZE):
        game_interface.start_episode()

        current_visuble_elements: list[GameElement] = []
        while(not current_visuble_elements):
            try:
                current_visuble_elements, player = game_interface.get_visible_elements()
            except RuntimeError:
                pass

        wrong_shot = 0
        episode_progress = 0
        current_genome = 0

        individual = Individual()
        game_interface.subscribe(individual)
        while(not game_interface.episode_is_finished()):
            pos_x = game_interface.get_current_x()
            pos_y = game_interface.get_current_y()
            angle = game_interface.get_current_angle()
            individual.add_genome(pos_x, pos_y, angle, current_visuble_elements, player)
            genome = individual.genome(current_genome)
            damege_count_before_action = game_interface.get_damage_count()
            step_evaluation = StepEvaluation(genome.action(), 
                                    game_interface.get_kill_count(), 
                                    game_interface.get_current_healt(),
                                    game_interface.get_items_count(),
                                    damege_count_before_action
                                    )
            game_interface.make_action(genome.action())
            step_evaluation.kills_after = game_interface.get_kill_count()
            step_evaluation.health_after = game_interface.get_current_healt()
            step_evaluation.items_after = game_interface.get_items_count()
            step_evaluation.damega_count_after = game_interface.get_damage_count()
            individual.evaluate_genome(current_genome, step_evaluation.step_results())
            current_genome += 1
            try:
                current_visuble_elements, player = game_interface.get_visible_elements()
            except RuntimeError:
                pass
        game_interface.unsubscribe()
        population.append(individual)

    fitness_score = (W_KILLS * game_interface.get_kill_count()) + \
                (W_HEALTH * game_interface.get_current_healt()) + \
                (W_AMMO * game_interface.get_selected_weapon_ammo()) + \
                (ITEM_COUNT * game_interface.get_items_count()) +\
                (DAMEGE_COUNT * game_interface.get_damage_count()) +\
                (DAMAGE_TAKEN * game_interface.get_damage_taken()) +\
                (MISSING_SHOT * wrong_shot) +\
                (GAME_PROGRESS * episode_progress)
    individual.fitness = fitness_score
    
    return population
    

if __name__ == "__main__":

    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    game_interface = initialize_game()
    fitness_history: list[int] = []
    for generation in range(MAX_GENERATIONS):
        print(f"\n{'='*20} GERAÇÃO {generation} {'='*20}")
        start_time_eval = time.time()

        population = calculate_fitness(game_interface)

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
        current_population = generate_new_population(sorted_population, num_possible_actions, generations_without_improvement > 10)

    print("\n" + "="*50)
    print("Evolução finalizada.")
    
    final_best_individual = sorted(current_population, key=lambda x: x['fitness'], reverse=True)[0]
    print(f"Melhor fitness final alcançado: {final_best_individual['fitness']:.2f}")
    print(f"Executado por {len(fitness_history)} gerações.")

    game_interface.close()
    print("Instância do jogo finalizada. Processo concluído.")