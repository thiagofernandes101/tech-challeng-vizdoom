import random
from typing import List, Optional
from models.genome import Genome
from models.individual import Individual
from models.movement import Movement
import numpy as np
from utils.mapper import Mapper


class Genetic():

    def __init__(self, elitism_count: int, populate_size: int, tournament_size: int, mutate_rate: int, valid_moves: List[Movement], rng: Optional[np.random.Generator] = None):
        self.__elitism_count = elitism_count
        self.__populate_size = populate_size
        self.__tournament_size = tournament_size
        self.__mutate_rate = mutate_rate
        self.__valid_moves = valid_moves
        self.__rng = rng or np.random.default_rng(seed=42)

    def generate_new_population(self, old_population: List[Individual]):
        new_population = old_population[:self.__elitism_count]

        while len(new_population) < self.__populate_size:
            parent1 = self.__tournament_selection(old_population)
            parent2 = self.__tournament_selection(old_population)

            # Crossover e mutação nos genomas (pesos)
            child_1_genome, child_2_genome = self.__blended_crossover(parent1.genome, parent2.genome)
            child_1_genome = self.__mutate(child_1_genome)
            child_2_genome = self.__mutate(child_2_genome)

            child_1 = Individual()
            child_1.genome = child_1_genome
            new_population.append(child_1)

            if len(new_population) < self.__populate_size:
                child_2 = Individual()
                child_2.genome = child_2_genome
                new_population.append(child_2)

        return new_population
    
    def __tournament_selection(self, population: List[Individual]) -> Individual:
        tournament_competitors = random.sample(population, self.__tournament_size)
        
        winner = max(tournament_competitors, key=lambda x: x.fitness)
        return winner
    
    def __blended_crossover(self, parent_1_genome: np.ndarray, parent_2_genome: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        # Gera um fator de mistura aleatório para cada gene
        blend = self.__rng.uniform(low=-alpha, high=1+alpha, size=parent_1_genome.shape)

        child1_genome = blend * parent_1_genome + (1 - blend) * parent_2_genome
        child2_genome = (1 - blend) * parent_1_genome + blend * parent_2_genome

        return child1_genome, child2_genome


    def __mutate(self, genome: np.ndarray) -> np.ndarray:
        mutation = self.__rng.normal(0, 0.1, genome.shape)
        mask = self.__rng.random(genome.shape) < self.__mutate_rate
        mutated_genome = genome + mutation * mask
        return mutated_genome