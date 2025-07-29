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

        del old_population[-self.__elitism_count:]

        while len(new_population) < self.__populate_size:
            parent1 = self.__tournament_selection(old_population)
            parent2 = self.__tournament_selection(old_population)

            child_1_genome, child_2_genome = self.__one_point_crossover(parent1.genomes, parent2.genomes)

            child_1_genome = self.__mutate(child_1_genome)
            
            child_2_genome = self.__mutate(child_2_genome)

            child_1 = Individual()
            child_1.genomes = child_1_genome
            new_population.append(child_1)
            if len(new_population) < self.__populate_size:
                child_2 = Individual()
                child_2.genomes = child_2_genome
                new_population.append(child_2)
                
        return new_population
    
    def __tournament_selection(self, population: List[Individual]) -> Individual:
        tournament_competitors = random.sample(population, self.__tournament_size)
        
        winner = max(tournament_competitors, key=lambda x: x.fitness)
        return winner
    
    def __one_point_crossover(self, parent_1_genome: List[Genome], parent_2_genome: List[Genome])-> tuple[List[Genome], List[Genome]]:
        genoma_len = len(parent_1_genome)
        if len(parent_1_genome) < 3:
            return parent_1_genome.copy(), parent_2_genome.copy()
        
        p1 = random.randint(1, genoma_len - 1)
        p2 = random.randint(1, genoma_len - 1)
        start_point = min(p1, p2)
        end_point = max(p1, p2)
        if start_point == end_point:
            if end_point < genoma_len -1:
                end_point += 1
            else:
                start_point -= 1
        child_1 = parent_1_genome[:start_point] + parent_2_genome[start_point:end_point] + parent_1_genome[end_point:]
        child_2 = parent_2_genome[:start_point] + parent_1_genome[start_point:end_point] + parent_2_genome[end_point:]
        return child_1, child_2


    def __mutate(self, genomes: List[Genome]) -> List[Genome]:
        mutated_genome: list[Genome] = []
        for genome in genomes:
            mutate_now = False
            new_genome = genome
            if mutate_now or random.random() < self.__mutate_rate:
                try:
                    new_output = genome.neural_output + self.__rng.normal(0, 0.2, size=genome.neural_output.shape)
                    new_genome = Mapper.neural_output_to_moviment(np.clip(new_output, 0.0, 1.0), self.__valid_moves)
                except Exception:
                    mutate_now = True
                    mutated_genome.append(new_genome)
                    continue
            mutated_genome.append(new_genome)
        return mutated_genome