import random
from typing import List
from models.genome import Genome
from models.individual import Individual
from models.movement import Movement


class Genetic():

    def __init__(self, elitism_count: int, populate_size: int, tournament_size: int, mutate_rate: int):
        self.__elitism_count = elitism_count
        self.__populate_size = populate_size
        self.__tournament_size = tournament_size
        self.__mutate_rate = mutate_rate

    def generate_new_population(self, old_population: List[Individual], movements: List[Movement], populate_stagnated: bool):
        new_population = old_population[:self.__elitism_count]

        if populate_stagnated:
            del old_population[-self.__elitism_count:]
        else:
            del old_population[:self.__elitism_count]
            del old_population[-self.__elitism_count:]

        del old_population[:self.__elitism_count]
        del old_population[-self.__elitism_count:]

        while len(new_population) < self.__populate_size:
            parent1 = self.__tournament_selection(old_population)
            parent2 = self.__tournament_selection(old_population)

            child1_genome, child2_genome = self.__one_point_crossover(parent1.genomes, parent2.genomes)

            child1_genome = mutate(child1_genome, movements)
            
            child2_genome = mutate(child2_genome, movements)

            new_population.append(Individual(child1_genome))
            if len(new_population) < self.__populate_size:
                new_population.append(Individual(child2_genome))
                
        return new_population
    
    def __tournament_selection(self, population: List[Individual]) -> Individual:
        tournament_competitors = random.sample(population, self.__tournament_size)
        
        winner = max(tournament_competitors, key=lambda x: x.fitness)
        return winner
    
    def __one_point_crossover(self, parent1_genome: List[Genome], parent2_genome: List[Genome])-> tuple[List[Genome], List[Genome]]:
        assert len(parent1_genome) == len(parent2_genome)
        
        negative_gene_1 = [(i, g) for i, g in enumerate(parent1_genome) if g.action_side_effect < 0]
        negative_gene_2 = [(i, g) for i, g in enumerate(parent1_genome) if g.action_side_effect < 0]

        for (i_1, gene_1), (i_2, gene_2) in zip(negative_gene_1, negative_gene_2):
            parent1_genome[i_1] = Genome(gene_2.action_index)
            parent2_genome[i_2] = Genome(gene_1.action_index)

    def mutate(self, genomes: List[Genome], movements: List[Movement]) -> List[Genome]:
        mutated_genome: List[Genome] = []
        for genome in genomes:
            if (genome.action_side_effect < 0):
                new_movement = random.choice(movements)
                while new_movement == genome.movement:
                    new_movement = random.choice(movements)
                mutated_genome.append(Genome(new_movement))
            elif (genome.action_side_effect == 0):
                if random.random() < self.__mutate_rate:
                    mutated_genome.append(Genome(random.choice(movements)))
                else:
                    mutated_genome.append(genome)
            else:
                mutated_genome.append(genome)
        return mutated_genome