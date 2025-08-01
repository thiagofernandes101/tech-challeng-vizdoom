import random
from typing import List, Tuple
import numpy as np

from models.entities.Individual import Individual


class GeneticAlgorith:
    population_size: int = 300
    tournament_size: int = 5
    elistis_count: int = 5
    
    def create_initial_population(self, genome_length: int) -> List[Individual]:
        """
        Cria uma população inicial de indivíduos com genomas aleatórios.

        Args:
            genome_length (int): O comprimento do genoma para cada indivíduo.

        Returns:
            List[Individual]: Uma lista de objetos Individual recém-criados.
        """
        population: List[Individual] = []
        
        for _ in range(self.population_size):
            genome: np.ndarray = (np.random.rand(genome_length) * 2 - 1) * 0.5 
            individual: Individual = Individual(genome=genome, fitness=None)
            population.append(individual)
            
        return population
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Seleciona um indivíduo da população usando o método de seleção por torneio.

        Args:
            population (List[Individual]): A lista de indivíduos na população atual.

        Returns:
            Individual: O indivíduo com o maior fitness do torneio selecionado.
        """
        # 1. Seleciona aleatoriamente um subconjunto de indivíduos da população.
        #    O tamanho do subconjunto é definido por tournament_size.
        #    random.sample garante que não há repetições.
        tournament_contestants: List[Individual] = random.sample(population, self.tournament_size)

        # 2. Encontra o indivíduo com o maior fitness entre os participantes do torneio.
        #    A chave 'key=lambda x: x.fitness' instrui a função max() a comparar os objetos
        #    Individual com base em seus atributos 'fitness'.
        winner: Individual = max(tournament_contestants, key=lambda individual: individual.fitness)
        
        return winner
    
    def two_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        # (Implementação atualizada como discutido anteriormente)
        p1_genome: np.ndarray = parent1.genome
        p2_genome: np.ndarray = parent2.genome
        genome_len: int = len(p1_genome)

        if genome_len < 3:
            return Individual(p1_genome.copy(), fitness=None), Individual(p2_genome.copy(), fitness=None)

        cut_point1: int = random.randint(1, genome_len - 1)
        cut_point2: int = random.randint(1, genome_len - 1)
        start_point: int = min(cut_point1, cut_point2)
        end_point: int = max(cut_point1, cut_point2)

        if start_point == end_point:
            if end_point < genome_len - 1: end_point += 1
            else: start_point -= 1
            if start_point < 0: start_point = 0

        p1_list: List[float] = list(p1_genome)
        p2_list: List[float] = list(p2_genome)

        child1_genome_array: np.ndarray = np.array(
            p1_list[:start_point] + p2_list[start_point:end_point] + p1_list[end_point:]
        )
        child2_genome_array: np.ndarray = np.array(
            p2_list[:start_point] + p1_list[start_point:end_point] + p2_list[end_point:]
        )

        return Individual(child1_genome_array, fitness=None), Individual(child2_genome_array, fitness=None)
    
    def mutate(individual: Individual, mutation_rate: float) -> Individual:
        """
        Aplica mutação ao genoma de um indivíduo e retorna um novo indivíduo mutado.

        Args:
            individual (Individual): O indivíduo cujo genoma será mutado.
            mutation_rate (float): A probabilidade (entre 0.0 e 1.0) de cada gene sofrer mutação.

        Returns:
            Individual: Um novo objeto Individual com o genoma mutado e fitness None.
        """
        mutated_genome: np.ndarray = individual.genome.copy()
        genome_length: int = len(mutated_genome)

        for i in range(genome_length):
            random_chance: float = random.random()
            
            if random_chance < mutation_rate:
                mutation_value: float = random.gauss(0, 0.2)
                mutated_genome[i] += mutation_value

        mutated_individual: Individual = Individual(genome=mutated_genome, fitness=None)
        return mutated_individual
    
    def generate_new_population(
            self,
            old_population: List[Individual],
            mutation_rate: float
        ) -> List[Individual]:
        """
        Gera uma nova população de indivíduos usando elitismo, seleção,
        crossover e mutação.

        Args:
            old_population (List[Individual]): A população da geração anterior.
            mutation_rate (float): A taxa de mutação a ser aplicada aos genomas dos filhos.

        Returns:
            List[Individual]: A nova população gerada.
        """
        # 1. Classifica a população antiga por fitness em ordem decrescente.
        #    Assumimos que 'fitness' já foi calculado para todos na old_population.
        #    Lidamos com o caso de fitness ser None para que o sorted() não falhe.
        #    Aqui, None fitness são tratados como -inf para classificação (piores).
        sorted_population: List[Individual] = sorted(
            old_population,
            key=lambda individual: individual.fitness if individual.fitness is not None else -np.inf,
            reverse=True
        )

        # 2. Elitismo: Os melhores indivíduos da população antiga são transferidos diretamente 
        #    para a nova população.
        new_population: List[Individual] = []
        num_elites: int = min(self.elistis_count, len(sorted_population)) 
        
        # Adicionamos uma cópia dos genomas dos elites para a nova população.
        # Isso evita que a nova população contenha referências diretas aos objetos da população antiga,
        # caso você decida modificar os indivíduos da nova população mais tarde sem afetar os antigos.
        for i in range(num_elites):
            elite_individual: Individual = sorted_population[i]
            # Criamos um novo Individual para o elite para garantir que sejam objetos separados
            new_population.append(Individual(genome=elite_individual.genome.copy(), fitness=elite_individual.fitness))

        # 3. Geração da População por Crossover e Mutação.
        while len(new_population) < self.population_size:
            parent1: Individual = self.tournament_selection(sorted_population)
            parent2: Individual = self.tournament_selection(sorted_population)

            child1_raw_individual: Individual
            child2_raw_individual: Individual
            child1_raw_individual, child2_raw_individual = self.two_point_crossover(parent1, parent2)

            mutated_child1: Individual = self.mutate(child1_raw_individual, mutation_rate)
            new_population.append(mutated_child1)

            if len(new_population) < self.population_size:
                mutated_child2: Individual = self.mutate(child2_raw_individual, mutation_rate)
                new_population.append(mutated_child2)
                
        return new_population
    
