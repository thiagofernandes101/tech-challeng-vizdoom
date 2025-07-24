from typing import List

from models.genome import Genome

class Individual:
    def __init__(self, genomes: List[Genome]):
        self._genomes = genomes
        self._fitness = 0.0
        self._wainting_evaluated = True

    @property
    def genomes(self) -> List[Genome]:
        return self._genomes
    
    @genomes.setter
    def genomes(self, genomes: List[Genome]) -> None:
        self._genomes = genomes
    
    @property
    def fitness(self) -> List[int]:
        return self._fitness
    
    @fitness.setter
    def fitness(self, fitness: List[fitness]) -> None:
        self._wainting_evaluated = False
        self._fitness = fitness

    @property
    def wainting_evaluated(self) -> bool:
        return self._wainting_evaluated
    
    def evaluate_steps(self, evaluations: List[int]) -> None:
        for genome, evaluation in zip(self._genomes, evaluations):
            genome.action_side_effect = evaluation