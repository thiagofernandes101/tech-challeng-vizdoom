import numpy as np
from models.genome import Genome
from models.individual_info import IndividualInfo


class Individual():
    def __init__(self):
        self.__genome: np.ndarray = np.array([])
        self.__fitness = 0.0
        self.__info: IndividualInfo = None
        
    @property
    def genome(self) -> np.ndarray:
        return self.__genome
    
    @genome.setter
    def genome(self, genome: np.ndarray) -> None :
        self.__genome = genome

    @property
    def fitness(self)-> float:
        return self.__fitness

    @fitness.setter
    def fitness(self, fitness: float)-> None:
        self.__fitness = fitness

    @property
    def evaluated(self) -> bool:
        return self.__evaluated
    
    @property
    def info(self) -> list:
        return self.__info
    
    @info.setter
    def info(self, individualInfo: IndividualInfo):
        self.__info = individualInfo
    
    def inc_genome(self, genome: Genome) -> None:
        self.genomes.append(genome)

