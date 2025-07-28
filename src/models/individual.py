from models.genome import Genome


class Individual():
    def __init__(self, genomes: list[Genome]):
        self.__genomes = genomes
        self.__fitness = 0.0
        self.__evaluated = False

    @property
    def genomes(self) -> list[Genome]:
        return self.__genomes
    
    @property
    def fitness(self)-> float:
        return self.__fitness

    @fitness.setter
    def fitness(self, fitness: float)-> None:
        self.__evaluated = True
        self.__fitness = fitness

    @property
    def evaluated(self) -> bool:
        return self.__evaluated

