from models.genome import Genome


class Individual():
    def __init__(self):
        self.__genomes: list[Genome] = []
        self.__fitness = 0.0
        
    @property
    def genomes(self) -> list[Genome]:
        return self.__genomes
    
    @genomes.setter
    def genomes(self, genomes: list[Genome]) -> None :
        self.__genomes = genomes

    @property
    def fitness(self)-> float:
        return self.__fitness

    @fitness.setter
    def fitness(self, fitness: float)-> None:
        self.__fitness = fitness

    @property
    def evaluated(self) -> bool:
        return self.__evaluated
    
    def inc_genome(self, genome: Genome) -> None:
        self.genomes.append(genome)

