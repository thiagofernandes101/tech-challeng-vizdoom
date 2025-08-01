from typing import Optional
import numpy as np


class Individual:
    """
    Representa um único indivíduo na população de um algoritmo genético,
    contendo seu genoma e seu valor de fitness.
    """
    
    def __init__(self, genome: np.ndarray, fitness: Optional[float] = None):
        """
        Inicializa um novo indivíduo.

        Args:
            genome (np.ndarray): O genoma do indivíduo, representado como um array NumPy.
            fitness (Optional[float]): O valor de fitness do indivíduo.
                                       Pode ser None se ainda não foi avaliado.
        """
        self.genome = genome
        self.fitness = fitness

    def __repr__(self) -> str:
        """
        Retorna uma representação em string do objeto Individual para depuração.
        """
        return f"Individual(fitness={self.fitness:.4f} | genome_len={len(self.genome)})"