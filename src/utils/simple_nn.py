import numpy as np
from typing import Optional

class SimpleNN():
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, rng: Optional[np.random.Generator] = None) -> None:
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__output_size = output_size
        self.__rng = rng or np.random.default_rng(seed=42)

        self.__w1 = self.__rng.standard_normal((hidden_size, input_size))
        self.__b1 = np.zeros((hidden_size, 1))
        self.__w2 = self.__rng.standard_normal((output_size, hidden_size))
        self.__b2 = np.zeros((output_size, 1))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = np.dot(self.__w1, x) + self.__b1
        a1 = self.tanh(z1) # Mudar aqui
        z2 = np.dot(self.__w2, a1) + self.__b2
        a2 = self.tanh(z2) # Mudar aqui
        return a2

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.__w1.flatten(),
            self.__b1.flatten(),
            self.__w2.flatten(),
            self.__b2.flatten()
        ])

    def set_weights(self, genome: np.ndarray) -> None:
        i, h, o = self.__input_size, self.__hidden_size, self.__output_size
        idx = 0

        self.__w1 = genome[idx:idx + h * i].reshape(h, i)
        idx += h * i

        self.__b1 = genome[idx:idx + h].reshape(h, 1)
        idx += h

        self.__w2 = genome[idx:idx + o * h].reshape(o, h)
        idx += o * h

        self.__b2 = genome[idx:idx + o].reshape(o, 1)
