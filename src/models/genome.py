from models.movement import Movement
import numpy as np


class Genome:
    def __init__(self, movement: Movement):
        self.__movement = movement
        self.__neural_output = np.empty(0, dtype=np.float32)

    @property
    def movement(self) -> Movement:
        return self.__movement
    
    @property
    def neural_output(self) -> np.ndarray:
        return self.__neural_output
    
    @neural_output.setter
    def neural_output(self, neural_output: np.ndarray) -> None:
        self.__neural_output = neural_output

    def __str__(self) -> str:
        return f"action: {self.__movement} neural_output: {self.__neural_output}"
