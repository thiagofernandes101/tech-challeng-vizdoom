import random
import numpy as np
from models.genome import Genome
from models.movement import Movement

class Mapper():

    @staticmethod
    def neural_output_to_moviment(neural_output: np.ndarray, valid_moves: list[Movement]) -> Genome:
        output_vec = neural_output.flatten()

        min_dist = float('inf')
        best_move = None
        for move in valid_moves:
            move_vec = np.array(move.to_list_command(), dtype=np.float64)
            dist = np.linalg.norm(output_vec - move_vec)
            if dist < min_dist:
                min_dist = dist
                best_move = move

        if best_move is None:
            best_move = random.choice(valid_moves)
    
        return Genome(best_move)