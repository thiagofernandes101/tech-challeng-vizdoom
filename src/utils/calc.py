import math

from models.game_element import GameElement

class Calc:
    @staticmethod
    def distance(source_x, source_y, target_x, target_y):
        return math.sqrt((target_x - source_x)**2 + (target_y - source_y)**2)
    
    @staticmethod
    def get_distance_between_elements(element_1: GameElement, element_2: GameElement):
        pos_x = element_1.pos_x - element_2.pos_x
        pos_y = element_1.pos_y - element_2.pos_y
        pos_z = element_1.pos_z - element_2.pos_z
        return math.sqrt(pos_x * pos_x + pos_y * pos_y + pos_z * pos_z)