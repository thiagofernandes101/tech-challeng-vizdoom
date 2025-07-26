import math
from models.game_element import GameElement
from models.movement import Movement


class Calc:
    @staticmethod
    def get_distance_between_elements(element_1: GameElement, element_2: GameElement):
        pos_x = element_1.pos_x - element_2.pos_x
        pos_y = element_1.pos_y - element_2.pos_y
        pos_z = element_1.pos_z - element_2.pos_z
        return math.sqrt(pos_x * pos_x + pos_y * pos_y + pos_z * pos_z)

    @staticmethod
    def angle_to_target(pos_x: float, pos_y: float, target_x: float, target_y: float):
        dx = target_x - pos_x
        dy = target_y - pos_y
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg
    
    @staticmethod
    def angle_difference(self_angle: float, element_angle: float):
        diff = (self_angle - element_angle + 360) % 360
        if diff > 180:
            diff -= 360
        return diff
    
    @staticmethod
    def is_target_visible(player_angle: float, target_angle: float) -> bool:
        return abs(Calc.angle_difference(player_angle, target_angle)) <= (90 / 2)
    
    @staticmethod
    def simulate_movement_effect(pos_x: float, pos_y: float, angle: float, movement: Movement) -> tuple[float, float, float]:
        if movement.move_forward:
            pos_y += 5
        elif movement.move_backward:
            pos_y -= 5
        elif movement.move_left:
            angle = (angle - 15) % 360
        elif movement.move_right:
            angle = (angle + 15) % 360
        return pos_x, pos_y, angle
