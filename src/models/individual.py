import copy
from enum import Enum
import random
from typing import List

from models.genome import Genome
from models.game_element import Enemy, Colectable, GameElement
from models.movement import Movement
from models.observer import Observer
from utils.calc import Calc

class Individual(Observer):

    def __init__(self):
        self.__genomes: list[Genome] = []
        self.__fitness = 0.0
        self.__wainting_evaluated = True
        self.__targets: list[GameElement] = []
        self.__fitness = 0
        self.__options = {Movement(move_backward=True): False,
                    Movement(move_forward=True): False,
                    Movement(move_forward=True, move_left=True): False,
                    Movement(move_backward=True, move_left=True): False,
                    Movement(move_left=True): False,
                    Movement(move_right=True, move_forward=True): False,
                    Movement(move_right=True, move_backward=True): False,
                    Movement(move_right=True): False}
        self.__objects_ids_insteracted: list[int] = []

    @property
    def fitness(self) -> List[int]:
        return self.__fitness
    
    @fitness.setter
    def fitness(self, fitness: List[fitness]) -> None:
        self.__wainting_evaluated = False
        self.__fitness = fitness

    @property
    def wainting_evaluated(self) -> bool:
        return self.__wainting_evaluated

    @property
    def fitness(self) -> float:
        return self.__fitness
    
    @fitness.setter
    def fitness(self, fitness: float) -> None :
        self.__fitness = fitness

    def genome(self, index: int) -> Genome:
        return self.__genomes[index]
    
    def update(self) -> None:
        element = self.__targets.pop(0)
        self.__objects_ids_insteracted.append(element.id)

    def evaluate_genome(self, index: int, action_side_effect: int):
        self.__genomes[index].action_side_effect = action_side_effect

    def add_genome(self, pos_x: float, pos_y: float, angle: float, elements: list[GameElement], player: GameElement) -> None:
        self.__generate_genome(pos_x, pos_y, angle, elements, player)

    def __generate_genome(self, pos_x: float, pos_y: float, angle: float, elements: list[GameElement], player: GameElement) -> None:
        DANGER_RADIUS = 180.0
        elements = list(filter(lambda e: e.id not in self.__objects_ids_insteracted, elements))
        if not self.__targets:
            self.__targets = elements
        else:
            new_elements = [e for e in elements if e not in self.__targets]
            if new_elements:
                self.__targets += new_elements
                self.__targets = sorted(self.__targets, key=lambda e: Calc.get_distance_between_elements(player, e))

        nearby_enemies = [e for e in self.__targets if isinstance(e, Enemy) and Calc.get_distance_between_elements(player, e) <= DANGER_RADIUS]


        if nearby_enemies:
            enemy = sorted(nearby_enemies, key=lambda e: Calc.get_distance_between_elements(player, e))[0]
            action = self.__figth_against_enemy(pos_x, pos_y, angle, enemy)
            self.__genomes.append(Genome(action))
        else:
            if isinstance(self.__targets[0], Colectable):
                self.__go_to_target(player, angle, self.__targets[0])

    def __figth_against_enemy(self, pos_x: float, pos_y: float, angle: float, enemy: GameElement) -> List[int]:
        movement = self.__random_movement_with_visibility()
        sim_pos_x, sim_pos_y, sim_angle = Calc.simulate_movement_effect(pos_x, pos_y, angle, movement)
        rotate_needed = self.__rotate_towards(sim_pos_x, sim_pos_y, sim_angle, enemy)
        if rotate_needed > 5:
            movement.turn_right = True
        elif rotate_needed < -5:
            movement.turn_left = True
        if self.__can_shoot(sim_pos_x, sim_pos_y, sim_angle, enemy):
            movement.attack = True
        return movement.to_list_command

    def __rotate_towards(self, pos_x: float, pos_y: float, current_angle: float, enemy: GameElement) -> float:
        target_angle = Calc.angle_to_target(pos_x, pos_y, enemy.pos_x, enemy.pos_y)
        diff = Calc.angle_difference(current_angle, target_angle)
        return diff

    def __can_shoot(self, pos_x: float, pos_y: float, self_angle: float, enemy: GameElement) -> bool:
        enemy_angle = Calc.angle_to_target(pos_x, pos_y, enemy.pos_x, enemy.pos_y)
        diff = Calc.angle_difference(self_angle, enemy_angle)
        return abs(diff) <= 5

    def __random_movement_with_visibility(self) -> Movement:
        unused_movements = [m for m, used in self.__options.items() if not used]
        if not unused_movements:
            for key in self.__options.keys():
                self.__options[key] = False
            unused_movements = [m for m, used in self.__options.items() if not used]

        random.shuffle(unused_movements)
        self.__options[unused_movements[0]] = True
        return copy.deepcopy(unused_movements[0])

    def __go_to_target(self, player: GameElement, target_angle: float, collectable: GameElement) -> List[int]:
        movement = Movement()
        angle_diff = Calc.angle_difference(player.angle, target_angle)

        if abs(angle_diff) > 5:
            if angle_diff > 0:
                movement.turn_right = True
            else:
                movement.turn_left = True
        else:
            movement.move_forward = True

        self.__genomes.append(Genome(movement.to_list_command))
        return movement.to_list_command
