from __future__ import annotations

class Movement():
    def __init__(self, 
                attack: bool = False, 
                move_left: bool = False, 
                move_right: bool = False, 
                move_forward: bool = False, 
                move_backward: bool = False, 
                turn_left: bool = False, 
                turn_right: bool = False,
                move_up: bool = False,
                move_down: bool = False):
        self.__attack = int(attack)
        self.__move_left = int(move_left)
        self.__move_right = int(move_right)
        self.__move_forward = int(move_forward)
        self.__move_backward = int(move_backward)
        self.__turn_left = int(turn_left)
        self.__turn_right = int(turn_right)
        self.__move_up = int(move_up)
        self.__move_down = int(move_down)

    @staticmethod
    def from_list(command: list[int]) -> Movement:
        if len(command) != 9:
            raise ValueError("Lista de comandos deve conter exatamente 9 elementos")
        return Movement(
            attack=bool(command[0]),
            move_left=bool(command[1]),
            move_right=bool(command[2]),
            move_forward=bool(command[3]),
            move_backward=bool(command[4]),
            turn_left=bool(command[5]),
            turn_right=bool(command[6]),
            move_up=bool(command[7]),
            move_down=bool(command[8]),
        )

    @property
    def attack(self) -> bool:
        return bool(self.__attack)
    
    @attack.setter
    def attack(self, attack: bool) -> None:
        self.__attack = int(attack)

    @property
    def move_left(self) -> bool:
        return bool(self.__move_left)
    
    @move_left.setter
    def move_left(self, move_left: bool) -> None:
        self.__move_left = int(move_left)

    @property
    def move_right(self) -> bool:
        return bool(self.__move_right)
    
    @move_right.setter
    def move_right(self, move_right: bool) -> None:
        self.__move_right = int(move_right)

    @property
    def move_forward(self) -> bool:
        return bool(self.__move_forward)
    
    @move_forward.setter
    def move_forward(self, move_forward: bool) -> None:
        self.__move_forward = int(move_forward)

    @property
    def move_backward(self) -> bool:
        return bool(self.__move_backward)
    
    @move_backward.setter
    def move_backward(self, move_backward: bool) -> None:
        self.__move_backward = int(move_backward)

    @property
    def turn_left(self) -> bool:
        return bool(self.__turn_left)
    
    @turn_left.setter
    def turn_left(self, turn_left: bool) -> None:
        self.__turn_left = int(turn_left)

    @property
    def turn_right(self) -> bool:
        return bool(self.__turn_right)
    
    @turn_right.setter
    def turn_right(self, turn_right: bool) -> None:
        self.__turn_right = int(turn_right)


    @property
    def move_up(self) -> bool:
        return bool(self.__move_up)
    
    @move_up.setter
    def move_up(self, move_up: bool) -> None:
        self.__move_up = int(move_up)


    @property
    def move_down(self) -> bool:
        return bool(self.__move_down)
    
    @move_down.setter
    def move_down(self, move_down: bool) -> None:
        self.__move_down = int(move_down)

    def make_some_move(self) -> bool:
        return self.move_backward or self.move_forward or self.move_down or self.move_up or self.move_left or self.move_right
    
    def turn_some_side(self) -> bool:
        return self.turn_left or self.turn_right
    
    def no_action(self) -> bool:
        return not self.make_some_move() and not self.turn_some_side() and not self.attack

    def to_list_command(self) -> list[int]:
        return [self.__attack, 
                self.__move_left, 
                self.__move_right, 
                self.__move_forward, 
                self.__move_backward, 
                self.__turn_left, 
                self.__turn_right, 
                self.__move_up, 
                self.__move_down]
    
    def __eq__(self, other):
        if not isinstance(other, Movement):
            return NotImplemented
        return self.to_list_command() == other.to_list_command()

    def __hash__(self):
        return hash(tuple(self.to_list_command()))