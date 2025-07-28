from models.movement import Movement


class Genome:
    def __init__(self, movement: Movement):
        self._movement = movement
        self._movement_side_effect = 0

    @property
    def movement(self) -> Movement:
        return self._movement
    
    @property
    def movement_side_effect(self) -> int:
        return self._movement_side_effect
    
    @movement_side_effect.setter
    def movement_side_effect(self, movement_side_effect: int) -> None:
        self._movement_side_effect = movement_side_effect

    def __str__(self) -> str:
        return f"action: {self._movement} movement_side_effect: {self._movement_side_effect}"
