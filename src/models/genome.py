class Genome:
    def __init__(self, action_index: int):
        self._action_index = action_index
        self._action_side_effect = 0

    @property
    def action_index(self) -> int:
        return self._action_index
    
    @property
    def action_side_effect(self) -> int:
        return self._action_side_effect
    
    @action_side_effect.setter
    def action_side_effect(self, action_side_effect: int) -> None:
        self._action_side_effect = action_side_effect

    def __str__(self) -> str:
        return f"action_index: {self._action_index} action_side_effect: {self._action_side_effect}"
