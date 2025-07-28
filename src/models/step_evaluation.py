from typing import List


class StepEvaluation:
    def __init__(self, action_performed: List[int], kills_before: int, health_before: int, items_before: int, damega_count_before: int):
        self._action_performed = action_performed
        self._kills_before = kills_before
        self._health_before = health_before
        self._items_before = items_before
        self._damega_count_before = damega_count_before
        self._kills_after = 0
        self._health_after = 0
        self._items_after = 0
        self._damega_count_after = 0

    @property
    def kills_after(self):
        raise AttributeError("'kills_after' getter is not defined")

    @kills_after.setter
    def kills_after(self, kills_after: int) -> None:
        self._kills_after = kills_after

    @property
    def health_after(self):
        raise AttributeError("'health_after' getter is not defined")

    @health_after.setter
    def health_after(self, health_after: int) -> None:
        self._health_after = health_after

    @property
    def items_after(self):
        raise AttributeError("'items_after' getter is not defined")

    @items_after.setter
    def items_after(self, items_after: int) -> None:
        self._items_after = items_after

    @property
    def damega_count_after(self):
        raise AttributeError("'damega_count_after' getter is not defined")

    @damega_count_after.setter
    def damega_count_after(self, damega_count_after: int) -> None:
        self._damega_count_after = damega_count_after

    def progress_statu(self, progress: int) -> None:
        self.__progress = progress

    def step_results(self) -> int:
        kill_npc = self._kills_after - self._kills_before > 0
        lost_health = self._health_after - self._health_before > 0
        get_items = self._items_after - self._items_before > 0
        miss_shot = self._action_performed[0] == 1 and self._damega_count_after == self._damega_count_before
        damage_count = self._damega_count_before - self._damega_count_after > 0
        return int(kill_npc) + int(get_items) - int(lost_health) - int(miss_shot) + int(damage_count) + self.__progress