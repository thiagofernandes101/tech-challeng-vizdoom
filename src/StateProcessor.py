import numpy as np
import vizdoom as vzd

class StateProcessor:
    """
    Processes the raw game state (GameState) and transforms it into a
    numerical state vector for the neural network.
    """
    def __init__(self, goal_name: str, enemy_names: list[str]):
        self.goal_name = goal_name
        self.enemy_names = set(enemy_names)

    def _find_nearest_entity(self, state: vzd.GameState, player_pos: np.ndarray) -> dict | None:
        visible_enemies = []
        if not state or not state.labels:
            return None
        for label in state.labels:
            if label.object_name in self.enemy_names:
                enemy_pos = np.array([label.object_position_x, label.object_position_y])
                distance = np.linalg.norm(player_pos - enemy_pos)
                visible_enemies.append({'pos': enemy_pos, 'dist': distance})
        if not visible_enemies:
            return None
        return min(visible_enemies, key=lambda e: e['dist'])

    def _find_goal(self, state: vzd.GameState) -> np.ndarray | None:
        if not state or not state.labels: return None
        for label in state.labels:
            if label.object_name == self.goal_name:
                return np.array([label.object_position_x, label.object_position_y])
        return None

    def process(self, state: vzd.GameState, player_pos: np.ndarray, player_health: float) -> np.ndarray:
        normalized_health = player_health / 100.0
        goal_pos = self._find_goal(state)
        relative_goal_vector = (goal_pos - player_pos) / 1000.0 if goal_pos is not None else np.array([0.0, 0.0])
        nearest_enemy = self._find_nearest_entity(state, player_pos)
        if nearest_enemy is not None:
            relative_enemy_vector = (nearest_enemy['pos'] - player_pos) / 1000.0
            enemy_is_present_flag = 1.0
        else:
            relative_enemy_vector = np.array([0.0, 0.0])
            enemy_is_present_flag = 0.0
        state_vector = np.concatenate([
            [normalized_health], relative_goal_vector,
            relative_enemy_vector, [enemy_is_present_flag]
        ])
        return state_vector