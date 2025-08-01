import numpy as np
from vizdoom import GameState, GameVariable

class StateProcessor:
    """
    Processes the raw game state (GameState) and transforms it into a
    numerical state vector for the neural network.
    """
    def __init__(self, goal_name: str, enemy_names: list[str]):
        """
        Initializes the processor with the names of the objects of interest.

        Args:
            goal_name (str): The label name of the main objective (e.g., "GreenArmor").
            enemy_names (list[str]): A list of label names for enemies (e.g., ["Zombieman", "Imp"]).
        """
        self.goal_name = goal_name
        self.enemy_names = set(enemy_names)  # Using a set for faster lookups

    def _find_nearest_entity(self, state: GameState, player_pos: np.ndarray) -> dict | None:
        """Helper to find the closest visible enemy to the player."""
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

        # Return the enemy with the smallest distance
        return min(visible_enemies, key=lambda e: e['dist'])

    def _find_goal(self, state: GameState) -> np.ndarray | None:
        """Helper to find the position of the objective."""
        if not state or not state.labels:
            return None
        
        for label in state.labels:
            if label.object_name == self.goal_name:
                return np.array([label.object_position_x, label.object_position_y])
        return None

    def process(self, state: GameState, player_pos: np.ndarray, player_health: float) -> np.ndarray:
        """
        Main method that converts the game state into a numerical vector.

        Args:
            state (GameState): The raw state object from ViZDoom.
            player_pos (np.ndarray): The player's current (x, y) position.
            player_health (float): The player's current health.

        Returns:
            np.ndarray: The normalized state vector, ready for the network.
        """
        # Normalize player health (0 to 1)
        normalized_health = player_health / 100.0

        # Process the goal's position
        goal_pos = self._find_goal(state)
        if goal_pos is not None:
            # Normalize distance by a reasonable factor
            relative_goal_vector = (goal_pos - player_pos) / 1000.0
        else:
            relative_goal_vector = np.array([0.0, 0.0])

        # Process the nearest enemy's position
        nearest_enemy = self._find_nearest_entity(state, player_pos)
        if nearest_enemy is not None:
            # Normalize distance by a reasonable factor
            relative_enemy_vector = (nearest_enemy['pos'] - player_pos) / 1000.0
            enemy_is_present_flag = 1.0
        else:
            relative_enemy_vector = np.array([0.0, 0.0])
            enemy_is_present_flag = 0.0

        # Build the final state vector
        state_vector = np.concatenate([
            [normalized_health],
            relative_goal_vector,
            relative_enemy_vector,
            [enemy_is_present_flag]
        ])
        
        return state_vector