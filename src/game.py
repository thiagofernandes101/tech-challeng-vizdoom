from typing import Dict, List, Set
import numpy as np
import vizdoom as vzd

class Game:
    def __init__(self):
        self.state = None

    @staticmethod
    def initialize_doom(scenario, render_window=False) -> vzd.DoomGame:
        # Initialize the game environment and return the game instance
        game = vzd.DoomGame()
        game.load_config(scenario)
        game.set_window_visible(render_window)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_depth_buffer_enabled(True)
        game.set_labels_buffer_enabled(True)
        game.set_doom_skill(4)

        game.set_death_penalty(100)
        game.set_living_reward(-1)
        game.set_kill_reward(50)
        game.set_hit_taken_penalty(10)
        game.set_damage_made_reward(5)

        game.init()
        return game
    
    @staticmethod
    def get_action_space(game_instance: vzd.DoomGame) -> List[List[int]]:
        """
        Generates a list of possible action combinations for the given DoomGame instance.
        Each action is represented as a list of 0s and 1s corresponding to available buttons.

        Args:
            game_instance (vzd.DoomGame): The active DoomGame instance.

        Returns:
            List[List[int]]: A list where each inner list represents a possible action 
            (e.g., [1, 0, 0, 1] for move forward + turn right).
        """
        core_action_templates: List[Set[str]] = [
            # Basic Movements
            {'MOVE_FORWARD'},
            {'MOVE_BACKWARD'},
            {'TURN_RIGHT'},
            {'TURN_LEFT'},
            {'MOVE_RIGHT'},  # Strafe Right
            {'MOVE_LEFT'},   # Strafe Left
            # Combat Actions
            {'ATTACK'},
            {'ATTACK', 'MOVE_FORWARD'},
            {'ATTACK', 'MOVE_BACKWARD'},
            {'ATTACK', 'MOVE_RIGHT'},
            {'ATTACK', 'MOVE_LEFT'},
        ]
        available_buttons: List[vzd.Button] = game_instance.get_available_buttons()
        button_name_to_index: Dict[str, int] = {
            button.name: i for i, button in enumerate(available_buttons)
        }
        valid_actions: List[List[int]] = []

        for template in core_action_templates:
            if template.issubset(button_name_to_index.keys()):
                action_vector: List[int] = [0] * len(available_buttons)
                for button_name in template:
                    index = button_name_to_index[button_name]
                    action_vector[index] = 1
                valid_actions.append(action_vector)
        
        return valid_actions