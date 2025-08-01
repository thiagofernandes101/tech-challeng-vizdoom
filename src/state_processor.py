# stateProcessor.py (VERSÃO CORRIGIDA E COMPLETA)

import numpy as np
import vizdoom as vzd
import math

class StateProcessor:
    """
    Processa o estado bruto do jogo e o transforma em um vetor de estado
    numérico, agora com informações de ângulo, mira e a distância correta.
    """
    def __init__(self, goal_name: str, enemy_names: list[str]):
        self.goal_name = goal_name
        self.enemy_names = set(enemy_names)

    def _find_entities(self, state: vzd.GameState, names: set[str]) -> list:
        """Encontra todas as entidades com os nomes fornecidos."""
        entities = []
        if not state or not state.labels:
            return entities
        for label in state.labels:
            if label.object_name in names:
                entities.append(label)
        return entities

    def _get_angle_to_entity(self, player_pos: np.ndarray, player_angle: float, entity_pos: np.ndarray) -> float:
        """Calcula o ângulo relativo do jogador para uma entidade."""
        vec_to_entity = entity_pos - player_pos
        angle_to_entity_rad = np.arctan2(vec_to_entity[1], vec_to_entity[0])
        angle_to_entity_deg = math.degrees(angle_to_entity_rad)
        
        relative_angle = player_angle - angle_to_entity_deg
        while relative_angle > 180:
            relative_angle -= 360
        while relative_angle < -180:
            relative_angle += 360
            
        return relative_angle / 180.0

    def process(self, state: vzd.GameState, player_pos: np.ndarray, player_angle: float, player_health: float, ammo_count: int, is_shooting: bool) -> np.ndarray:
        """
        Cria o vetor de estado com 7 características para a rede neural.
        """
        normalized_health = player_health / 100.0
        normalized_ammo = ammo_count / 50.0
        shooting_flag = 1.0 if is_shooting else 0.0
        visible_enemies = self._find_entities(state, self.enemy_names)
        
        enemy_is_present_flag = 0.0
        crosshair_on_enemy_flag = 0.0
        relative_enemy_angle = 0.0
        relative_enemy_distance = 1.0 

        player_pos_3d = np.append(player_pos, 0)
        if visible_enemies:
            
            distances = [np.linalg.norm(player_pos_3d - np.array([e.object_position_x, e.object_position_y, e.object_position_z])) for e in visible_enemies]
            nearest_enemy_label = visible_enemies[np.argmin(distances)]
            
            enemy_is_present_flag = 1.0
            
            enemy_distance = min(distances)
            relative_enemy_distance = min(1.0, 100.0 / (enemy_distance + 1e-6))
            
            enemy_pos = np.array([nearest_enemy_label.object_position_x, nearest_enemy_label.object_position_y])
            relative_enemy_angle = self._get_angle_to_entity(player_pos, player_angle, enemy_pos)
            
            if abs(relative_enemy_angle) < 0.1:
                crosshair_on_enemy_flag = 1.0

        goal_label = next((label for label in state.labels if label.object_name == self.goal_name), None)
        relative_goal_angle = 0.0
        relative_goal_distance = 1.0

        if goal_label:
            goal_pos_3d = np.array([goal_label.object_position_x, goal_label.object_position_y, goal_label.object_position_z])
            goal_distance = np.linalg.norm(player_pos_3d - goal_pos_3d)
            relative_goal_distance = min(1.0, 100.0 / (goal_distance + 1e-6))

            goal_pos = np.array([goal_label.object_position_x, goal_label.object_position_y])
            relative_goal_angle = self._get_angle_to_entity(player_pos, player_angle, goal_pos)

        # --- VETOR DE ESTADO CORRIGIDO (AGORA COM 7 ENTRADAS) ---
        state_vector = np.array([
            normalized_health,
            enemy_is_present_flag,
            relative_enemy_distance,
            relative_enemy_angle,
            crosshair_on_enemy_flag,
            relative_goal_angle,
            relative_goal_distance
        ])
        
        return state_vector