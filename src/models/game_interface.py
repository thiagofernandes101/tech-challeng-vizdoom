from enum import Enum
import vizdoom as vzd

from models.movement import Movement
from models.game_element import ElementFactory, FakeLabel, GameElement, Player
from models.individual_info import IndividualInfo
from utils.calc import Calc

class GameInfo(Enum):
    HEALTH = vzd.GameVariable.HEALTH
    ITEMS_COUNT = vzd.GameVariable.ITEMCOUNT
    DAMAGE_COUNT = vzd.GameVariable.DAMAGECOUNT
    KILL_COUNT = vzd.GameVariable.KILLCOUNT
    DAMAGE_TAKEN = vzd.GameVariable.DAMAGE_TAKEN
    WEAPON_AMMO = vzd.GameVariable.SELECTED_WEAPON_AMMO
    POSITION_X = vzd.GameVariable.POSITION_X

class GameInterface():

    def __init__(self, scenario_path: str = "deadly_corridor.cfg", show_screen: bool = False):
        self.__game = vzd.DoomGame()
        self.__configure_screen(scenario_path, show_screen)        
        self.__configure_buttons()
        self.__game.set_mode(vzd.Mode.PLAYER)
        self.__game.set_depth_buffer_enabled(True)
        self.__game.set_labels_buffer_enabled(True)
        self.__game.set_available_game_variables([
            vzd.POSITION_X,
            vzd.POSITION_Y,
            vzd.ANGLE,
            vzd.POSITION_Z
        ])
        self.__game.set_seed(42)
        self.__game.set_doom_skill(1)
        self.__game.init()
        self.__target_x = 1312.00
        self.__target_y = 0.0
        self.__wrong_shot = 0
        self.__checkpoints = [
            (300.0, 200.0),
            (600.0, 300.0),
            (900.0, 400.0),
            (1200.0, 600.0)
        ]

    def get_avaliable_buttons_amount(self) -> int:
        return self.__game.get_available_buttons_size()

    def start_episode(self)-> None:
        self.__distance = Calc.distance(0.0, 0.0, self.__target_x, self.__target_y)
        self.__wrong_shot = 0
        self.__game.new_episode()
        for _ in range(7):
            self.__game.advance_action()

    def episode_is_finished(self) -> bool:
        finished = self.__game.is_episode_finished()
        if finished:
            self.__individual_info = IndividualInfo(
                self.__distance - self.__current_distance, 
                self.get_state_info(GameInfo.HEALTH), 
                self.get_state_info(GameInfo.KILL_COUNT),
                self.get_fitness()
            )
        return finished
    
    def individual_info(self):
        return self.__individual_info

    def get_fitness(self)-> float:
        checkpoint_bonus = 0.0
        final_x_pos = self.get_state_info(GameInfo.POSITION_X)
        for checkpoint_x, bonus_value in self.__checkpoints:
            # Se a posição final do jogador ultrapassou o checkpoint...
            if final_x_pos >= checkpoint_x:
                checkpoint_bonus = bonus_value
            else:
                break

        base_fitness = (
            (6.0 * self.get_state_info(GameInfo.KILL_COUNT)) +
            (1.0 * self.get_state_info(GameInfo.HEALTH)) +
            (0.4 * self.get_state_info(GameInfo.WEAPON_AMMO)) +
            (0.5 * self.get_state_info(GameInfo.ITEMS_COUNT)) +
            (1.0 * self.get_state_info(GameInfo.DAMAGE_COUNT)) +
            (-0.5 * self.get_state_info(GameInfo.DAMAGE_TAKEN)) +
            (-0.5 * self.__wrong_shot) +
            (3.0 * (self.__distance - self.__current_distance))
        )

        return base_fitness + checkpoint_bonus

    def get_state_info(self, info: GameInfo)-> float:
        return self.__game.get_game_variable(info.value)

    __EPISODE_FINISHED = 'O episódio foi finalizado'

    def get_current_y(self)-> float:
        if (not self.__game.get_state()):
            raise RuntimeError(self.__EPISODE_FINISHED)
        return self.__game.get_state().game_variables[1]

    def get_current_x(self)-> float:
        if (not self.__game.get_state()):
            raise RuntimeError(self.__EPISODE_FINISHED)
        return self.__game.get_state().game_variables[0]
    
    def get_current_angle(self)-> float:
        if (not self.__game.get_state()):
            raise RuntimeError(self.__EPISODE_FINISHED)
        return self.__game.get_state().game_variables[2]
    
    def get_current_z(self)-> float:
        if (not self.__game.get_state()):
            raise RuntimeError(self.__EPISODE_FINISHED)
        return self.__game.get_state().game_variables[3]

    def get_visible_elements(self) -> tuple[list[GameElement], GameElement]:
        if (not self.__game.get_state()):
            raise RuntimeError('Elementos indisponíveis')
        labels = self.__game.get_state().labels
        elements: list[GameElement] = []
        player = None
        for label in labels:
            element = ElementFactory.create(label)
            if element is not None:
                if isinstance(element, Player):
                    player = element
                else:
                    elements.append(element)
        if player is None:
            label = FakeLabel(-1, 'DoomPlayer', self.get_current_x(), self.get_current_y(), self.get_current_z(), self.get_current_angle())
            player = Player(label)
        return sorted(elements, key=lambda e: Calc.get_distance_between_elements(player, e)), player

    def make_action(self, movement: Movement) -> None:
        commands = movement.to_list_command()
        if len(commands) > self.get_avaliable_buttons_amount():
            raise ValueError(f'actions list must have {len(self.get_avaliable_buttons_amount())} but you sent {len(commands)}')
        
        before_damage_count = self.get_state_info(GameInfo.DAMAGE_COUNT)

        self.__game.make_action(commands)
        if movement.attack and before_damage_count == self.get_state_info(GameInfo.DAMAGE_COUNT):
            self.__wrong_shot +=1

        if self.__game.get_state():
            self.__current_distance = Calc.distance(self.__game.get_state().game_variables[0], self.__game.get_state().game_variables[1], self.__target_x, self.__target_y)

    def close(self) -> None:
        self.__game.close()

    def __configure_screen(self, scenario_path: str, show_screen: bool) -> None:
        self.__game.load_config(scenario_path)
        self.__game.set_window_visible(show_screen)
        self.__game.set_screen_format(vzd.ScreenFormat.BGR24)
        self.__game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.__game.set_depth_buffer_enabled(True)
        self.__game.set_labels_buffer_enabled(True)

    def __configure_buttons(self) -> None:
        self.__game.set_available_buttons([
            vzd.Button.ATTACK,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.MOVE_UP,
            vzd.Button.MOVE_DOWN
        ])
