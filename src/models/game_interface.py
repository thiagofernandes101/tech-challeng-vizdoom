import vizdoom as vzd

from models.game_element import ElementFactory, GameElement, Player
from models.event_emitter import EventEmitter
from models.observer import Observer
from utils.calc import Calc

class GameInterface(EventEmitter):

    def __init__(self, scenario_path: str = "deadly_corridor.cfg", show_screen: bool = False):
        self.__game = vzd.DoomGame()
        self.__configure_screen(scenario_path, show_screen)        
        self.__configure_buttons()
        self.__game.set_mode(vzd.Mode.PLAYER)
        self.__game.set_available_game_variables([
            vzd.POSITION_X,
            vzd.POSITION_Y,
            vzd.ANGLE,
        ])
        self.__game.set_doom_skill(1)
        self.__game.init()
        self.__enemy_die_amount = 0
        self.__item_collected_amount = 0

    def get_avaliable_buttons_amount(self) -> int:
        return self.__game.get_available_buttons_size()

    def start_episode(self)-> None:
        self.__game.new_episode()
        for _ in range(7):
            self.__game.advance_action()

    def episode_is_finished(self) -> bool:
        finished = self.__game.is_episode_finished()
        if finished:
            self.__enemy_die_amount = 0
            self.__item_collected_amount = 0
        return finished

    def get_kill_count(self)-> int:
        current_kills = self.__game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        if current_kills != self.__enemy_die_amount:
            self.notify()
            self.__enemy_die_amount += 1
        return current_kills

    def get_damage_count(self)-> int:
        return self.__game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
    
    def get_current_healt(self)-> int:
        return self.__game.get_game_variable(vzd.GameVariable.HEALTH)
    
    def get_items_count(self)-> int:
        current_items = self.__game.get_game_variable(vzd.GameVariable.ITEMCOUNT)
        if current_items != self.__item_collected_amount:
            self.notify()
            self.__enemy_die_amount += 1
        return current_items
    
    def get_damage_taken(self)-> int:
        return self.__game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
    
    def get_selected_weapon_ammo(self)-> int:
        return self.__game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)

    def make_action(self, actions: list[int]) -> None:
        if len(actions) > self.get_avaliable_buttons_amount():
            raise ValueError(f'actions list must have {len(self.get_avaliable_buttons_amount())} but you sent {len(actions)}')
        self.__game.make_action(actions)

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
        return sorted(elements, key=lambda e: Calc.get_distance_between_elements(player, e)), player

    def subscribe(self, observer: Observer) -> None:
        self.__observer = observer

    def unsubscribe(self) -> None:
        self.__observer = None

    def notify(self) -> None:
        if self.__observer is not None:
            self.__observer.update()

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