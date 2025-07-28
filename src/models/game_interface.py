import vizdoom as vzd

from models.movement import Movement
from models.step_evaluation import StepEvaluation
from utils.calc import Calc

class GameInterface():

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
        self.__target_x = 1312.00
        self.__target_y = 0.0

    def get_avaliable_buttons_amount(self) -> int:
        return self.__game.get_available_buttons_size()

    def start_episode(self)-> None:
        self.__distance = Calc.distance(0.0, 0.0, self.__target_x, self.__target_y)
        self.__episode_progress = 0.0
        self.__wrong_shot = 0
        self.__game.new_episode()
        for _ in range(7):
            self.__game.advance_action()

    def episode_is_finished(self) -> bool:
        return self.__game.is_episode_finished()
    
    def get_fitness(self)-> float:
        print(f'maior espaço percorrido {self.__distance - self.__current_distance}')
        if self.__game.get_game_variable(vzd.GameVariable.KILLCOUNT) > 4:
            print(f'kills: {self.__game.get_game_variable(vzd.GameVariable.KILLCOUNT)}')
        return (
            (0.5 * self.__game.get_game_variable(vzd.GameVariable.KILLCOUNT)) +
            (1.0 * self.__game.get_game_variable(vzd.GameVariable.HEALTH)) +
            (0.5 * self.__game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)) +
            (0.5 * self.__game.get_game_variable(vzd.GameVariable.ITEMCOUNT)) +
            (0.5 * self.__game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)) +
            (0.5 * self.__game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)) +
            (1.0 * self.__wrong_shot) -
            (2.0 * (self.__distance - self.__current_distance))
        )

    def make_action(self, movement: Movement) -> StepEvaluation:
        commands = movement.to_list_command()
        if len(commands) > self.get_avaliable_buttons_amount():
            raise ValueError(f'actions list must have {len(self.get_avaliable_buttons_amount())} but you sent {len(commands)}')
        
        before_kill_count = self.__game.get_game_variable(vzd.GameVariable.KILLCOUNT)

        step_evaluation = StepEvaluation(commands, 
                                            before_kill_count, 
                                            self.__game.get_game_variable(vzd.GameVariable.HEALTH),
                                            self.__game.get_game_variable(vzd.GameVariable.ITEMCOUNT),
                                            self.__game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
                                        )
        self.__game.make_action(commands)
        step_evaluation.kills_after = self.__game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        step_evaluation.health_after = self.__game.get_game_variable(vzd.GameVariable.HEALTH)
        step_evaluation.items_after = self.__game.get_game_variable(vzd.GameVariable.ITEMCOUNT)
        step_evaluation.damega_count_after = self.__game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        if movement.attack and before_kill_count == self.__game.get_game_variable(vzd.GameVariable.KILLCOUNT):
            self.__wrong_shot +=1

        if self.__game.get_state():
            self.__current_distance = Calc.distance(self.__game.get_state().game_variables[0], self.__game.get_state().game_variables[1], self.__target_x, self.__target_y)
            if self.__current_distance < self.__distance:
                step_evaluation.progress_statu(1)
            elif self.__current_distance > self.__distance:
                step_evaluation.progress_statu(-1)
            else:
                step_evaluation.progress_statu(0)   
        else:
            step_evaluation.progress_statu(0)   

        return step_evaluation

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