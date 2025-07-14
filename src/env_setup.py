# env_setup.py
import gymnasium
from vizdoom import gymnasium_wrapper

def create_env(render=False):
    mode = 'human' if render else 'none'  # render_mode='none' para treino
    env = gymnasium.make("VizdoomCorridor-v1", render_mode=mode)
    return env