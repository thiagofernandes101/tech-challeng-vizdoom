# main.py
import numpy as np
from env_setup import create_env
from genetic import (
    adaptive_mut_rate, init_population, evaluate, tournament, crossover, mutate,
    POP_SIZE, GENOME_LEN
)
from realtime_plot import RealtimePlot

NUM_GENS = 1000

def main():
    env = create_env(render=False)
    pop = init_population(env)
    plot = RealtimePlot(max_gens=NUM_GENS)

    for gen in range(NUM_GENS):
      fits = [evaluate(env, ind) for ind in pop]
      best_score = max(fits)
      avg_score = sum(fits) / len(fits)

      print(f"Geração {gen+1} — melhor: {best_score:.2f}, média: {avg_score:.2f}")
      plot.update(best_score, avg_score)

      # best_idx = np.argmax(fits)
      # elite = pop[best_idx]
      # new_pop = [elite]

      elite_count = max(1, POP_SIZE // 10)  # 10%
      elite_indices = np.argsort(fits)[-elite_count:]
      elites = [pop[i] for i in elite_indices]
      new_pop = elites.copy()
      while len(new_pop) < POP_SIZE:
         p1 = tournament(pop, fits)
         p2 = tournament(pop, fits)
         c1, c2 = crossover(p1, p2)
         current_mut_rate = adaptive_mut_rate(gen, NUM_GENS)
         mutate(env, c1, current_mut_rate)
         mutate(env, c2, current_mut_rate)
         new_pop.extend([c1, c2])
      
      pop = new_pop[:POP_SIZE]

    env.close()
    plot.finalize()

if __name__ == "__main__":
    main()



# from random import random
# import gymnasium
# import numpy as np
# from vizdoom import gymnasium_wrapper

# env = gymnasium.make("VizdoomCorridor-v1", render_mode='human')

# individual = [env.action_space.sample() for _ in range(200)]

# population = [
#     [env.action_space.sample() for _ in range(200)]
#     for _ in range(100)
# ]

# def evaluate(individual):
#     obs, info = env.reset()
#     total_reward = 0
#     for action in individual:
#         obs, reward, term, trunc, info = env.step(action)
#         total_reward += reward
#         if term or trunc:
#             break
#     return total_reward

# def tournament_selection(pop, fitnesses, k=2):
#     selected = random.sample(list(zip(pop, fitnesses)), k)
#     return max(selected, key=lambda x: x[1])[0]  # retorna indivíduo com maior fitness

# def crossover(parent1, parent2):
#     point = random.randint(1, 199)
#     child1 = parent1[:point] + parent2[point:]
#     child2 = parent2[:point] + parent1[point:]
#     return child1, child2

# def mutate(individual, pm=0.01):
#     for i in range(len(individual)):
#         if random.random() < pm:
#             individual[i] = env.action_space.sample()

# pop_size = 100
# gen_len = 200
# num_gens = 100  # escolha conforme necessidade

# # inicializa população
# pop = [[env.action_space.sample() for _ in range(gen_len)] for _ in range(pop_size)]

# for gen in range(num_gens):
#     fitnesses = [evaluate(ind) for ind in pop]
#     new_pop = []
#     while len(new_pop) < pop_size:
#         p1 = tournament_selection(pop, fitnesses)
#         p2 = tournament_selection(pop, fitnesses)
#         c1, c2 = crossover(p1, p2)
#         mutate(c1); mutate(c2)
#         new_pop.extend([c1, c2])
#     pop = new_pop[:pop_size]

# import gymnasium
# from vizdoom import gymnasium_wrapper

# # Criação do ambiente
# env = gymnasium.make("VizdoomCorridor-v1", render_mode='human')
# n_actions = env.action_space.shape[0]  # ex: 3 para [MOVE_LEFT, MOVE_RIGHT, ATTACK]
# n_individuals = 100
# n_steps = 200  # número de ações que cada indivíduo executa

# def policy(observation):
#    radom_actions = env.action_space.sample()
#    print("Action space:", env.action_space)
#    return env.action_space.sample()  # Random action for demonstration

# for _ in range(1000):
#    action = policy(observation)  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()

# env.close()

# import os
# import vizdoom as vzd
# from random import choice
# from time import sleep

# if __name__ == "__main__":
#    # Create DoomGame instance. It will run the game and communicate with you.
#    game = vzd.DoomGame()

#    # Now it's time for configuration!
#    # load_config could be used to load configuration instead of doing it here with code.
#    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
#    # game.load_config("../../scenarios/basic.cfg")

#    # Sets path to additional resources wad file which is basically your scenario wad.
#    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
#    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "deadly_corridor.wad"))

#    # Sets map to start (scenario .wad files can contain many maps).
#    game.set_doom_map("map01")

#    # Sets resolution. Default is 320X240
#    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

#    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
#    game.set_screen_format(vzd.ScreenFormat.RGB24)

#    # Enables depth buffer (turned off by default).
#    game.set_depth_buffer_enabled(True)

#    # Enables labeling of in-game objects labeling (turned off by default).
#    game.set_labels_buffer_enabled(True)

#    # Enables buffer with a top-down map of the current episode/level (turned off by default).
#    game.set_automap_buffer_enabled(True)

#    # Enables information about all objects present in the current episode/level (turned off by default).
#    game.set_objects_info_enabled(True)

#    # Enables information about all sectors (map layout/geometry, turned off by default).
#    game.set_sectors_info_enabled(True)

#    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
#    game.set_render_hud(False)
#    game.set_render_minimal_hud(False)  # If hud is enabled
#    game.set_render_crosshair(False)
#    game.set_render_weapon(True)
#    game.set_render_decals(False)  # Bullet holes and blood on the walls
#    game.set_render_particles(False)
#    game.set_render_effects_sprites(False)  # Like smoke and blood
#    game.set_render_messages(False)  # In-game text messages
#    game.set_render_corpses(False)
#    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

#    # Adds buttons that will be allowed to use.
#    # This can be done by adding buttons one by one:
#    # game.clear_available_buttons()
#    # game.add_available_button(vzd.Button.MOVE_LEFT)
#    # game.add_available_button(vzd.Button.MOVE_RIGHT)
#    # game.add_available_button(vzd.Button.ATTACK)
#    # Or by setting them all at once:
#    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.MOVE_UP, vzd.Button.MOVE_DOWN, vzd.Button.ATTACK])
#    # Buttons that will be used can be also checked by:
#    print("Available buttons:", [b.name for b in game.get_available_buttons()])

#    # Adds game variables that will be included in state.
#    # Similarly to buttons, they can be added one by one:
#    # game.clear_available_game_variables()
#    # game.add_available_game_variable(vzd.GameVariable.AMMO2)
#    # Or:
#    game.set_available_game_variables([vzd.GameVariable.AMMO2])
#    print(
#    "Available game variables:",
#    [v.name for v in game.get_available_game_variables()],
#    )

#    # Causes episodes to finish after 200 tics (actions)
#    game.set_episode_timeout(200)

#    # Makes episodes start after 10 tics (~after raising the weapon)
#    game.set_episode_start_time(10)

#    # Makes the window appear (turned on by default)
#    game.set_window_visible(True)

#    # Turns on the sound. (turned off by default)
#    # game.set_sound_enabled(True)
#    # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented,
#    # the sound is only useful for humans watching the game.

#    # Turns on the audio buffer. (turned off by default)
#    # If this is switched on, the audio will stop playing on device, even with game.set_sound_enabled(True)
#    # Setting game.set_sound_enabled(True) is not required for audio buffer to work.
#    # game.set_audio_buffer_enabled(True)
#    # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented.

#    # Sets the living reward (for each move) to -1
#    game.set_living_reward(-1)


#    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
#    game.set_mode(vzd.Mode.PLAYER)

#    # Enables engine output to console, in case of a problem this might provide additional information.
#    # game.set_console_enabled(True)

#    # Initialize the game. Further configuration won't take any effect from now on.
#    game.init()

#    # Define some actions. Each list entry corresponds to declared buttons:
#    # MOVE_LEFT, MOVE_RIGHT, ATTACK
#    # game.get_available_buttons_size() can be used to check the number of available buttons.
#    # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
#    actions = [
#       [True, False, False, False, False],   # Apenas MOVE_LEFT
#       [False, True, False, False, False],   # Apenas MOVE_RIGHT
#       [False, False, True, False, False],   # Apenas MOVE_UP
#       [False, False, False, True, False],   # Apenas MOVE_DOWN
#       [False, False, False, False, True],   # Apenas ATTACK
#       [True, False, False, False, True],    # MOVE_LEFT + ATTACK
#       [False, True, False, False, True],    # MOVE_RIGHT + ATTACK
#       [False, False, True, False, True],    # MOVE_UP + ATTACK
#       [False, False, False, True, True],    # MOVE_DOWN + ATTACK
#       [True, False, True, False, False],    # MOVE_LEFT + MOVE_UP
#       [False, True, True, False, False],    # MOVE_RIGHT + MOVE_UP
#       [True, False, False, True, False],    # MOVE_LEFT + MOVE_DOWN
#       [False, True, False, True, False],    # MOVE_RIGHT + MOVE_DOWN
#       [True, False, True, False, True],     # MOVE_LEFT + MOVE_UP + ATTACK
#       [False, True, True, False, True],     # MOVE_RIGHT + MOVE_UP + ATTACK
#       [True, True, False, False, False],    # MOVE_LEFT + MOVE_RIGHT (também incoerente, mas possível)
#       [True, True, True, False, True],      # MOVE_LEFT + MOVE_RIGHT + MOVE_UP + ATTACK
#       [False, False, False, False, True],   # Apenas ATTACK (repetido para reforçar)
#       [False, False, False, False, False],  # Nenhuma ação
#    ]

#    # Run this many episodes
#    episodes = 10

#    # Sets time that will pause the engine after each action (in seconds)
#    # Without this everything would go too fast for you to keep track of what's happening.
#    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

#    for i in range(episodes):
#       print(f"Episode #{i + 1}")

#       # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
#       game.new_episode()

#       while not game.is_episode_finished():
#          # Gets the state
#          state = game.get_state()

#          # Which consists of:
#          n = state.number
#          vars = state.game_variables

#          # Different buffers (screens, depth, labels, automap, audio)
#          # Expect of screen buffer some may be None if not first enabled.
#          screen_buf = state.screen_buffer
#          depth_buf = state.depth_buffer
#          labels_buf = state.labels_buffer
#          automap_buf = state.automap_buffer
#          audio_buf = state.audio_buffer

#          # List of labeled objects visible in the frame, may be None if not first enabled.
#          labels = state.labels

#          # List of all objects (enemies, pickups, etc.) present in the current episode, may be None if not first enabled
#          objects = state.objects

#          # List of all sectors (map geometry), may be None if not first enabled.
#          sectors = state.sectors

#          # Games variables can be also accessed via
#          # (including the ones that were not added as available to a game state):
#          # game.get_game_variable(GameVariable.AMMO2)

#          # Makes an action (here random one) and returns a reward.
#          r = game.make_action(choice(actions))

#          # Makes a "prolonged" action and skip frames:
#          # skiprate = 4
#          # r = game.make_action(choice(actions), skiprate)

#          # The same could be achieved with:
#          # game.set_action(choice(actions))
#          # game.advance_action(skiprate)
#          # r = game.get_last_reward()

#          # Prints state's game variables and reward.
#          print(f"State #{n}")
#          print("Game variables:", vars)
#          print("Reward:", r)
#          print("=====================")

#          if sleep_time > 0:
#             sleep(sleep_time)

#       # Check how the episode went.
#       print("Episode finished.")
#       print("Total reward:", game.get_total_reward())
#       print("************************")

#    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
#    game.close()