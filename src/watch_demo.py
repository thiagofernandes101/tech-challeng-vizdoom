import sys
import vizdoom as vzd
import time

if len(sys.argv) != 2:
    print("Uso: python watch_demo.py <caminho_para_o_demo.lmp>")
    sys.exit(1)

demo_path = sys.argv[1]

game = vzd.DoomGame()
game.load_config("deadly_corridor.cfg") # Use o mesmo config do treino
game.set_window_visible(True)
game.set_mode(vzd.Mode.SPECTATOR)
game.init()

print(f"Reproduzindo demonstração: {demo_path}")
game.replay_episode(demo_path)

while not game.is_episode_finished():
    game.advance_action()
    time.sleep(1.0 / 35.0)

game.close()