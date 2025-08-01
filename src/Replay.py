import os
import sys
import neat
import time
import vizdoom as vzd
import numpy as np

# Importa as classes que você já criou nos outros arquivos
from Game import Game
from StateProcessor import StateProcessor

def replay_from_checkpoint(config_path: str, checkpoint_path: str):
    """
    Carrega um checkpoint do NEAT, extrai o melhor genoma e o exibe jogando uma partida.
    """
    print(f"Carregando checkpoint: {checkpoint_path}")

    # Carrega a configuração do NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Restaura a população a partir do arquivo de checkpoint
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)

    # Tenta obter o melhor genoma pelo atalho
    best_genome = p.best_genome

    # Se o atalho falhar (retornar None), procuramos manualmente na população
    if best_genome is None:
        print("Atributo 'best_genome' não encontrado. Procurando manualmente na população...")
        
        # ==========================================================
        #           LÓGICA CORRIGIDA PARA LIDAR COM FITNESS 'None'
        # ==========================================================
        # 1. Filtra a população para incluir apenas genomas que foram avaliados.
        evaluated_genomes = [g for g in p.population.values() if g.fitness is not None]

        # 2. Verifica se a lista de genomas avaliados não está vazia.
        if not evaluated_genomes:
            print("Falha crítica: Nenhum genoma AVALIADO foi encontrado no checkpoint.")
            return
        
        # 3. Encontra o melhor genoma a partir da lista filtrada.
        best_genome = max(evaluated_genomes, key=lambda g: g.fitness)


    # Verificação final para garantir que encontramos um genoma
    if best_genome is None:
        print("Falha crítica: Nenhum genoma encontrado na população do checkpoint após a busca.")
        return

    print("\nMelhor genoma encontrado no checkpoint:")
    print(best_genome)

    # Cria a rede neural a partir do genoma
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    # Inicializa o ambiente ViZDoom
    game = Game.initialize_doom("deadly_corridor.cfg", render_window=True)
    game.set_mode(vzd.Mode.SPECTATOR)

    processor = StateProcessor(goal_name="GreenArmor", enemy_names=["Zombieman", "Imp"])
    
    print("\nIniciando apresentação...")
    
    for i in range(3):
        game.new_episode()
        total_reward = 0.0
        
        while not game.is_episode_finished():
            raw_state = game.get_state()
            
            player_pos = np.array([game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)])
            player_health = game.get_game_variable(vzd.GameVariable.HEALTH)

            state_vector = processor.process(raw_state, player_pos, player_health)
            output = net.activate(state_vector)

            turn_left = output[0] < -0.5
            turn_right = output[0] > 0.5
            move_forward = output[1] > 0.5
            attack = output[2] > 0.5
            action = [turn_left, turn_right, attack, move_forward, False, False, False]

            game.make_action(action)
            total_reward += game.get_last_reward()
            time.sleep(1.0 / 35.0)
        
        print(f"Episódio {i + 1} finalizado. Recompensa total: {total_reward:.2f}")
        time.sleep(0.0020)

    game.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python src/replay.py <nome_do_checkpoint>")
        sys.exit(1)
        
    checkpoint_file = sys.argv[1]
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.txt')
    
    project_root = os.path.dirname(local_dir)
    checkpoint_path = os.path.join(project_root, checkpoint_file)

    if not os.path.exists(config_path):
        print(f"Erro: Arquivo de configuração 'config-neat.txt' não encontrado em {local_dir}")
        sys.exit(1)
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Arquivo de checkpoint '{checkpoint_path}' não encontrado.")
        sys.exit(1)

    replay_from_checkpoint(config_path, checkpoint_path)