import neat
import pickle
import os
import sys

# VERSÃO CORRIGIDA
# Este script carrega um checkpoint do NEAT, encontra o melhor genoma
# inspecionando manualmente o fitness de cada indivíduo salvo,
# e o salva em um arquivo .pkl que pode ser usado pelo replay.py.

def save_winner_from_checkpoint(checkpoint_file: str, config_path: str):
    """
    Carrega um checkpoint, encontra o melhor genoma e o salva como 'winner.pkl'.
    """
    if not os.path.exists(checkpoint_file):
        print(f"Erro: Arquivo de checkpoint '{checkpoint_file}' não encontrado.")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"Erro: Arquivo de configuração '{config_path}' não encontrado.")
        sys.exit(1)

    # Para restaurar um checkpoint, também precisamos do arquivo de configuração
    # que foi usado durante o treinamento.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    print(f"Carregando checkpoint de: {checkpoint_file}")
    population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

    # --- LÓGICA CORRIGIDA ---
    # Em vez de usar population.run(), vamos encontrar o melhor genoma manualmente.
    # A população salva contém um dicionário de todos os genomas daquela geração.
    
    all_genomes = list(population.population.values())

    if not all_genomes:
        print("O checkpoint não contém genomas.")
        sys.exit(1)
    
    # Usamos a função max() para encontrar o genoma com o maior valor de fitness.
    # A verificação 'g.fitness is not None' garante que genomas não avaliados sejam ignorados.
    winner = max(all_genomes, key=lambda g: g.fitness if g.fitness is not None else -float('inf'))

    # --- FIM DA LÓGICA CORRIGIDA ---
    
    if winner is None or winner.fitness is None:
        print("Não foi possível encontrar um genoma vencedor com fitness válido no checkpoint.")
        sys.exit(1)

    output_path = "winner.pkl"
    print(f"\nMelhor genoma encontrado (ID: {winner.key}, Fitness: {winner.fitness})")
    print(f"Salvando genoma vencedor em: {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(winner, f)

    print("\nConcluído! Agora você pode usar 'winner.pkl' com seu script replay.py.")
    print(f"Execute: python src/replay.py {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python src/load_and_save_winner.py <caminho_para_o_checkpoint>")
        print("Exemplo: python src/load_and_save_winner.py neat-checkpoint-0")
        sys.exit(1)
    
    checkpoint_to_load = sys.argv[1]
    
    # O script precisa saber onde está o config-neat.txt
    local_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(local_dir, 'config-neat.txt')
    
    save_winner_from_checkpoint(checkpoint_to_load, config_file_path)