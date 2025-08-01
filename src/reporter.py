import os
import pickle
import neat
import numpy as np
import vizdoom as vzd
from game import Game
from state_processor import StateProcessor

class SaveBestGenomeReporter(neat.reporting.BaseReporter):
    """
    Um reporter customizado que:
    1. Salva o melhor genoma (.pkl) quando um novo recorde de fitness é atingido.
    2. Grava uma demonstração (.lmp) da performance recordista.
    """
    def __init__(self, scenario_path: str, processor_instance: StateProcessor, path_prefix='src/genomes/gen'):
        self.genome_path_prefix = path_prefix
        self.demo_path_prefix = path_prefix.replace('genomes', 'demos')
        
        # Armazene o caminho do cenário e a instância do processador
        self.scenario_path = scenario_path
        self.processor = processor_instance
        
        self.best_fitness = -float('inf')
        self.generation = 0
        
        os.makedirs(os.path.dirname(self.genome_path_prefix), exist_ok=True)
        os.makedirs(os.path.dirname(self.demo_path_prefix), exist_ok=True)

    def _record_demo(self, best_genome, config):
        """
        Roda um episódio adicional com o melhor genoma para gravar a demonstração.
        CRIA E DESTROI UMA INSTÂNCIA LOCAL DO JOGO PARA NÃO INTERFERIR NO TREINO.
        """
        demo_filename = f"{self.demo_path_prefix}-{self.generation}.lmp"
        print(f"Gravando demonstração da performance em: {demo_filename}")

        # Crie uma instância do jogo LOCALMENTE para a gravação
        recorder_game = Game.initialize_doom(self.scenario_path, render_window=False)
        
        try:
            net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            
            recorder_game.new_episode(demo_filename)
            is_shooting = False 
            while not recorder_game.is_episode_finished():
                raw_state = recorder_game.get_state()
                player_pos = np.array([recorder_game.get_game_variable(vzd.GameVariable.POSITION_X), recorder_game.get_game_variable(vzd.GameVariable.POSITION_Y)])
                player_health = recorder_game.get_game_variable(vzd.GameVariable.HEALTH)
                player_angle = recorder_game.get_game_variable(vzd.GameVariable.ANGLE)
                player_ammo = recorder_game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
                
                state_vector = self.processor.process(raw_state, player_pos, player_angle, player_health, player_ammo, is_shooting)
                output = net.activate(state_vector)

                turn_left = output[0] < -0.3
                turn_right = output[0] > 0.3
                move_forward = output[1] > 0.3
                attack = output[2] > 0.7
                action = [turn_left, turn_right, attack, move_forward, False, False, False]
                is_shooting = attack
                
                recorder_game.make_action(action)
        
        finally:
            # Garanta que a instância local do jogo seja fechada, mesmo se ocorrer um erro
            recorder_game.close()
            print("Gravação da demonstração concluída.")


    def end_generation(self, config, population, species_set):
        """
        Chamado no final de cada geração.
        """
        evaluated_genomes = [g for g in population.values() if g.fitness is not None]
        if not evaluated_genomes:
            self.generation += 1
            return

        current_best_genome = max(evaluated_genomes, key=lambda g: g.fitness)

        if current_best_genome.fitness > self.best_fitness:
            self.best_fitness = current_best_genome.fitness
            print(f"\n--- NOVO RECORDE DE FITNESS! ---")
            print(f"Geração: {self.generation}, Fitness: {self.best_fitness:.2f}")
            
            genome_filename = f"{self.genome_path_prefix}-{self.generation}.pkl"
            print(f"Salvando o melhor genoma em: {genome_filename}")
            with open(genome_filename, 'wb') as f:
                pickle.dump(current_best_genome, f)
            
            self._record_demo(current_best_genome, config)

        self.generation += 1