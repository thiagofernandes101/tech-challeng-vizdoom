import vizdoom as vzd
import numpy as np
import random
import time
import json
import os
from itertools import product


## VALIDAR OS PESOS, PENSAR EM ALGO PARA DESESTAGNAR, AVALIAR A DISTANCIA E A DIREÇAO PERCORRIDA E AUMENTAR A ALEATORIEDADE CONFORME AUMENTA A ESTAGNACAO

## REMOVER O ANDAR EM CIRCULOS (ultimas mudanças)

# ==============================================================================
# 1. PARÂMETROS DO ALGORITMO GENÉTICO E DO JOGO
# ==============================================================================

# --- Parâmetros da População ---
POPULATION_SIZE = 100  # Tamanho da população (100 indivíduos)
GENOME_LENGTH = 1000   # Número máximo de ações por episódio (duração da "vida" do agente)

# --- Pesos da Função de Fitness (Ajuste estes valores para guiar a evolução!) ---
# O objetivo é maximizar essa pontuação
W_KILLS = 50.0      # Peso para cada inimigo morto
W_HEALTH = 0.05       # Peso para cada ponto de vida restante
W_AMMO = 0.02         # Peso para cada ponto de munição restante
W_STEPS = -0.05       # Penalidade por passo dado (incentiva a terminar rápido)
W_FORWARD = 0.5
W_REWARD = 1.0 

# --- Parâmetros dos Operadores Genéticos ---
TOURNAMENT_SIZE = 3     # Número de indivíduos que competem em cada torneio de seleção
MUTATION_RATE = 0.2    # Probabilidade de um gene sofrer mutação (2%)
ELITISM_COUNT = 1       # Número de melhores indivíduos a serem passados diretamente para a próxima geração

# --- Parâmetros de Critério de Parada ---
MAX_GENERATIONS = 999999    # O número máximo de gerações que o algoritmo irá executar
STAGNATION_LIMIT = 10000    # N: O número de gerações sem melhoria antes de parar
IMPROVEMENT_THRESHOLD = 0.1 # A melhoria mínima no fitness para ser considerada "significativa"

# --- Configuração do ViZDoom ---
SCENARIO_PATH = "deadly_corridor.cfg"  # Nome do arquivo de configuração do cenário

# --- Configuração de Salvamento ---
SAVE_DIR = "saved_individuals"  # Diretório para salvar os melhores indivíduos
BEST_INDIVIDUAL_FILE = "best_individual.json"  # Arquivo do melhor indivíduo

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def initialize_game(headless=True):
    """
    Cria e configura a instância do jogo ViZDoom.
    
    Args:
        headless (bool): Se True, executa sem janela (para treinamento rápido)
                        Se False, mostra a janela (para demonstração)
    """
    print("Inicializando ViZDoom...")
    game = vzd.DoomGame()
    game.load_config(SCENARIO_PATH)
    
    # Configura visibilidade da janela baseado no parâmetro
    game.set_window_visible(not headless)
    
    # Se não for headless, adiciona delay para visualização
    if not headless:
        game.set_ticrate(35)  # 35 FPS para visualização mais suave
    
    game.set_mode(vzd.Mode.PLAYER)  # type: ignore
    game.set_available_buttons([
        vzd.Button.ATTACK,  # type: ignore
        vzd.Button.MOVE_LEFT,  # type: ignore
        vzd.Button.MOVE_RIGHT,  # type: ignore
        vzd.Button.MOVE_FORWARD,  # type: ignore
        vzd.Button.MOVE_BACKWARD,  # type: ignore
        vzd.Button.TURN_LEFT,  # type: ignore
        vzd.Button.TURN_RIGHT,  # type: ignore
        vzd.Button.MOVE_UP,  # type: ignore
        vzd.Button.MOVE_DOWN  # type: ignore
    ])
    game.init()
    
    # Crie o "cardápio" de ações possíveis.
    # Cada ação é uma lista de 0s e 1s com o mesmo tamanho do número de botões.
    
    num_buttons = game.get_available_buttons_size()
    
    # Gera todas as combinações possíveis de botões (0 ou 1 para cada)
    # product([0, 1], repeat=num_buttons) -> (0,0,0,0), (0,0,0,1), (0,0,1,0), ...
    actions = [list(p) for p in product([0, 1], repeat=num_buttons)]
    
    # Opcional: Remova a ação "não fazer nada" se não for desejada
    actions.remove([0] * num_buttons)
    # --- define pares conflitantes ---
    conflicts = [
        (1, 2),  # left vs right
        (3, 4),  # forward vs backward
        (5, 6),  # turn left vs turn right
        (7, 8)   # move up vs move down
    ]

    filtered = []
    for a in actions:
        # pula se apertou ambos de qualquer par conflitante
        if any(a[i] and a[j] for i, j in conflicts):
            continue
        # opcional: limitar a no máximo 2 botões de cada vez
        if sum(a) > 2:
            continue
        filtered.append(a)

    actions = filtered
    print(f"Ações filtradas: de {len(actions)} para {len(filtered)} possíveis movimentos.")
    print(f"Cardápio de ações definido com {len(actions)} movimentos possíveis.")
    
    return game, actions

def save_individual(individual, filename=None):
    """
    Salva um indivíduo em arquivo JSON.
    
    Args:
        individual (dict): Dicionário contendo 'genome' e 'fitness'
        filename (str): Nome do arquivo. Se None, usa nome padrão com timestamp
    """
    # Cria diretório se não existir
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"individual_fitness_{individual['fitness']:.2f}_{timestamp}.json"
    
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Prepara dados para salvar
    data_to_save = {
        'genome': individual['genome'],
        'fitness': individual['fitness'],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'population_size': POPULATION_SIZE,
            'genome_length': GENOME_LENGTH,
            'w_kills': W_KILLS,
            'w_health': W_HEALTH,
            'w_ammo': W_AMMO,
            'w_steps': W_STEPS
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Indivíduo salvo em: {filepath}")
    return filepath

def load_individual(filepath):
    """
    Carrega um indivíduo de arquivo JSON.
    
    Args:
        filepath (str): Caminho para o arquivo JSON
        
    Returns:
        dict: Dicionário contendo 'genome' e 'fitness'
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    individual = {
        'genome': data['genome'],
        'fitness': data['fitness']
    }
    
    print(f"Indivíduo carregado de: {filepath}")
    print(f"Fitness: {individual['fitness']:.2f}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    
    return individual

def demonstrate_individual(individual, actions, num_episodes=3):
    """
    Demonstra um indivíduo executando no jogo com tela visível.
    
    Args:
        individual (dict): Indivíduo a ser demonstrado
        actions (list): Lista de ações possíveis
        num_episodes (int): Número de episódios para demonstrar
    """
    print(f"\n{'='*50}")
    print(f"DEMONSTRAÇÃO DO MELHOR INDIVÍDUO")
    print(f"Fitness: {individual['fitness']:.2f}")
    print(f"Episódios: {num_episodes}")
    print(f"{'='*50}")
    
    # Inicializa jogo com tela visível
    game, _ = initialize_game(headless=False)
    
    total_kills = 0
    total_health = 0
    total_ammo = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        print(f"\n--- Episódio {episode + 1}/{num_episodes} ---")
        
        game.new_episode()
        episode_steps = 0
        
        # Executa cada ação do genoma
        for action_index in individual["genome"]:
            if game.is_episode_finished():
                break
            
            action_to_perform = actions[action_index]
            game.make_action(action_to_perform)
            episode_steps += 1
            
            # Pequena pausa para visualização
            time.sleep(0.05)
        
        # Coleta estatísticas do episódio
        kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)  # type: ignore
        health = game.get_game_variable(vzd.GameVariable.HEALTH)  # type: ignore
        ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)  # type: ignore
        
        total_kills += kills
        total_health += health
        total_ammo += ammo
        total_steps += episode_steps
        
        print(f"Episódio {episode + 1}: Kills={kills}, Health={health}, Ammo={ammo}, Steps={episode_steps}")
        
        # Pausa entre episódios
        if episode < num_episodes - 1:
            print("Pausa de 3 segundos antes do próximo episódio...")
            time.sleep(3)
    
    # Estatísticas finais
    avg_kills = total_kills / num_episodes
    avg_health = total_health / num_episodes
    avg_ammo = total_ammo / num_episodes
    avg_steps = total_steps / num_episodes
    
    print(f"\n{'='*50}")
    print(f"ESTATÍSTICAS FINAIS (média de {num_episodes} episódios):")
    print(f"Kills por episódio: {avg_kills:.1f}")
    print(f"Health por episódio: {avg_health:.1f}")
    print(f"Ammo por episódio: {avg_ammo:.1f}")
    print(f"Steps por episódio: {avg_steps:.1f}")
    print(f"{'='*50}")
    
    game.close()

def list_saved_individuals():
    """
    Lista todos os indivíduos salvos no diretório.
    
    Returns:
        list: Lista de caminhos para os arquivos salvos
    """
    if not os.path.exists(SAVE_DIR):
        print(f"Diretório {SAVE_DIR} não existe.")
        return []
    
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.json')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True)
    
    print(f"\nIndivíduos salvos em '{SAVE_DIR}':")
    for i, filename in enumerate(files):
        filepath = os.path.join(SAVE_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            fitness = data.get('fitness', 'N/A')
            timestamp = data.get('timestamp', 'N/A')
            print(f"{i+1}. {filename}")
            print(f"   Fitness: {fitness}")
            print(f"   Timestamp: {timestamp}")
        except:
            print(f"{i+1}. {filename} (erro ao ler)")
    
    return [os.path.join(SAVE_DIR, f) for f in files]

def create_initial_population(num_actions):
    """Cria a população inicial com genomas aleatórios."""
    population = []
    for _ in range(POPULATION_SIZE):
        # Genoma é uma lista de índices de ações aleatórias
        genome = [random.randint(0, num_actions - 1) for _ in range(GENOME_LENGTH)]
        individual = {
            "genome": genome,
            "fitness": 0.0  # Fitness inicial
        }
        population.append(individual)
    return population

# def calculate_fitness(game, individual, actions):
#     """
#     Executa um episódio do Doom para um indivíduo e calcula seu fitness.
#     Esta é a função mais importante e demorada do processo.
#     """
#     game.new_episode()

#     # Executa cada ação (gene) do genoma do indivíduo
#     for action_index in individual["genome"]:
#         if game.is_episode_finished():
#             break
        
#         # Ação é a combinação de botões correspondente ao índice
#         action_to_perform = actions[action_index]
#         game.make_action(action_to_perform)  # Corrigido: Removido os colchetes extras

#     # Coleta os resultados no final do episódio
#     if game.is_episode_finished():
#         # Se o episódio terminou (morreu ou completou), coletamos os dados finais
#         kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)  # type: ignore
#         health = game.get_game_variable(vzd.GameVariable.HEALTH)  # type: ignore
#         ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)  # type: ignore
#         steps_taken = game.get_episode_time()
#     else:
#         # Se o episódio não terminou (atingiu o limite de ações), o agente está "vivo"
#         # mas pode não ter feito nada útil.
#         kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)  # type: ignore
#         health = game.get_game_variable(vzd.GameVariable.HEALTH)  # type: ignore
#         ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)  # type: ignore
#         steps_taken = GENOME_LENGTH

#     # Aplica a fórmula de fitness com os pesos definidos
#     fitness_score = (W_KILLS * kills) + \
#                     (W_HEALTH * health) + \
#                     (W_AMMO * ammo) + \
#                     (W_STEPS * steps_taken)
                    
#     return fitness_score

def calculate_fitness(game, individual, actions):
    """
    Executa um episódio e retorna o fitness:
      - kills, health, ammo
      - forward_count (incentivo a andar pra frente)
      - stuck_counter (penaliza ficar parado)
      - total_reward (recompensa interna do ViZDoom)
    """
    # 1) Inicia novo episódio
    game.new_episode()

    # estados iniciais
    steps = 0
    total_reward = 0.0
    forward_count = 0
    stuck_counter = 0

    # pega posição inicial para medir stuck
    prev_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
    prev_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)

    # 2) Loop pelas ações do genoma
    for gene_idx in individual["genome"]:
        if game.is_episode_finished():
            break

        # 2.1) Executa ação e soma reward interno
        #         -- use gene_idx, não id
        reward = game.make_action(actions[gene_idx], 1)
        total_reward += reward
        steps += 1

        # 2.2) Contador de forward
        # supondo que o botão MOVE_FORWARD seja o índice 3
        if actions[gene_idx][3] == 1:
            forward_count += 1

        # 2.3) Verifica se ficou “parado”
        x = game.get_game_variable(vzd.GameVariable.POSITION_X)
        y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
        if abs(x - prev_x) + abs(y - prev_y) < 1e-3:
            stuck_counter += 1
        prev_x, prev_y = x, y

    # 3) Coleta variáveis finais
    kills  = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    health = game.get_game_variable(vzd.GameVariable.HEALTH)
    ammo   = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)

    # 4) Monta o fitness ponderado
    fitness = (
        W_KILLS    * kills
      + W_HEALTH   * health
      + W_AMMO      * ammo # W_STEPS pode ser zero se você não quiser punir todo passo
      + W_STEPS    * steps
      + W_FORWARD  * forward_count
      - 5.0       * stuck_counter
      + W_REWARD   * total_reward
    )

    return fitness



def tournament_selection(population):
    """
    Seleciona um indivíduo vencedor de um torneio.
    """
    # Seleciona 'TOURNAMENT_SIZE' indivíduos aleatórios da população
    tournament_competitors = random.sample(population, TOURNAMENT_SIZE)
    
    # O vencedor é aquele com o maior fitness
    winner = max(tournament_competitors, key=lambda x: x['fitness'])
    return winner

def one_point_crossover(parent1_genome, parent2_genome):
    """
    Realiza o crossover de um ponto entre os genomas de dois pais.
    """
    assert len(parent1_genome) == len(parent2_genome)
    genome_len = len(parent1_genome)
    
    # Escolhe um ponto de corte aleatório, exceto nas extremidades
    crossover_point = random.randint(1, genome_len - 1)
    
    # Cria os genomas dos filhos combinando as partes dos pais
    child1_genome = parent1_genome[:crossover_point] + parent2_genome[crossover_point:]
    child2_genome = parent2_genome[:crossover_point] + parent1_genome[crossover_point:]
    
    return child1_genome, child2_genome

def mutate(genome, num_actions):
    """
    Aplica mutação a um genoma com base na MUTATION_RATE.
    """
    mutated_genome = []
    for gene in genome:
        if random.random() < MUTATION_RATE:
            # Se a mutação ocorrer, substitui o gene por uma nova ação aleatória
            mutated_genome.append(random.randint(0, num_actions - 1))
        else:
            # Caso contrário, mantém o gene original
            mutated_genome.append(gene)
    return mutated_genome

def generate_new_population(old_population, num_actions):
    """
    Gera uma nova população completa usando elitismo, seleção, crossover e mutação.
    """
    # Primeiro, ordena a população antiga para encontrar o melhor indivíduo
    sorted_old_population = sorted(old_population, key=lambda x: x['fitness'], reverse=True)
    
    new_population = []
    
    # 1. Elitismo: Adiciona os melhores indivíduos diretamente à nova população
    # Regra 1 e 5: Manter o melhor indivíduo sem alterações.
    for i in range(ELITISM_COUNT):
        new_population.append(sorted_old_population[i])
        
    # 2. Geração dos Indivíduos Restantes (99, neste caso)
    # Regra 5: O restante da população é gerado pelo processo evolutivo.
    while len(new_population) < POPULATION_SIZE:
        # Seleção dos pais
        parent1 = tournament_selection(sorted_old_population)
        parent2 = tournament_selection(sorted_old_population)
        
        # Crossover para criar os filhos
        child1_genome, child2_genome = one_point_crossover(parent1['genome'], parent2['genome'])
        
        # Mutação dos filhos
        mutated_child1_genome = mutate(child1_genome, num_actions)
        mutated_child2_genome = mutate(child2_genome, num_actions)
        
        # Adiciona os novos filhos à população
        # O fitness é zerado, pois eles ainda não foram avaliados
        new_population.append({'genome': mutated_child1_genome, 'fitness': 0.0})
        if len(new_population) < POPULATION_SIZE:
            new_population.append({'genome': mutated_child2_genome, 'fitness': 0.0})
            
    return new_population

# ==============================================================================
# 3. SCRIPT PRINCIPAL DE AVALIAÇÃO
# ==============================================================================

if __name__ == "__main__":
    print("🎯 ALGORITMO GENÉTICO - TREINAMENTO")
    print("="*50)
    print("Este script executa o treinamento do algoritmo genético.")
    print("Para demonstrar os agentes treinados, use: python demo.py")
    print("="*50)
    
    # --- INICIALIZAÇÃO GERAL ---
    game_instance, possible_actions = initialize_game(headless=True)
    num_possible_actions = len(possible_actions)
    
    print(f"\nCriando população inicial com {POPULATION_SIZE} indivíduos...")
    current_population = create_initial_population(num_possible_actions)
    
    # Variáveis para controlar o critério de parada
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0
    fitness_history = [] # Para guardar o histórico de fitness de cada geração

    # --- LOOP PRINCIPAL DE EVOLUÇÃO ---
    for generation in range(MAX_GENERATIONS):
        print(f"\n{'='*20} GERAÇÃO {generation} {'='*20}")

        # 1. AVALIAÇÃO DA POPULAÇÃO ATUAL
        print(f"Avaliando {len(current_population)} indivíduos...")
        start_time_eval = time.time()
        # Nota: Só avaliamos os indivíduos que ainda não têm fitness (os gerados, não o elite)
        for i, individual in enumerate(current_population):
            if individual['fitness'] == 0.0: # Fitness 0.0 indica um novo indivíduo a ser avaliado
                # Descomente a linha abaixo para ver o progresso da avaliação
                # print(f"Avaliando indivíduo {i + 1}/{POPULATION_SIZE}...", end='\r')
                fitness = calculate_fitness(game_instance, individual, possible_actions)
                individual["fitness"] = fitness
        
        eval_time = time.time() - start_time_eval
        print(f"Avaliação concluída em {eval_time:.2f}s.")

        # 2. SELEÇÃO DE DADOS E VERIFICAÇÃO DE PARADA
        sorted_population = sorted(current_population, key=lambda x: x['fitness'], reverse=True)
        current_best_fitness = sorted_population[0]['fitness']
        fitness_history.append(current_best_fitness)
        
        print(f"Melhor Fitness da Geração: {current_best_fitness:.2f}")

        # Lógica do Critério de Parada
        if current_best_fitness > best_fitness_overall + IMPROVEMENT_THRESHOLD:
            best_fitness_overall = current_best_fitness
            generations_without_improvement = 0
            print(f"✨ Nova melhoria significativa encontrada! Melhor fitness geral: {best_fitness_overall:.2f}")
            # Salva automaticamente o melhor indivíduo quando há melhoria
            save_individual(sorted_population[0])
        else:
            generations_without_improvement += 1
            print(f"Sem melhoria significativa. Gerações estagnadas: {generations_without_improvement}/{STAGNATION_LIMIT}")

        if generations_without_improvement >= STAGNATION_LIMIT:
            print(f"CRITÉRIO DE PARADA ATINGIDO: Ausência de melhoria por {STAGNATION_LIMIT} gerações.")
            break
        
        # 3. GERAÇÃO DA PRÓXIMA POPULAÇÃO
        print("Gerando a próxima população...")
        current_population = generate_new_population(sorted_population, num_possible_actions)

    # --- FIM DA EVOLUÇÃO ---
    print("\n" + "="*50)
    print("Evolução finalizada.")
    
    final_best_individual = sorted(current_population, key=lambda x: x['fitness'], reverse=True)[0]
    print(f"Melhor fitness final alcançado: {final_best_individual['fitness']:.2f}")
    print(f"Executado por {len(fitness_history)} gerações.")
    
    # Salva o melhor indivíduo final
    save_individual(final_best_individual, "best_individual_final.json")

    game_instance.close()
    print("Instância do jogo finalizada. Processo concluído.")
    print("\n✅ Treinamento finalizado!")
    print("📁 Indivíduos salvos em: saved_individuals/")
    print("🎮 Para demonstrar os agentes, execute: python demo.py")