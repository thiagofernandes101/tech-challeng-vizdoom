#!/usr/bin/env python3
"""
Script para demonstrar indivíduos salvos do algoritmo genético.
Permite visualizar como o melhor agente se comporta no jogo.
"""

import vizdoom as vzd
import numpy as np
import json
import os
import time
import sys
from itertools import product

# Importa funções do main.py
sys.path.append(os.path.dirname(__file__))
try:
    from main import initialize_game, load_individual, list_saved_individuals
except ImportError:
    print("❌ Erro: Não foi possível importar funções do main.py")
    print("Certifique-se de que o arquivo main.py está no mesmo diretório.")
    sys.exit(1)

def demonstrate_individual(individual, actions, num_episodes=3, delay=0.05):
    """
    Demonstra um indivíduo executando no jogo com tela visível.
    
    Args:
        individual (dict): Indivíduo a ser demonstrado
        actions (list): Lista de ações possíveis
        num_episodes (int): Número de episódios para demonstrar
        delay (float): Delay entre ações para visualização
    """
    print(f"\n{'='*60}")
    print(f"🎮 DEMONSTRAÇÃO DO AGENTE GENÉTICO")
    print(f"Fitness: {individual['fitness']:.2f}")
    print(f"Episódios: {num_episodes}")
    print(f"Delay entre ações: {delay}s")
    print(f"{'='*60}")
    
    # Inicializa jogo com tela visível
    print("Inicializando jogo com tela visível...")
    game, _ = initialize_game(headless=False)
    
    total_kills = 0
    total_health = 0
    total_ammo = 0
    total_steps = 0
    episodes_completed = 0
    
    for episode in range(num_episodes):
        print(f"\n--- Episódio {episode + 1}/{num_episodes} ---")
        print("Iniciando episódio...")
        
        game.new_episode()
        episode_steps = 0
        
        # Executa cada ação do genoma
        for action_index in individual["genome"]:
            if game.is_episode_finished():
                break
            
            action_to_perform = actions[action_index]
            game.make_action(action_to_perform)
            episode_steps += 1
            
            # Pausa para visualização
            time.sleep(delay)
        
        # Coleta estatísticas do episódio
        kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)  # type: ignore
        health = game.get_game_variable(vzd.GameVariable.HEALTH)  # type: ignore
        ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)  # type: ignore
        
        total_kills += kills
        total_health += health
        total_ammo += ammo
        total_steps += episode_steps
        episodes_completed += 1
        
        print(f"Episódio {episode + 1} finalizado:")
        print(f"  Kills: {kills}")
        print(f"  Health: {health}")
        print(f"  Ammo: {ammo}")
        print(f"  Steps: {episode_steps}")
        
        # Pausa entre episódios
        if episode < num_episodes - 1:
            print("Pausa de 3 segundos antes do próximo episódio...")
            time.sleep(3)
    
    # Estatísticas finais
    if episodes_completed > 0:
        avg_kills = total_kills / episodes_completed
        avg_health = total_health / episodes_completed
        avg_ammo = total_ammo / episodes_completed
        avg_steps = total_steps / episodes_completed
        
        print(f"\n{'='*60}")
        print(f"📊 ESTATÍSTICAS FINAIS (média de {episodes_completed} episódios):")
        print(f"Kills por episódio: {avg_kills:.1f}")
        print(f"Health por episódio: {avg_health:.1f}")
        print(f"Ammo por episódio: {avg_ammo:.1f}")
        print(f"Steps por episódio: {avg_steps:.1f}")
        print(f"{'='*60}")
    
    game.close()
    print("Demonstração finalizada!")

def main():
    """Função principal do script de demonstração."""
    print("🎮 DEMONSTRADOR DE AGENTES GENÉTICOS")
    print("="*50)
    print("Este script demonstra agentes treinados pelo algoritmo genético.")
    print("Execute o treinamento primeiro com: python main.py")
    print("="*50)
    
    # Lista indivíduos salvos
    saved_files = list_saved_individuals()
    
    if not saved_files:
        print("❌ Nenhum indivíduo salvo encontrado.")
        print("📋 Execute o treinamento primeiro:")
        print("   python main.py")
        print("\n📁 Os indivíduos serão salvos em: saved_individuals/")
        return
    
    # Interface para escolher indivíduo
    print(f"\nEscolha um indivíduo para demonstrar:")
    for i, filepath in enumerate(saved_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            fitness = data.get('fitness', 'N/A')
            timestamp = data.get('timestamp', 'N/A')
            filename = os.path.basename(filepath)
            print(f"{i+1}. {filename}")
            print(f"   Fitness: {fitness}")
            print(f"   Data: {timestamp}")
        except Exception as e:
            print(f"{i+1}. {os.path.basename(filepath)} (erro ao ler: {e})")
    
    # Seleção do usuário
    try:
        choice = input(f"\nDigite o número (1-{len(saved_files)}): ").strip()
        if not choice:
            print("Operação cancelada.")
            return
            
        choice_idx = int(choice) - 1
        if not (0 <= choice_idx < len(saved_files)):
            print("❌ Escolha inválida.")
            return
            
        selected_file = saved_files[choice_idx]
        
        # Carrega o indivíduo
        print(f"\nCarregando indivíduo...")
        individual = load_individual(selected_file)
        
        # Configurações da demonstração
        print(f"\nConfigurações da demonstração:")
        num_episodes_input = input("Número de episódios (padrão: 3): ").strip()
        num_episodes = int(num_episodes_input) if num_episodes_input else 3
        
        delay_input = input("Delay entre ações em segundos (padrão: 0.05): ").strip()
        delay = float(delay_input) if delay_input else 0.05
        
        # Confirmação
        print(f"\nDemonstrando indivíduo com fitness {individual['fitness']:.2f}")
        print(f"Episódios: {num_episodes}")
        print(f"Delay: {delay}s")
        
        confirm = input("Continuar? (s/n): ").strip().lower()
        if confirm not in ['s', 'sim', 'y', 'yes']:
            print("Demonstração cancelada.")
            return
        
        # Inicializa jogo para obter as ações
        print("Inicializando jogo...")
        game, actions = initialize_game(headless=True)
        game.close()
        
        # Demonstra o indivíduo
        demonstrate_individual(individual, actions, num_episodes, delay)
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n❌ Operação cancelada: {e}")
    except Exception as e:
        print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    main() 