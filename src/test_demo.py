#!/usr/bin/env python3
"""
Script de teste para verificar se as funcionalidades de demonstração estão funcionando.
"""

import os
import json
import sys

def test_save_load():
    """Testa as funções de salvar e carregar indivíduos."""
    print("🧪 Testando funções de salvar/carregar...")
    
    # Importa funções do main.py
    try:
        from main import save_individual, load_individual, list_saved_individuals
    except ImportError as e:
        print(f"❌ Erro ao importar: {e}")
        return False
    
    # Cria um indivíduo de teste
    test_individual = {
        'genome': [1, 2, 3, 4, 5] * 200,  # 1000 genes
        'fitness': 1234.56
    }
    
    # Testa salvar
    try:
        filepath = save_individual(test_individual, "test_individual.json")
        print(f"✅ Indivíduo salvo em: {filepath}")
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        return False
    
    # Testa carregar
    try:
        loaded_individual = load_individual(filepath)
        if loaded_individual['fitness'] == test_individual['fitness']:
            print("✅ Indivíduo carregado corretamente")
        else:
            print("❌ Fitness não corresponde")
            return False
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return False
    
    # Testa listar
    try:
        files = list_saved_individuals()
        if len(files) > 0:
            print(f"✅ Listagem funcionando: {len(files)} arquivos encontrados")
        else:
            print("❌ Nenhum arquivo encontrado")
            return False
    except Exception as e:
        print(f"❌ Erro ao listar: {e}")
        return False
    
    # Limpa arquivo de teste
    try:
        os.remove(filepath)
        print("✅ Arquivo de teste removido")
    except:
        pass
    
    return True

def test_game_initialization():
    """Testa a inicialização do jogo."""
    print("\n🎮 Testando inicialização do jogo...")
    
    try:
        from main import initialize_game
    except ImportError as e:
        print(f"❌ Erro ao importar: {e}")
        return False
    
    try:
        # Testa inicialização headless (mais rápida)
        game, actions = initialize_game(headless=True)
        print(f"✅ Jogo inicializado com {len(actions)} ações possíveis")
        game.close()
        print("✅ Jogo fechado corretamente")
        return True
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return False

def main():
    """Função principal de teste."""
    print("🧪 TESTE DAS FUNCIONALIDADES DE DEMONSTRAÇÃO")
    print("="*50)
    
    # Testa funções básicas
    if not test_save_load():
        print("\n❌ Teste de salvar/carregar falhou")
        return False
    
    # Testa inicialização do jogo
    if not test_game_initialization():
        print("\n❌ Teste de inicialização do jogo falhou")
        return False
    
    print("\n✅ Todos os testes passaram!")
    print("\n📋 Próximos passos:")
    print("1. Execute o treinamento: python main.py")
    print("2. Demonstre os agentes: python demo.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 