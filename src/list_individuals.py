#!/usr/bin/env python3
"""
Script para listar rapidamente os indivíduos salvos.
"""

import os
import json
import sys

def list_individuals():
    """Lista todos os indivíduos salvos."""
    save_dir = "../saved_individuals"
    
    if not os.path.exists(save_dir):
        print("❌ Diretório 'saved_individuals' não encontrado.")
        print("Execute o treinamento primeiro: python main.py")
        return
    
    files = [f for f in os.listdir(save_dir) if f.endswith('.json')]
    
    if not files:
        print("❌ Nenhum indivíduo salvo encontrado.")
        print("Execute o treinamento primeiro: python main.py")
        return
    
    # Ordena por data de modificação (mais recente primeiro)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True)
    
    print(f"📁 Indivíduos salvos ({len(files)} encontrados):")
    print("="*60)
    
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            fitness = data.get('fitness', 'N/A')
            timestamp = data.get('timestamp', 'N/A')
            
            print(f"{i:2d}. {filename}")
            print(f"    Fitness: {fitness}")
            print(f"    Data: {timestamp}")
            print()
            
        except Exception as e:
            print(f"{i:2d}. {filename} (erro ao ler: {e})")
            print()
    
    print("🎮 Para demonstrar um agente, execute: python demo.py")

if __name__ == "__main__":
    list_individuals() 