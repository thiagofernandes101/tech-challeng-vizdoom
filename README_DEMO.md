# Demonstração de Agentes Genéticos - ViZDoom

Este projeto implementa um algoritmo genético para treinar agentes no jogo Doom usando ViZDoom. Agora inclui funcionalidades para salvar e demonstrar os melhores agentes encontrados.

## 🎮 Funcionalidades

### 1. Treinamento Automático (Sem Tela)
- Executa o algoritmo genético para evoluir agentes
- **Roda completamente sem mostrar tela** (headless)
- Salva automaticamente o melhor indivíduo quando há melhoria
- Salva o melhor indivíduo final ao terminar o treinamento

### 2. Demonstração Visual (Com Tela)
- **Completamente separada do treinamento**
- Visualiza como o melhor agente se comporta no jogo
- Executa múltiplos episódios para estatísticas
- Interface interativa para escolher qual agente demonstrar
- Pode ser executada a qualquer momento, independente do treinamento

## 📁 Estrutura de Arquivos

```
src/
├── main.py              # Script de treinamento (sem tela)
├── demo.py              # Script de demonstração (com tela)
├── list_individuals.py  # Lista agentes salvos
├── test_demo.py         # Script de teste
└── saved_individuals/   # Diretório com indivíduos salvos
    ├── individual_fitness_1234.56_20231201_143022.json
    ├── individual_fitness_2345.67_20231201_150145.json
    └── best_individual_final.json
```

## 🚀 Como Usar

### 1. Treinamento (Sem Tela)

Execute o treinamento do algoritmo genético:

```bash
cd src
python main.py
```

O script irá:
- Executar o treinamento **sem mostrar tela** (headless)
- Criar uma população inicial de agentes
- Executar o algoritmo genético por várias gerações
- Salvar automaticamente o melhor indivíduo quando houver melhoria
- Salvar o melhor indivíduo final ao terminar

### 2. Listar Agentes Salvos

Para ver quais agentes foram treinados:

```bash
cd src
python list_individuals.py
```

### 3. Demonstração (Com Tela)

Após o treinamento, você pode demonstrar os agentes salvos:

```bash
cd src
python demo.py
```

**Importante**: A demonstração é completamente separada do treinamento. Você pode executá-la a qualquer momento, mesmo dias depois do treinamento.

### 3. Interface de Demonstração

O script de demonstração oferece:

1. **Lista de agentes salvos** - Mostra todos os indivíduos disponíveis com fitness e data
2. **Seleção interativa** - Escolha qual agente demonstrar
3. **Configurações personalizáveis**:
   - Número de episódios (padrão: 3)
   - Delay entre ações (padrão: 0.05s)
4. **Estatísticas detalhadas** - Kills, health, ammo e steps por episódio

## 📊 Formato dos Arquivos Salvos

Cada indivíduo salvo contém:

```json
{
  "genome": [1, 5, 2, 8, ...],  // Sequência de ações do agente
  "fitness": 1234.56,           // Pontuação do agente
  "timestamp": "2023-12-01 14:30:22",
  "parameters": {
    "population_size": 100,
    "genome_length": 1000,
    "w_kills": 150.0,
    "w_health": 1.0,
    "w_ammo": 0.2,
    "w_steps": -0.5
  }
}
```

## ⚙️ Configurações

### Parâmetros do Algoritmo Genético
Edite as constantes no início de `main.py`:

```python
POPULATION_SIZE = 100      # Tamanho da população
GENOME_LENGTH = 1000       # Duração máxima do episódio
W_KILLS = 150.0           # Peso para kills
W_HEALTH = 1.0            # Peso para health
W_AMMO = 0.2              # Peso para ammo
W_STEPS = -0.5            # Penalidade por steps
```

### Configurações de Demonstração
No script `demo.py`, você pode ajustar:
- `delay` - Velocidade da demonstração
- `num_episodes` - Quantos episódios executar

## 🎯 Dicas de Uso

1. **Treinamento**: Execute o treinamento primeiro para gerar agentes
2. **Demonstração**: Use o script `demo.py` para visualizar os resultados
3. **Comparação**: Execute múltiplos agentes para comparar performance
4. **Ajuste de parâmetros**: Modifique os pesos de fitness para diferentes comportamentos

## 🔧 Solução de Problemas

### Erro: "Nenhum indivíduo salvo encontrado"
- Execute o treinamento primeiro: `python main.py`
- Verifique se o diretório `saved_individuals/` foi criado

### Erro: "DoomGame is not a known attribute"
- Verifique se o ViZDoom está instalado corretamente
- Execute: `pip install vizdoom`

### Demonstração muito rápida/lenta
- Ajuste o parâmetro `delay` no script de demonstração
- Valores menores = mais rápido, valores maiores = mais lento

## 📈 Interpretando os Resultados

- **Fitness**: Pontuação geral do agente (maior = melhor)
- **Kills**: Número de inimigos eliminados
- **Health**: Vida restante (maior = melhor)
- **Ammo**: Munição restante
- **Steps**: Passos dados (menor = mais eficiente)

## 🎮 Controles do Jogo

Durante a demonstração, o agente executa automaticamente. O jogo mostra:
- Visão em primeira pessoa do agente
- Interface do Doom com health, ammo, etc.
- Ações sendo executadas em tempo real 