# 🕹️ ViZDoom + Gymnasium no WSL (Windows)

Este projeto utiliza o [ViZDoom](https://vizdoom.farama.org/) em conjunto com o [Gymnasium](https://gymnasium.farama.org/) para testar e treinar agentes em ambientes do clássico jogo Doom. O ambiente está configurado para rodar no **WSL (Windows Subsystem for Linux)**.

---

## 🧬 Algoritmo Genético aplicado ao ViZDoom
Este projeto utiliza um algoritmo genético para treinar um agente a jogar Doom. Abaixo estão os principais componentes do processo evolutivo:

**1. O que está sendo otimizado?**

O comportamento do agente para maximizar sua pontuação no ambiente, eliminando inimigos e coletando itens com eficiência.

**2. Representação da solução (genoma)**

Sequência de ações ou parâmetros codificados como vetores (floats ou inteiros) representando comandos do jogo ao longo do tempo.

3. Função de fitness

Combina desempenho ofensivo, exploração e eficiência temporal.

```python
fitness = (kills + itens_coletados) / tempo_total
```

4. Método de seleção

Torneio ou roleta viciada, favorecendo indivíduos com maior fitness.

5. Crossover

Cruzamento de um ponto ou uniforme entre dois genomas, misturando comandos ou estratégias parciais.

6. Inicialização

População inicial gerada com ações aleatórias dentro dos limites válidos do ambiente.

7. Critério de parada

Número máximo de gerações ou ausência de melhoria significativa após N gerações.

---

## ✅ Pré-requisitos

- **Windows 10 ou 11 com WSL2**
- **Ubuntu instalado no WSL**
- **Python 3.8 ou superior** (use `venv` de preferência)

> ℹ️ Caso ainda não tenha o WSL configurado, siga a [documentação oficial da Microsoft](https://learn.microsoft.com/windows/wsl/install).

---

## ⚙️ Instalação do ambiente no Ubuntu (WSL)

Execute os comandos abaixo no terminal Ubuntu (WSL):

### 1. Atualize os pacotes e instale as dependências

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
    libsdl2-dev libboost-all-dev libopenal-dev \
    python3-dev python3-pip
```

### 2. Crie um ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale o ViZDoom diretamente do repositório

```bash
pip install git+https://github.com/Farama-Foundation/ViZDoom
```

### 4. Instale o Gymnasium

```bash
pip install gymnasium
```