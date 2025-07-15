# 🕹️ ViZDoom + Gymnasium no WSL (Windows)

Este projeto utiliza o [ViZDoom](https://vizdoom.farama.org/) em conjunto com o [Gymnasium](https://gymnasium.farama.org/) para testar e treinar agentes em ambientes do clássico jogo Doom. O ambiente está configurado para rodar no **WSL (Windows Subsystem for Linux)**.

---

## 🧬 Algoritmo Genético aplicado ao ViZDoom
Este projeto utiliza um algoritmo genético para treinar um agente a jogar Doom. Abaixo estão os principais componentes do processo evolutivo:

**1. O que está sendo otimizado?**
A melhor sequência de ações para alcançar um objetivo específico no ambiente do jogo matando a maior quantidade de inimigos com o menor dano

**2. Representação da solução (genoma)**

Sequência de ações representando comandos do jogo ao longo do tempo.

**3. Função de fitness**

Combina desempenho ofensivo, exploração e eficiência temporal.

```python
Fitness = (wk * Kills)+(wh * Healthfinal)−(ws * Steps)
```
Exemplo:
- w_kills = 100
- w_health = 10
- w_steps = 0.5

**4. Método de seleção**

Torneio favorecendo indivíduos com maior fitness.

**5. Crossover**

Crossover de um Ponto:

- Escolha um ponto de corte aleatório ao longo do genoma.
- Crie um filho pegando a primeira parte do genoma do Pai A e a segunda parte do Pai B.
- O segundo filho pode ser criado com as partes restantes.

> Pai A: [A, A, A, A | A, A, A, A]
> 
> Pai B: [B, B, B, B | B, B, B, B]
> 
> Filho 1: [A, A, A, A | B, B, B, B]
> 
> Filho 2: [B, B, B, B | A, A, A, A]

**6. Inicialização**

População inicial gerada com ações aleatórias dentro dos limites válidos do ambiente.

**7. Critério de parada**

Ausência de melhoria significativa no fitness após N gerações.

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

# Citação

> M Wydmuch, M Kempka & W Jaśkowski, ViZDoom Competitions: Playing Doom from Pixels, IEEE Transactions on Games, vol. 11, no. 3, pp. 248-259, 2019
([arXiv:1809.03470](https://arxiv.org/abs/1809.03470))
```
@article{Wydmuch2019ViZdoom,
  author  = {Marek Wydmuch and Micha{\l} Kempka and Wojciech Ja\'skowski},
  title   = {{ViZDoom} {C}ompetitions: {P}laying {D}oom from {P}ixels},
  journal = {IEEE Transactions on Games},
  year    = {2019},
  volume  = {11},
  number  = {3},
  pages   = {248--259},
  doi     = {10.1109/TG.2018.2877047},
  note    = {The 2022 IEEE Transactions on Games Outstanding Paper Award}
}
```

or/and

> M. Kempka, M. Wydmuch, G. Runc, J. Toczek & W. Jaśkowski, ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning, IEEE Conference on Computational Intelligence and Games, pp. 341-348, Santorini, Greece, 2016	([arXiv:1605.02097](http://arxiv.org/abs/1605.02097))
```
@inproceedings{Kempka2016ViZDoom,
  author    = {Micha{\l} Kempka and Marek Wydmuch and Grzegorz Runc and Jakub Toczek and Wojciech Ja\'skowski},
  title     = {{ViZDoom}: A {D}oom-based {AI} Research Platform for Visual Reinforcement Learning},
  booktitle = {IEEE Conference on Computational Intelligence and Games},
  year      = {2016},
  address   = {Santorini, Greece},
  month     = {Sep},
  pages     = {341--348},
  publisher = {IEEE},
  doi       = {10.1109/CIG.2016.7860433},
  note      = {The Best Paper Award}
}
```