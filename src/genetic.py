# genetic.py
import numpy as np

# Parâmetros do algoritmo
POP_SIZE = 100
GENOME_LEN = 200
MUT_RATE = 0.05

def init_population(env, pop_size=POP_SIZE, genome_len=GENOME_LEN):
    return [
        [env.action_space.sample() for _ in range(genome_len)]
        for _ in range(pop_size)
    ]

def evaluate(env, individual):
    obs, info = env.reset()
    total = 0.0
    for step, action in enumerate(individual):
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        if term or trunc:
            # Penalidade leve para terminar cedo
            # total -= 5.0
            break
    return total

def tournament(pop, fits, k=3):
    indices = np.random.choice(len(pop), size=k, replace=False)
    best_idx = indices[0]
    for i in indices[1:]:
        if fits[i] > fits[best_idx]:
            best_idx = i
    return pop[best_idx]

def crossover(p1, p2):
    pt = np.random.randint(1, len(p1))
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

# def mutate(env, individual, mut_rate=MUT_RATE):
#     for i in range(len(individual)):
#         if np.random.rand() < mut_rate:
#             individual[i] = env.action_space.sample()
def mutate(env, individual, mut_rate):
    for i in range(len(individual)):
        if np.random.rand() < mut_rate:
            individual[i] = env.action_space.sample()

def adaptive_mut_rate(current_gen, max_gen, base=0.05, min_rate=0.005):
    return max(min_rate, base * (1 - current_gen / max_gen))