import numpy as np
from env_setup import create_env
from genetic import (
    adaptive_mut_rate, init_population, evaluate, tournament, crossover, mutate,
    POP_SIZE, GENOME_LEN
)
from realtime_plot import RealtimePlot

NUM_GENS = 1000

def main():
    env = create_env(render=False)
    pop = init_population(env)
    plot = RealtimePlot(max_gens=NUM_GENS)

    for gen in range(NUM_GENS):
      fits = [evaluate(env, ind) for ind in pop]
      best_score = max(fits)
      avg_score = sum(fits) / len(fits)

      print(f"Geração {gen+1} — melhor: {best_score:.2f}, média: {avg_score:.2f}")
      plot.update(best_score, avg_score)

      elite_count = max(1, POP_SIZE // 10)
      elite_indices = np.argsort(fits)[-elite_count:]
      elites = [pop[i] for i in elite_indices]
      new_pop = elites.copy()
      while len(new_pop) < POP_SIZE:
         p1 = tournament(pop, fits)
         p2 = tournament(pop, fits)
         c1, c2 = crossover(p1, p2)
         current_mut_rate = adaptive_mut_rate(gen, NUM_GENS)
         mutate(env, c1, current_mut_rate)
         mutate(env, c2, current_mut_rate)
         new_pop.extend([c1, c2])
      
      pop = new_pop[:POP_SIZE]

    env.close()
    plot.finalize()

if __name__ == "__main__":
    main()