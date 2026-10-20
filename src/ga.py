import random
from .chromosome import Chromosome
from .fitness import evaluate
from .operators import tournament_selection, crossover, mutate

class GeneticAlgorithm:
    def __init__(self, env, cfg, population_size=None, generations=None,
                 mutation_rate=None, seed=None):
        self.env, self.cfg = env, cfg
        self.population_size = population_size or cfg.POPULATION_SIZE
        self.generations = generations or cfg.GENERATIONS
        self.mutation_rate = cfg.MUTATION_RATE if mutation_rate is None else mutation_rate
        self.rng = random.Random(cfg.RANDOM_SEED if seed is None else seed)

    def run(self):
        pop = [Chromosome.random(self.env, self.cfg.NUM_WAYPOINTS, self.rng)
               for _ in range(self.population_size)]
        best_history, avg_history = [], []
        best_overall, best_score = None, float("-inf")

        for _ in range(self.generations):
            scored = [evaluate(self.env, c, self.cfg)[0] for c in pop]
            gen_best_idx = max(range(len(pop)), key=lambda i: scored[i])
            if scored[gen_best_idx] > best_score:
                best_score = scored[gen_best_idx]
                best_overall = pop[gen_best_idx].copy()
            best_history.append(max(scored))
            avg_history.append(sum(scored) / len(scored))

            ranked = sorted(zip(scored, pop), key=lambda x: x[0], reverse=True)
            new_pop = [c.copy() for _, c in ranked[:self.cfg.ELITE_COUNT]]
            while len(new_pop) < self.population_size:
                p1 = tournament_selection(pop, scored, self.cfg.TOURNAMENT_SIZE, self.rng)
                p2 = tournament_selection(pop, scored, self.cfg.TOURNAMENT_SIZE, self.rng)
                if self.rng.random() < self.cfg.CROSSOVER_RATE:
                    c1, c2 = crossover(p1, p2, self.rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                new_pop.append(mutate(c1, self.env, self.mutation_rate, self.rng))
                if len(new_pop) < self.population_size:
                    new_pop.append(mutate(c2, self.env, self.mutation_rate, self.rng))

        return best_overall, best_history, avg_history
