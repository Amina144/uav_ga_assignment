from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .fitness import evaluate

def save_plots(env, best, best_history, avg_history, cfg, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, metrics, route = evaluate(env, best, cfg)

    plt.figure(figsize=(8, 4.5))
    plt.plot(best_history, label="Best fitness")
    plt.plot(avg_history, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (higher is better)")
    plt.title("GA Fitness Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fitness_convergence.png", dpi=160)
    plt.close()

    grid = np.zeros((env.size, env.size))
    for x, y in env.obstacles: grid[y, x] = 1
    for x, y in env.hazards: grid[y, x] = 2
    for x, y in env.targets: grid[y, x] = 3
    plt.figure(figsize=(7, 7))
    plt.imshow(grid, origin="lower", interpolation="nearest")
    rx, ry = zip(*route)
    plt.plot(rx, ry, marker=".", linewidth=1.5, label="Best UAV route")
    plt.scatter([env.start[0]], [env.start[1]], marker="s", s=70, label="Start")
    plt.scatter([env.goal[0]], [env.goal[1]], marker="*", s=120, label="Goal")
    plt.xlim(-0.5, env.size - 0.5); plt.ylim(-0.5, env.size - 0.5)
    plt.xlabel("X coordinate"); plt.ylabel("Y coordinate")
    plt.title("Evolved UAV Route (1=obstacle, 2=hazard, 3=target)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "best_route.png", dpi=160)
    plt.close()

    visits = np.zeros((env.size, env.size))
    for x, y in route:
        if 0 <= x < env.size and 0 <= y < env.size:
            visits[y, x] += 1
    plt.figure(figsize=(7, 6))
    plt.imshow(visits, origin="lower", interpolation="nearest")
    plt.colorbar(label="Number of route visits")
    plt.xlabel("X coordinate"); plt.ylabel("Y coordinate")
    plt.title("Route Coverage Heatmap")
    plt.tight_layout()
    plt.savefig(out / "coverage_heatmap.png", dpi=160)
    plt.close()
    return metrics, route

def save_sensitivity(env, cfg, out_dir):
    from .ga import GeneticAlgorithm
    out = Path(out_dir)
    cases = [
        ("Small population", 20, 0.18),
        ("Default", 60, 0.18),
        ("Large population", 100, 0.18),
        ("Low mutation", 60, 0.05),
        ("High mutation", 60, 0.35),
    ]
    scores = []
    for label, pop_size, mut in cases:
        ga = GeneticAlgorithm(env, cfg, population_size=pop_size, generations=35,
                              mutation_rate=mut, seed=cfg.RANDOM_SEED)
        best, _, _ = ga.run()
        scores.append((label, __import__("src.fitness", fromlist=["evaluate"]).evaluate(env, best, cfg)[0]))
    plt.figure(figsize=(8, 4.5))
    plt.bar([x[0] for x in scores], [x[1] for x in scores])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Best fitness after 35 generations")
    plt.title("Hyperparameter Sensitivity (one seeded run per setting)")
    plt.tight_layout()
    plt.savefig(out / "hyperparameter_sensitivity.png", dpi=160)
    plt.close()
    return {label: score for label, score in scores}
