import json
from pathlib import Path
from . import config as cfg
from .environment import Environment
from .ga import GeneticAlgorithm
from .fitness import evaluate
from .plots import save_plots, save_sensitivity

def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs"
    env = Environment.demo(cfg.GRID_SIZE, cfg.RANDOM_SEED)
    ga = GeneticAlgorithm(env, cfg)
    best, best_history, avg_history = ga.run()
    metrics, route = save_plots(env, best, best_history, avg_history, cfg, out)
    sensitivity = save_sensitivity(env, cfg, out)
    result = {
        "grid_size": cfg.GRID_SIZE,
        "start": env.start,
        "goal": env.goal,
        "best_waypoints": best.waypoints,
        "speed_levels": best.speeds,
        "sensor_activation": best.sensors,
        "metrics": metrics,
        "sensitivity_best_fitness": sensitivity,
        "note": "Synthetic teaching simulation; not for real-world flight."
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Simulation complete. Files saved in:", out)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
