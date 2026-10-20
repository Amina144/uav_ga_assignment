# UAV Disaster-Surveillance Strategy Engine (Genetic Algorithm)

A beginner-friendly, modular Python simulation for a UAV route-planning assignment.

## What the simulation models
- A 20 x 20 discretized grid (easy to run; change `GRID_SIZE` in `src/config.py` to 50 if required).
- Start and goal positions, blocked cells, hazard cells, and high-priority target cells.
- A chromosome containing waypoint coordinates, speed settings, and binary sensor activation values.
- A weighted fitness function that rewards coverage and target detection while penalizing energy, time, and hazard exposure.
- Tournament selection, segment-based crossover, spatial mutation, and repair of blocked/out-of-bounds waypoints.
- A random-search baseline and plots for fitness convergence, route paths, and coverage heatmaps.

## Requirements
Python 3.10+ recommended.

## Run locally
Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
python -m src.main
```

If running from an IDE, run `src/main.py` from the project root. The program writes plots and a JSON summary to `outputs/`.

## Output files
- `outputs/fitness_convergence.png`: best and average fitness per generation.
- `outputs/best_route.png`: grid map, obstacles, hazards, targets, and best route.
- `outputs/coverage_heatmap.png`: cells visited by the best route.
- `outputs/summary.json`: best fitness and metric summary.
- `outputs/hyperparameter_sensitivity.png`: illustrative sensitivity experiment.

## Important modelling note
This is a teaching simulation, not flight-control software. It does not control real drones. The example uses a single UAV to keep the assignment understandable; the chromosome design can be extended to multiple UAVs by adding a drone-index segment for each vehicle.
