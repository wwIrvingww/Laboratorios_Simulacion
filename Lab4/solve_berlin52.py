import numpy as np
from tsp_genetic_algorithm import TSPGeneticAlgorithm
import time

def load_berlin52_coordinates():
    coordinates = []
    
    with open("berlin52.tsp/berlin52.tsp", 'r') as file:
        reading_coords = False
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            elif line == "EOF":
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates.append((x, y))
    
    return np.array(coordinates)

def main():
    print("=== SOLVING BERLIN52 TSP WITH GENETIC ALGORITHM ===\n")
    
    coordinates = load_berlin52_coordinates()
    print(f"Loaded {len(coordinates)} cities from Berlin52 dataset")
    
    tsp_solver = TSPGeneticAlgorithm(
        population_size=200,
        max_iterations=2000,
        survivor_percentage=0.3,
        crossover_percentage=0.5,
        mutation_percentage=0.2,
        elite_percentage=0.1
    )
    
    tsp_solver.load_coordinates(coordinates)
    
    print("Starting optimization...")
    print("Parameters:")
    print(f"  Population size: {tsp_solver.population_size}")
    print(f"  Max iterations: {tsp_solver.max_iterations}")
    print(f"  Survivor percentage: {tsp_solver.survivor_percentage}")
    print(f"  Crossover percentage: {tsp_solver.crossover_percentage}")
    print(f"  Mutation percentage: {tsp_solver.mutation_percentage}")
    print(f"  Elite percentage: {tsp_solver.elite_percentage}")
    print()
    
    start_time = time.time()
    
    best_tour, best_distance = tsp_solver.solve(
        selection_method='tournament',
        crossover_method='order',
        mutation_method='swap',
        visualize=True,
        update_interval=100
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Best distance found: {best_distance:.2f}")
    print(f"Best tour: {best_tour}")
    
    print("\nOptimal distance for Berlin52 is approximately 7542.0")
    print(f"Gap from optimal: {((best_distance - 7542.0) / 7542.0) * 100:.2f}%")

if __name__ == "__main__":
    main()