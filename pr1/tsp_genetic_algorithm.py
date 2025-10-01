import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
import copy
import time

class TSPGeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 100,
                 max_iterations: int = 1000,
                 survivor_percentage: float = 0.3,
                 crossover_percentage: float = 0.5,
                 mutation_percentage: float = 0.2,
                 elite_percentage: float = 0.1):
        
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.survivor_percentage = survivor_percentage
        self.crossover_percentage = crossover_percentage
        self.mutation_percentage = mutation_percentage
        self.elite_percentage = elite_percentage
        
        self.cities = None
        self.distance_matrix = None
        self.num_cities = 0
        self.population = []
        self.fitness_history = []
        
    def load_coordinates(self, coordinates: Union[List[Tuple[float, float]], np.ndarray, str]):
        if isinstance(coordinates, str):
            coordinates = np.loadtxt(coordinates, delimiter=',')
        
        self.cities = np.array(coordinates)
        self.num_cities = len(self.cities)
        self._calculate_distance_matrix()
        
    def load_distance_matrix(self, distance_matrix: Union[np.ndarray, str]):
        if isinstance(distance_matrix, str):
            self.distance_matrix = np.loadtxt(distance_matrix, delimiter=',')
        else:
            self.distance_matrix = np.array(distance_matrix)
        
        self.num_cities = self.distance_matrix.shape[0]
        
    def _calculate_distance_matrix(self):
        n = self.num_cities
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.cities[i][0] - self.cities[j][0]
                    dy = self.cities[i][1] - self.cities[j][1]
                    self.distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)
                    
    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            tour = list(range(self.num_cities))
            random.shuffle(tour)
            self.population.append(tour)
            
    def _calculate_fitness(self, tour: List[int]) -> float:
        total_distance = 0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[current_city][next_city]
        return 1.0 / (1.0 + total_distance)
    
    def _calculate_tour_distance(self, tour: List[int]) -> float:
        total_distance = 0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[current_city][next_city]
        return total_distance
    
    def _tournament_selection(self, tournament_size: int = 5) -> List[int]:
        tournament = random.sample(self.population, tournament_size)
        tournament_fitness = [self._calculate_fitness(tour) for tour in tournament]
        winner_index = tournament_fitness.index(max(tournament_fitness))
        return tournament[winner_index]
    
    def _roulette_selection(self) -> List[int]:
        fitness_scores = [self._calculate_fitness(tour) for tour in self.population]
        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            return random.choice(self.population)
        
        probabilities = [f / total_fitness for f in fitness_scores]
        return self.population[np.random.choice(len(self.population), p=probabilities)]
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        offspring1 = [-1] * size
        offspring2 = [-1] * size
        
        offspring1[start:end] = parent1[start:end]
        offspring2[start:end] = parent2[start:end]
        
        def fill_offspring(offspring, other_parent):
            remaining = [item for item in other_parent if item not in offspring]
            j = 0
            for i in range(size):
                if offspring[i] == -1:
                    offspring[i] = remaining[j]
                    j += 1
        
        fill_offspring(offspring1, parent2)
        fill_offspring(offspring2, parent1)
        
        return offspring1, offspring2
    
    def _pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        offspring1 = parent1[:]
        offspring2 = parent2[:]
        
        mapping1 = {}
        mapping2 = {}
        
        for i in range(start, end):
            mapping1[parent1[i]] = parent2[i]
            mapping2[parent2[i]] = parent1[i]
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]
        
        def resolve_conflicts(offspring, mapping):
            for i in range(size):
                if i < start or i >= end:
                    while offspring[i] in mapping:
                        offspring[i] = mapping[offspring[i]]
        
        resolve_conflicts(offspring1, mapping1)
        resolve_conflicts(offspring2, mapping2)
        
        return offspring1, offspring2
    
    def _swap_mutation(self, tour: List[int]) -> List[int]:
        mutated_tour = tour[:]
        i, j = random.sample(range(len(tour)), 2)
        mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]
        return mutated_tour
    
    def _inversion_mutation(self, tour: List[int]) -> List[int]:
        mutated_tour = tour[:]
        i, j = sorted(random.sample(range(len(tour)), 2))
        mutated_tour[i:j+1] = reversed(mutated_tour[i:j+1])
        return mutated_tour
    
    def _scramble_mutation(self, tour: List[int]) -> List[int]:
        mutated_tour = tour[:]
        i, j = sorted(random.sample(range(len(tour)), 2))
        subset = mutated_tour[i:j+1]
        random.shuffle(subset)
        mutated_tour[i:j+1] = subset
        return mutated_tour
    
    def solve(self, selection_method: str = 'tournament', 
              crossover_method: str = 'order',
              mutation_method: str = 'swap',
              visualize: bool = False,
              update_interval: int = 50) -> Tuple[List[int], float]:
        
        if self.distance_matrix is None:
            raise ValueError("No input data provided. Use load_coordinates or load_distance_matrix first.")
        
        self._initialize_population()
        self.fitness_history = []
        
        if visualize and self.cities is not None:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax1.set_title('Mejor Ruta TSP')
            ax1.set_xlabel('Coordenada X')
            ax1.set_ylabel('Coordenada Y')
            ax1.grid(True, alpha=0.3)
            
            ax2.set_title('Convergencia del Algoritmo')
            ax2.set_xlabel('Generación')
            ax2.set_ylabel('Mejor Distancia')
            ax2.grid(True, alpha=0.3)
            
        for generation in range(self.max_iterations):
            fitness_scores = [self._calculate_fitness(tour) for tour in self.population]
            distances = [self._calculate_tour_distance(tour) for tour in self.population]
            
            best_fitness = max(fitness_scores)
            best_distance = min(distances)
            self.fitness_history.append(best_distance)
            
            if generation % 100 == 0:
                print(f"Generation {generation}: Best distance = {best_distance:.2f}")
            
            if visualize and self.cities is not None and generation % update_interval == 0:
                ax1.clear()
                ax2.clear()
                
                best_index = distances.index(best_distance)
                best_tour = self.population[best_index]
                
                x = [self.cities[i][0] for i in best_tour] + [self.cities[best_tour[0]][0]]
                y = [self.cities[i][1] for i in best_tour] + [self.cities[best_tour[0]][1]]
                
                ax1.plot(x, y, '-', linewidth=2, alpha=0.7, color='blue')
                ax1.plot(x, y, 'o', markersize=6, color='blue', alpha=0.7)
                ax1.scatter([self.cities[i][0] for i in range(self.num_cities)], 
                           [self.cities[i][1] for i in range(self.num_cities)], 
                           c='red', s=100, alpha=0.8, zorder=5)
                
                for i, city in enumerate(range(self.num_cities)):
                    ax1.annotate(str(city), (self.cities[city][0], self.cities[city][1]), 
                                xytext=(5, 5), textcoords='offset points', fontsize=10, 
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                ax1.set_title(f'Mejor Ruta TSP - Gen {generation} (Dist: {best_distance:.2f})')
                ax1.set_xlabel('Coordenada X')
                ax1.set_ylabel('Coordenada Y')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(range(len(self.fitness_history)), self.fitness_history, 'g-', linewidth=2)
                ax2.set_title('Convergencia del Algoritmo Genético')
                ax2.set_xlabel('Generación')
                ax2.set_ylabel('Mejor Distancia')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.pause(0.01)
                plt.draw()
            
            population_with_fitness = list(zip(self.population, fitness_scores))
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)
            
            num_survivors = int(self.population_size * self.survivor_percentage)
            num_elite = int(self.population_size * self.elite_percentage)
            num_crossover = int(self.population_size * self.crossover_percentage)
            num_mutation = self.population_size - num_survivors - num_crossover
            
            new_population = []
            
            for i in range(num_elite):
                new_population.append(population_with_fitness[i][0][:])
            
            for i in range(num_elite, num_survivors):
                new_population.append(population_with_fitness[i][0][:])
            
            for _ in range(0, num_crossover, 2):
                if selection_method == 'tournament':
                    parent1 = self._tournament_selection()
                    parent2 = self._tournament_selection()
                else:
                    parent1 = self._roulette_selection()
                    parent2 = self._roulette_selection()
                
                if crossover_method == 'order':
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = self._pmx_crossover(parent1, parent2)
                
                new_population.extend([child1, child2])
            
            for _ in range(num_mutation):
                if selection_method == 'tournament':
                    parent = self._tournament_selection()
                else:
                    parent = self._roulette_selection()
                
                if mutation_method == 'swap':
                    mutated_child = self._swap_mutation(parent)
                elif mutation_method == 'inversion':
                    mutated_child = self._inversion_mutation(parent)
                else:
                    mutated_child = self._scramble_mutation(parent)
                
                new_population.append(mutated_child)
            
            self.population = new_population[:self.population_size]
        
        final_fitness_scores = [self._calculate_fitness(tour) for tour in self.population]
        best_index = final_fitness_scores.index(max(final_fitness_scores))
        best_tour = self.population[best_index]
        best_distance = self._calculate_tour_distance(best_tour)
        
        if visualize and self.cities is not None:
            ax1.clear()
            ax2.clear()
            
            x = [self.cities[i][0] for i in best_tour] + [self.cities[best_tour[0]][0]]
            y = [self.cities[i][1] for i in best_tour] + [self.cities[best_tour[0]][1]]
            
            ax1.plot(x, y, '-', linewidth=3, markersize=10, alpha=0.8, color='darkblue')
            ax1.plot(x, y, 'o', markersize=8, color='darkblue', alpha=0.6)
            ax1.scatter([self.cities[i][0] for i in range(self.num_cities)], 
                       [self.cities[i][1] for i in range(self.num_cities)], 
                       c='red', s=150, alpha=0.9, zorder=5, edgecolor='black', linewidth=2)
            
            for i, city in enumerate(range(self.num_cities)):
                ax1.annotate(str(city), (self.cities[city][0], self.cities[city][1]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=12, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
            
            ax1.set_title(f'RESULTADO FINAL - Mejor Ruta TSP (Distancia: {best_distance:.2f})', 
                         fontsize=14, weight='bold')
            ax1.set_xlabel('Coordenada X')
            ax1.set_ylabel('Coordenada Y')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(range(len(self.fitness_history)), self.fitness_history, 'g-', linewidth=3)
            ax2.set_title('Convergencia Final del Algoritmo Genético', fontsize=14, weight='bold')
            ax2.set_xlabel('Generación')
            ax2.set_ylabel('Mejor Distancia')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(2)
            plt.ioff()
            plt.show()
        
        print(f"\n=== RESULTADO FINAL ===")
        print(f"Mejor recorrido encontrado: {best_tour}")
        print(f"Distancia total del recorrido: {best_distance:.2f}")
        
        # Mostrar la ruta detallada en consola
        self._print_detailed_route(best_tour, best_distance)
        
        return best_tour, best_distance
    
    def _print_detailed_route(self, tour: List[int], total_distance: float):
        """Imprime la ruta detallada en consola"""
        if self.cities is None:
            print("\n(No se pueden mostrar coordenadas porque no se proporcionaron)")
            return
        
        print(f"\n{'='*60}")
        print("RUTA DETALLADA:")
        print(f"{'='*60}")
        
        for i, ciudad in enumerate(tour):
            x, y = self.cities[ciudad]
            if i == len(tour) - 1:
                siguiente_ciudad = tour[0]
                x_sig, y_sig = self.cities[siguiente_ciudad]
                distancia = self.distance_matrix[ciudad][siguiente_ciudad]
                print(f"Paso {i+1:2d}: Ciudad {ciudad} ({x:.2f}, {y:.2f}) -> Ciudad {siguiente_ciudad} ({x_sig:.2f}, {y_sig:.2f}) [Distancia: {distancia:.2f}]")
            else:
                siguiente_ciudad = tour[i+1]
                x_sig, y_sig = self.cities[siguiente_ciudad]
                distancia = self.distance_matrix[ciudad][siguiente_ciudad]
                print(f"Paso {i+1:2d}: Ciudad {ciudad} ({x:.2f}, {y:.2f}) -> Ciudad {siguiente_ciudad} ({x_sig:.2f}, {y_sig:.2f}) [Distancia: {distancia:.2f}]")
        
        print(f"\nRuta completa: {' -> '.join(map(str, tour))} -> {tour[0]} (vuelta al inicio)")
        print(f"Distancia total verificada: {total_distance:.2f}")
        print(f"{'='*60}")
    
    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Convergencia del Algoritmo Genético TSP')
        plt.xlabel('Generación')
        plt.ylabel('Mejor Distancia')
        plt.grid(True)
        plt.show()
    
    def plot_tour(self, tour: List[int]):
        if self.cities is None:
            print("No se pueden graficar las coordenadas porque no se proporcionaron coordenadas de ciudades.")
            return
        
        plt.figure(figsize=(10, 8))
        
        x = [self.cities[i][0] for i in tour] + [self.cities[tour[0]][0]]
        y = [self.cities[i][1] for i in tour] + [self.cities[tour[0]][1]]
        
        plt.plot(x, y, 'bo-', linewidth=2, markersize=8)
        plt.scatter([self.cities[i][0] for i in range(self.num_cities)], 
                   [self.cities[i][1] for i in range(self.num_cities)], 
                   c='red', s=100, alpha=0.7)
        
        for i, city in enumerate(tour):
            plt.annotate(str(city), (self.cities[city][0], self.cities[city][1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.title('Mejor Ruta TSP Encontrada')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.grid(True, alpha=0.3)
        plt.show()