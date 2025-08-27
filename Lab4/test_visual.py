#!/usr/bin/env python3
"""
Prueba rápida del algoritmo TSP con visualización
"""

import numpy as np
from tsp_genetic_algorithm import TSPGeneticAlgorithm

def main():
    print("=== PRUEBA TSP CON VISUALIZACION ===")
    
    # Crear ciudades de ejemplo
    np.random.seed(42)
    num_ciudades = 10
    ciudades = np.random.rand(num_ciudades, 2) * 100
    
    print(f"Probando con {num_ciudades} ciudades")
    print("Coordenadas de las ciudades:")
    for i, (x, y) in enumerate(ciudades):
        print(f"  Ciudad {i}: ({x:.2f}, {y:.2f})")
    
    # Configurar algoritmo
    tsp = TSPGeneticAlgorithm(
        population_size=50,
        max_iterations=200,
        survivor_percentage=0.3,
        crossover_percentage=0.5,
        mutation_percentage=0.2,
        elite_percentage=0.1
    )
    
    # Cargar datos
    tsp.load_coordinates(ciudades)
    
    print("\nEjecutando algoritmo con visualizacion...")
    print("Se mostrara una ventana con:")
    print("- Izquierda: Mejor ruta actual")
    print("- Derecha: Convergencia del algoritmo")
    
    # Ejecutar con visualización
    tsp.solve(
        selection_method='tournament',
        crossover_method='order',
        mutation_method='swap',
        visualize=True,
        update_interval=10  # Actualizar cada 10 generaciones
    )
    
    print("\n¡Ejecución completada! Revisa la información detallada mostrada arriba.")

if __name__ == "__main__":
    main()