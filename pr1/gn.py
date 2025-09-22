import os
import sys
import time
import csv
import pandas as pd

# Añadir el directorio padre al path para poder importar Lab4
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Lab4.tsp_genetic_algorithm import TSPGeneticAlgorithm

def parse_tsp_file(file_path):
    """
    Parsea un archivo .tsp y extrae las coordenadas de las ciudades
    """
    coordinates = []
    reading_coords = False
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == 'NODE_COORD_SECTION':
                reading_coords = True
                continue
            elif line == 'EOF':
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Índice de ciudad, x, y
                        x, y = float(parts[1]), float(parts[2])
                        coordinates.append([x, y])
                    except (ValueError, IndexError):
                        continue
    
    print(f"Se leyeron {len(coordinates)} coordenadas del archivo {file_path}")
    return coordinates

def solve_tsp_ga(cities, population_size=100, generations=1000, visualize=False):
    """
    Resuelve el problema TSP usando el algoritmo genético con medición de tiempo
    """
    start_time = time.time()
    
    ga = TSPGeneticAlgorithm(population_size, generations)
    ga.load_coordinates(cities)
    best_route, best_distance = ga.solve(visualize=visualize)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return best_route, best_distance, execution_time

# Ejemplo de uso
if __name__ == "__main__":
    # Crear directorio para soluciones si no existe
    os.makedirs('pr1/data/solutions', exist_ok=True)
    
    # Lista para guardar resultados
    results = []
    
    # Casos de prueba con archivos TSP
    test_cases = [
        {
            'name': '3case',
            'file_path': 'pr1/data/3case.tsp/3case.tsp',
            'population_size': 75,
            'generations': 400
        },
        {
            'name': 'eil101',
            'file_path': 'pr1/data/eil101.tsp/eil101.tsp',
            'population_size': 100,
            'generations': 500
        },
        {
            'name': 'gr229', 
            'file_path': 'pr1/data/gr229.tsp/gr229.tsp',
            'population_size': 150,
            'generations': 300
        }
    ]
    
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Resolviendo problema: {case['name']}")
        print(f"{'='*50}")
        
        try:
            # Cargar coordenadas del archivo TSP
            cities = parse_tsp_file(case['file_path'])
            num_cities = len(cities)
            
            print(f"Número de ciudades: {num_cities}")
            print(f"Tamaño de población: {case['population_size']}")
            print(f"Número de iteraciones: {case['generations']}")
            
            # Resolver con algoritmo genético
            route, distance, exec_time = solve_tsp_ga(
                cities, 
                population_size=case['population_size'], 
                generations=case['generations'],
                visualize=True
            )
            
            print(f"Tiempo de ejecución: {exec_time:.2f} segundos")
            print(f"Mejor distancia encontrada: {distance:.2f}")
            
            # Guardar resultado
            results.append({
                'problema': case['name'],
                'num_ciudades': num_cities,
                'poblacion_ga': case['population_size'],
                'iteraciones_ga': case['generations'],
                'tiempo_ejecucion': round(exec_time, 2),
                'distancia_suboptima': round(distance, 2)
            })
            
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {case['file_path']}")
        except Exception as e:
            print(f"Error procesando {case['name']}: {str(e)}")
    
    # Guardar resultados en CSV
    if results:
        csv_path = 'pr1/data/solutions/ga_solutions.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['problema', 'num_ciudades', 'poblacion_ga', 'iteraciones_ga', 'tiempo_ejecucion', 'distancia_suboptima']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*50}")
        print(f"Resultados guardados en: {csv_path}")
        print(f"{'='*50}")
        
        # Mostrar tabla de resultados
        df = pd.DataFrame(results)
        print("\nTabla de resultados:")
        print(df.to_string(index=False))
    else:
        print("No se pudieron generar resultados.")

