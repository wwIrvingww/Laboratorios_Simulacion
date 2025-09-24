import math
import pandas as pd
import pulp
import matplotlib.pyplot as plt


"""
Estructura:
1. Calcular distancias
2. Definición de variables de decisión: recordar son variables binarias, (sí o no)
   Además, acá es por pares por ejemplo Xij, esto representa si viaja de i a j.
3. Función Objetivo: Es la parte más importante ya que acá es donde minimiza la distancia total
   si Xij es 1, entonces significa que sí la vamos a añadir al recorrido.
4. Reestriciones: 
    a) Cada ciudad debe tener una salida. No se puede salir dos veces de la misma ciudad, ni quedarse atrapado.
    b) Cada ciudad debe tener SOLO una entrada (no se puede visitar dos veces)
    c) No se puede viajar de una ciudad a sí misma
5. Evitar subciclos: Obligatoriamente tiene que ser una ruta que pase por todas las ciudades.
6. Resolver
"""
#---cALCULAR DISTANCIAS--#

cities = {
    0: (0, 0),
    1: (0, 2),
    2: (3, 0),
    3: (3, 3)
}

# Diccionario para guardar distancias
distances = {}


# Número de ciudades
n = len(cities)

# Crear matriz de distancias vacía
matrix = [[0.0 for _ in range(n)] for _ in range(n)]

# Llenar matriz y diccionario con distancias
for i, (x1, y1) in cities.items():
    for j, (x2, y2) in cities.items():
        if i != j:
            d = math.dist((x1, y1), (x2, y2))
            matrix[i][j] = d
            distances[(i, j)] = d   # <<--- AGREGAR ESTA LÍNEA


# Mostrar como DataFrame bonito
df = pd.DataFrame(matrix, index=cities.keys(), columns=cities.keys())
print(df)


#---VARIABLES DE DECISIÓN---#

n = len(cities)

#---Definir variables binarias x_ij---#
x = pulp.LpVariable.dicts(
    "x",
    ((i, j) for i in range(n) for j in range(n) if i != j),
    cat="Binary"
)

#---FUNCION OBJETIVO---#
# Crear el problema de optimización
prob = pulp.LpProblem("TSP", pulp.LpMinimize)

# Función objetivo: minimizar la suma de distancias
prob += pulp.lpSum(
    distances[(i, j)] * x[(i, j)]
    for i in range(n) for j in range(n) if i != j
)


# --- Restricciones ---#
# Una salida por ciudad
for i in range(n):
    prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1

# Una entrada por ciudad
for j in range(n):
    prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1


# Variables auxiliares para MTZ
u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n-1, cat="Integer")

# Restricciones MTZ
for i in range(1, n):      # desde ciudad 1 en adelante
    for j in range(1, n):  # desde ciudad 1 en adelante
        if i != j:
            prob += u[i] - u[j] + (n-1) * x[(i, j)] <= n - 2



# --- Imprimir modelo ---#
print(prob)

# --- Resolver ---#
prob.solve(pulp.PULP_CBC_CMD(msg=1))  # msg=1 para ver el log del solver

print("\nEstado de la solución:", pulp.LpStatus[prob.status])
print("Distancia mínima:", pulp.value(prob.objective))

# Mostrar qué variables x_ij se activaron
print("\nArcos seleccionados en la ruta óptima:")
for (i, j) in x:
    if pulp.value(x[(i, j)]) == 1:
        print(f"De {i} a {j}")



def plot_route(cities, x_vars):
    # --- Verificar que hay solución válida ---
    arcs = [(i, j) for (i, j) in x_vars if pulp.value(x_vars[(i, j)]) == 1]
    if not arcs:
        print("⚠️ No hay arcos activos en la solución. Revisa las restricciones o el estado del solver.")
        return []

    # --- Reconstruir ruta ---
    route = []
    current_city = 0
    visited = {0}
    route.append(current_city)

    # límite de pasos = número de ciudades + 1
    max_steps = len(cities) + 1
    steps = 0

    while len(visited) < len(cities) and steps < max_steps:
        found_next = False
        for j in cities:
            if current_city != j and pulp.value(x_vars[(current_city, j)]) == 1:
                route.append(j)
                visited.add(j)
                current_city = j
                found_next = True
                break
        if not found_next:
            print(f"⚠️ No se encontró salida desde la ciudad {current_city}. La solución puede ser inválida.")
            break
        steps += 1

    # cerrar ciclo
    if route[0] != route[-1]:
        route.append(0)

    # --- Si excede pasos, avisar ---
    if steps >= max_steps:
        print("⚠️ Se alcanzó el límite de pasos, posible subciclo o ruta inválida.")
        return route

    # --- Graficar ---
    import matplotlib.pyplot as plt
    xs = [cities[i][0] for i in route]
    ys = [cities[i][1] for i in route]

    plt.figure(figsize=(6,6))
    plt.scatter([cities[i][0] for i in cities], [cities[i][1] for i in cities], c="red", s=50)
    for idx, (x, y) in cities.items():
        plt.text(x+0.1, y+0.1, str(idx), fontsize=8)
    plt.plot(xs, ys, c="blue", linewidth=1.2, marker="o")
    plt.title("Ruta óptima del TSP")
    plt.show()

    return route



# Resolver
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# resolver con un límite de tiempo razonable (ej. 120s)
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=120)  # msg=1 para ver log
prob.solve(solver)

print("Estado del solver:", pulp.LpStatus[prob.status])

# contar arcos activos
active_arcs = [(i,j) for (i,j) in x if pulp.value(x[(i,j)]) >= 0.5]
print("Arcos activos (ejemplo primeros 20):", active_arcs[:20])
print("Número total de arcos activos:", len(active_arcs))

# comprobar salidas por ciudad
out_counts = {i: sum(1 for j in range(n) if pulp.value(x[(i,j)]) >= 0.5) for i in range(n)}
in_counts  = {j: sum(1 for i in range(n) if pulp.value(x[(i,j)]) >= 0.5) for j in range(n)}
print("Máx salidas por ciudad:", max(out_counts.values()), "  Máx entradas por ciudad:", max(in_counts.values()))
print("Min salidas:", min(out_counts.values()), "  Min entradas:", min(in_counts.values()))


print("Estado:", pulp.LpStatus[prob.status])
print("Distancia mínima:", pulp.value(prob.objective))

# Mostrar ruta y graficar
optimal_route = plot_route(cities, x)
print("Ruta óptima:", optimal_route)

## ADAPTADO A LOS 3 PROBLEMAS ##
## ADAPTADO A LOS 2 PROBLEMAS DE TSPLIB ##
def parse_tsp_file(file_path):
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
                    x, y = float(parts[1]), float(parts[2])
                    coordinates.append((x, y))
    return coordinates


import os
import math
import pulp
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))

scenarios = [
    {"name": "eil101", "file": os.path.join(base_path, "data", "eil101.tsp", "eil101.tsp")},
    {"name": "gr229", "file": os.path.join(base_path, "data", "gr229.tsp", "gr229.tsp")},
    {"name": "3case",  "file": os.path.join(base_path, "data", "3case.tsp", "3case.tsp")},
]

for case in scenarios:
    print(f"\nResolviendo escenario: {case['name']}")
    # --- leer coords ---
    coords = parse_tsp_file(case["file"])
    cities = {i: coords[i] for i in range(len(coords))}
    n = len(cities)
    print("Número de ciudades:", n)

    # --- distancias ---
    distances = {}
    for i, (x1, y1) in cities.items():
        for j, (x2, y2) in cities.items():
            if i != j:
                distances[(i,j)] = math.dist((x1,y1),(x2,y2))

    # --- variables ---
    x = pulp.LpVariable.dicts("x", ((i,j) for i in range(n) for j in range(n) if i!=j), cat="Binary")
    u = pulp.LpVariable.dicts("u", list(range(n)), lowBound=0, upBound=n-1, cat="Integer")

    # --- problema ---
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)
    prob += pulp.lpSum(distances[(i,j)] * x[(i,j)] for i in range(n) for j in range(n) if i!=j)

    # restric. salida/entrada
    for i in range(n):
        prob += pulp.lpSum(x[(i,j)] for j in range(n) if i!=j) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i,j)] for i in range(n) if i!=j) == 1

    # MTZ (i,j desde 1..n-1)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n-1) * x[(i,j)] <= n-2

    # --- resolver (pone un timeLimit razonable) ---
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=120)  # ajustar tiempo según recurso
    prob.solve(solver)

    print("Estado:", pulp.LpStatus[prob.status])
    # revisar arcos activos y validez
    active_arcs = [(i,j) for (i,j) in x if pulp.value(x[(i,j)]) is not None and pulp.value(x[(i,j)]) > 0.5]
    print("Arcos activos:", len(active_arcs))

    # Validar que haya exactamente 1 salida por ciudad
    next_map = {}
    for (i,j) in x:
        val = pulp.value(x[(i,j)])
        if val is not None and val > 0.5:
            next_map[i] = j

    if len(next_map) != n:
        print("La solución NO tiene un arco de salida por cada ciudad. Saltando visualización.")
        continue

    # Reconstruir tour (simple)
    route = []
    visited = set()
    cur = 0
    for _ in range(n):
        route.append(cur)
        visited.add(cur)
        cur = next_map[cur]
    route.append(route[0])

    # opcional: verificar ciclo único
    if len(set(route[:-1])) != n:
        print("Advertencia: el ciclo construido no visita todas las ciudades exactamente una vez.")
        continue

    # --- visualizar ---
    xs = [cities[i][0] for i in route]
    ys = [cities[i][1] for i in route]
    plt.figure(figsize=(8,8))
    plt.scatter([cities[i][0] for i in cities], [cities[i][1] for i in cities], c='red', s=10)
    plt.plot(xs, ys, '-o')
    plt.title(f"Ruta TSP {case['name']} (n={n})")
    plt.show()

    print("Ruta óptima aproximada/actual:", route)
    print("Distancia objetivo:", pulp.value(prob.objective))
