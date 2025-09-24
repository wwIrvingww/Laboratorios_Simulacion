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
    # Reconstruir la ruta a partir de las variables activas
    route = []
    current_city = 0
    visited = {0}
    route.append(current_city)
    
    while len(visited) < len(cities):
        for j in cities:
            if current_city != j and pulp.value(x_vars[(current_city, j)]) == 1:
                route.append(j)
                visited.add(j)
                current_city = j
                break
    # Volvemos al inicio
    route.append(0)
    
    # --- Graficar ---
    xs = [cities[i][0] for i in route]
    ys = [cities[i][1] for i in route]

    plt.figure(figsize=(6,6))
    plt.scatter([cities[i][0] for i in cities], [cities[i][1] for i in cities], c="red", s=50)
    
    for idx, (x, y) in cities.items():
        plt.text(x+0.1, y+0.1, str(idx), fontsize=12)
    
    plt.plot(xs, ys, c="blue", linewidth=1.5, marker="o")
    plt.title("Ruta óptima del TSP")
    plt.show()
    
    return route


# Resolver
prob.solve(pulp.PULP_CBC_CMD(msg=0))

print("Estado:", pulp.LpStatus[prob.status])
print("Distancia mínima:", pulp.value(prob.objective))

# Mostrar ruta y graficar
optimal_route = plot_route(cities, x)
print("Ruta óptima:", optimal_route)
