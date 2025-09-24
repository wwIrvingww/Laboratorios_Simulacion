# pl_run_final.py
import os
import math
import time
import csv
import pulp
import matplotlib.pyplot as plt

# ---------- helpers ----------
def parse_tsp_file(file_path):
    coords = []
    reading = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if l == 'NODE_COORD_SECTION':
                reading = True
                continue
            if l == 'EOF':
                break
            if reading and l:
                parts = l.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return coords

def nearest_neighbor_tour(cities, start=0):
    n = len(cities)
    unvisited = set(range(n))
    cur = start
    tour = [cur]
    unvisited.remove(cur)
    while unvisited:
        nxt = min(unvisited, key=lambda j: math.dist(cities[cur], cities[j]))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour.append(tour[0])
    return tour

def tour_distance(tour, cities):
    return sum(math.dist(cities[tour[i]], cities[tour[i+1]]) for i in range(len(tour)-1))

def plot_and_save(cities, route, out_path, title=None, show_seconds=3):
    xs = [cities[i][0] for i in route]
    ys = [cities[i][1] for i in route]
    plt.figure(figsize=(8,8))
    plt.scatter([cities[i][0] for i in cities], [cities[i][1] for i in cities], c='red', s=10)
    for idx,(x,y) in cities.items():
        plt.text(x+0.5, y+0.5, str(idx), fontsize=6)
    plt.plot(xs, ys, '-o')
    if title:
        plt.title(title)
    # show non-blocking for a few seconds, save, then close
    plt.show(block=False)
    plt.pause(show_seconds)
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------- solver / instance ----------
def solve_tsp_instance(cities, time_limit=1200, verbose=False):
    n = len(cities)
    # build distances
    distances = {}
    for i,(x1,y1) in cities.items():
        for j,(x2,y2) in cities.items():
            if i!=j:
                distances[(i,j)] = math.dist((x1,y1),(x2,y2))

    # variables
    x = pulp.LpVariable.dicts("x", ((i,j) for i in range(n) for j in range(n) if i!=j), cat='Binary')
    u = pulp.LpVariable.dicts("u", list(range(n)), lowBound=0, upBound=n-1, cat='Integer')

    # problem
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)
    prob += pulp.lpSum(distances[(i,j)] * x[(i,j)] for i in range(n) for j in range(n) if i!=j)

    # degree constraints
    for i in range(n):
        prob += pulp.lpSum(x[(i,j)] for j in range(n) if i!=j) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i,j)] for i in range(n) if i!=j) == 1

    # MTZ subtour elimination
    if n > 1:
        for i in range(1,n):
            for j in range(1,n):
                if i!=j:
                    prob += u[i] - u[j] + (n-1)*x[(i,j)] <= n-2

    # solve (CBC) with time limit
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit)
    t0 = time.time()
    prob.solve(solver)
    elapsed = time.time() - t0

    status = pulp.LpStatus[prob.status]
    obj = pulp.value(prob.objective)

    active_arcs = [(i,j) for (i,j) in x.keys() if pulp.value(x[(i,j)]) is not None and pulp.value(x[(i,j)]) > 0.5]
    out_counts = {i: sum(1 for (a,b) in active_arcs if a==i) for i in range(n)}
    in_counts  = {j: sum(1 for (a,b) in active_arcs if b==j) for j in range(n)}
    valid_degrees = (len(active_arcs) == n) and all(out_counts[i]==1 for i in range(n)) and all(in_counts[j]==1 for j in range(n))

    result = {
        "status": status,
        "objective": obj,
        "solve_time": elapsed,
        "active_arcs": active_arcs,
        "active_arcs_count": len(active_arcs),
        "out_counts": out_counts,
        "in_counts": in_counts,
        "valid_degrees": valid_degrees,
        "route": None,
        "route_source": None
    }

    if valid_degrees:
        next_map = {a:b for (a,b) in active_arcs}
        route = []
        cur = 0
        for _ in range(n):
            route.append(cur)
            cur = next_map.get(cur, None)
            if cur is None:
                break
        if len(route)==n and next_map.get(route[-1], None)==route[0]:
            route.append(route[0])
            result["route"] = route
            result["route_source"] = "ILP"
        else:
            result["valid_degrees"] = False

    if not result["valid_degrees"]:
        greedy = nearest_neighbor_tour(cities, start=0)
        result["route"] = greedy
        result["route_source"] = "GREEDY"
        result["greedy_distance"] = tour_distance(greedy, cities)

    return result

# ---------- main ----------
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))

    scenarios = [
        {"name": "eil101", "file": os.path.join(base_path, "data", "eil101.tsp", "eil101.tsp")},
        {"name": "gr229", "file": os.path.join(base_path, "data", "gr229.tsp", "gr229.tsp")},
        {"name": "3case",  "file": os.path.join(base_path, "data", "3case.tsp", "3case.tsp")},
    ]

    # time limit (20 minutes)
    time_limit_seconds = 4200

    # output dirs
    out_dir = os.path.join(base_path, "data", "solutions")
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)

    # CSV files
    csv_ga_format = os.path.join(out_dir, "ilp_solutions_ga_format.csv")
    csv_detailed = os.path.join(out_dir, "ilp_solutions_detailed.csv")

    # create/append headers if files don't exist
    if not os.path.exists(csv_ga_format):
        with open(csv_ga_format, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['problema','num_ciudades','poblacion_ga','iteraciones_ga','tiempo_ejecucion','distancia_suboptima'])

    if not os.path.exists(csv_detailed):
        with open(csv_detailed, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['problema','num_ciudades','method','ilp_status','time_s','ilp_obj','greedy_dist','active_arcs_count','png_path'])

    # run scenarios sequentially
    for case in scenarios:
        name = case["name"]
        file_path = case["file"]
        print("\n--- Escenario:", name, "---")
        if not os.path.exists(file_path):
            print("Archivo no encontrado:", file_path)
            continue

        coords = parse_tsp_file(file_path)
        cities = {i: coords[i] for i in range(len(coords))}
        n = len(cities)
        print("Número de ciudades:", n)

        res = solve_tsp_instance(cities, time_limit=time_limit_seconds, verbose=True)

        # decide png path and save
        method = res["route_source"]
        png_name = f"{name}_{method}.png"
        png_path = os.path.join(png_dir, png_name)
        # plot and save (show 3 seconds)
        try:
            plot_and_save(cities, res["route"], png_path, title=f"{name} ({method})", show_seconds=3)
        except Exception as e:
            print("Error al graficar/guardar PNG:", e)

        # write GA-compatible CSV row (poblacion_ga, iteraciones_ga = 0 for ILP)
        ga_row = [
            name,
            n,
            0,                 # poblacion_ga (no aplica)
            0,                 # iteraciones_ga (no aplica)
            round(res["solve_time"], 2),
            round(res["objective"] if res["route_source"]=="ILP" else res.get("greedy_distance", 0), 4)
        ]
        with open(csv_ga_format, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(ga_row)

        # write detailed CSV
        detailed_row = [
            name,
            n,
            method,
            res["status"],
            round(res["solve_time"],2),
            round(res["objective"],4) if res["objective"] is not None else "",
            round(res.get("greedy_distance", 0),4) if res.get("greedy_distance") else "",
            res["active_arcs_count"],
            png_path
        ]
        with open(csv_detailed, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(detailed_row)

        # print summary to console
        print("Estado (solver):", res["status"])
        print("Metodo usado:", method)
        print("Tiempo (s):", round(res["solve_time"],2))
        if method == "ILP":
            print("Distancia ILP:", round(res["objective"],4))
        else:
            print("Distancia GREEDY:", round(res["greedy_distance"],4))
        print("PNG guardado en:", png_path)

    print("\n--- FIN: resultados guardados en:", out_dir, "---")
