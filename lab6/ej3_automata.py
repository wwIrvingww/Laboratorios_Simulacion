import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
from ej2 import (
    generate_initial_map,
    make_circular_kernel,
    run_single,
    plot_single,
    SUSCEPTIBLE,
    INFECTED,
    RECOVERED,
)

# -----------------------------------------------------------
# Parámetros (idénticos a la parte 2)
# -----------------------------------------------------------
M, N = 200, 200
I0 = 100
T = 200
dt = 1.0
r = 1.5
beta = 0.6
gamma = 0.1
seed_init = 42
boundary = "periodic"

# -----------------------------------------------------------
# Preparar condiciones iniciales fijas (una sola vez)
# -----------------------------------------------------------
initial_grid = generate_initial_map(M, N, I0, seed_init=seed_init)
kernel = make_circular_kernel(r)
n_steps = int(T)
times = np.arange(0, n_steps + 1) * dt

# -----------------------------------------------------------
# Repetir simulaciones
# -----------------------------------------------------------
Nexp = 10  # número de repeticiones
all_timeseries = []

for exp in range(Nexp):
    rng = Generator(PCG64(100 + exp))  # cambia la semilla solo para el azar dinámico
    timeseries, _ = run_single(
        initial_grid,
        kernel,
        beta,
        gamma,
        dt,
        n_steps,
        rng,
        boundary=boundary,
        record_frames=False,
    )
    all_timeseries.append(timeseries)

# -----------------------------------------------------------
# Promediar resultados
# -----------------------------------------------------------
all_timeseries = np.array(all_timeseries)  # (Nexp, n_steps+1, 3)
mean_timeseries = np.mean(all_timeseries, axis=0)

# -----------------------------------------------------------
# Graficar curvas promedio normalizadas
# -----------------------------------------------------------
Ntot = M * N
S_mean = mean_timeseries[:, 0] / Ntot
I_mean = mean_timeseries[:, 1] / Ntot
R_mean = mean_timeseries[:, 2] / Ntot

plt.figure(figsize=(8,5))
plt.plot(times, S_mean, label="S promedio", color="#1f77b4")
plt.plot(times, I_mean, label="I promedio", color="#d62728")
plt.plot(times, R_mean, label="R promedio", color="#2ca02c")
plt.xlabel("Tiempo")
plt.ylabel("Fracción de población")
plt.title(f"Promedio sobre {Nexp} simulaciones (autómata celular)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sir_mean_ca.png", dpi=200)
plt.show()

print("Simulaciones múltiples completadas. Curvas promedio guardadas en sir_mean_ca.png")
