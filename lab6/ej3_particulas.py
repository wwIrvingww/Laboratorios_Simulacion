import numpy as np
import matplotlib.pyplot as plt
from ej1 import Particle, run_simulation, L, Ntotal, I0, vmax, r, beta, gamma, dt, T

# -----------------------------------------------------------
# Generar las condiciones iniciales (una sola vez, misma semilla)
# -----------------------------------------------------------
np.random.seed(42)

initial_particles = []
for i in range(Ntotal):
    x = np.random.uniform(0, L)
    y = np.random.uniform(0, L)
    angle = np.random.uniform(0, 2*np.pi)
    speed = np.random.uniform(0, vmax)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)
    state = 1 if i < I0 else 0
    initial_particles.append(Particle(x, y, vx, vy, state))

# -----------------------------------------------------------
# Repetir simulaciones
# -----------------------------------------------------------
Nexp = 10  # número de repeticiones
all_S, all_I, all_R = [], [], []

for exp in range(Nexp):
    S_hist, I_hist, R_hist, time_hist = run_simulation(initial_particles, L, r, beta, gamma, dt, T)
    all_S.append(S_hist)
    all_I.append(I_hist)
    all_R.append(R_hist)

# -----------------------------------------------------------
# Promediar resultados
# -----------------------------------------------------------
S_mean = np.mean(all_S, axis=0)
I_mean = np.mean(all_I, axis=0)
R_mean = np.mean(all_R, axis=0)

# -----------------------------------------------------------
# Graficar curvas promedio
# -----------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(time_hist, S_mean, 'b', label='S promedio')
plt.plot(time_hist, I_mean, 'r', label='I promedio')
plt.plot(time_hist, R_mean, 'g', label='R promedio')
plt.xlabel('Tiempo')
plt.ylabel('Número de individuos')
plt.title(f'Promedio sobre {Nexp} simulaciones (modelo de partículas)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sir_mean_particles.png', dpi=200)
plt.show()
