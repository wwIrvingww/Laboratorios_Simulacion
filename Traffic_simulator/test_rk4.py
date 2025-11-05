from src.models.microscopic import MicroscopicModel
from src.solvers.runge_kutta import simulate

params = {"v0": 30.0, "a": 1.2, "b": 1.5, "T": 1.5, "s0": 2.0, "s_min": 2.0}
model = MicroscopicModel(n_cars=8, road_length=800.0, params=params)


dt = 0.1
n_steps = 200
positions_rec, velocities_rec = simulate(model, dt, n_steps, periodic=False, record=True)

print("Estado final posiciones:", positions_rec[-1])
print("Estado final velocidades:", velocities_rec[-1])
