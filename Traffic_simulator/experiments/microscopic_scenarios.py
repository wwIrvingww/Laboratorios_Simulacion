# experiments/microscopic_scenarios.py
"""
Escenarios de simulación para el modelo microscópico (IDM).

Este módulo define escenarios reproducibles para:
  - Flujo estable de tráfico.
  - Flujo perturbado con formación de congestión.

Los resultados se guardan como diagramas espacio-tiempo en la carpeta
results/figures/.
"""

import os
from typing import Dict, Tuple

import numpy as np

from src.models.microscopic import MicroscopicModel
from src.solvers.runge_kutta import simulate
from src.visualization.spacetime_diagrams import plot_spacetime_diagram_micro


def _ensure_output_dir(path: str) -> None:
    """
    Crea el directorio padre del archivo dado si no existe.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def create_default_params() -> Dict[str, float]:
    """
    Parámetros por defecto para el modelo IDM en los escenarios microscópicos.
    """
    return {
        "v0": 30.0,   # velocidad deseada (m/s)
        "a": 1.2,     # aceleración máxima (m/s^2)
        "b": 1.5,     # deceleración cómoda (m/s^2)
        "T": 1.5,     # tiempo de reacción (s)
        "s0": 2.0,    # distancia mínima (m)
        "s_min": 2.0  # separación mínima numérica (hard-core)
    }


def run_stable_flow_scenario(
    n_cars: int = 20,
    road_length: float = 800.0,
    final_time: float = 60.0,
    dt: float = 0.1,
    output_path: str = "results/figures/micro_stable_spacetime.png"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Escenario 1: flujo estable.

    Vehículos distribuidos uniformemente, todos con la misma velocidad inicial
    ligeramente por debajo de la velocidad deseada. Esperamos que el sistema
    converja de forma suave a un flujo casi uniforme, sin formación de atascos.

    Retorna:
        t_array           : vector de tiempos
        positions_record  : posiciones de todos los vehículos en el tiempo
        velocities_record : velocidades de todos los vehículos en el tiempo
    """
    params = create_default_params()

    # Tiempo y pasos
    t_array = np.arange(0.0, final_time + dt, dt)
    n_steps = len(t_array) - 1

    # Posiciones iniciales: distribución uniforme en la carretera
    init_positions = np.linspace(0.0, road_length * 0.9, n_cars)

    # Velocidades iniciales: 80% de v0
    v0 = params["v0"]
    init_velocities = np.ones(n_cars, dtype=float) * (0.8 * v0)

    model = MicroscopicModel(
        n_cars=n_cars,
        road_length=road_length,
        params=params,
        init_positions=init_positions,
        init_velocities=init_velocities,
    )

    # Simulación con RK4
    positions_record, velocities_record = simulate(
        model,
        dt=dt,
        n_steps=n_steps,
        periodic=False,
        road_length=road_length,
        record=True,
    )

    # Guardar diagrama espacio-tiempo
    _ensure_output_dir(output_path)
    plot_spacetime_diagram_micro(
        positions_record,
        t_array,
        filename=output_path,
    )

    return t_array, positions_record, velocities_record


def run_perturbation_scenario(
    n_cars: int = 20,
    road_length: float = 800.0,
    final_time: float = 60.0,
    dt: float = 0.1,
    output_path: str = "results/figures/micro_perturbation_spacetime.png"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Escenario 2: flujo con perturbación.

    Se parte de un flujo casi estable, pero se introduce una perturbación
    en las velocidades iniciales de algunos vehículos (por ejemplo, los primeros
    vehículos más lentos). Esto genera una región de mayor densidad y una onda
    de congestión que se propaga hacia atrás.

    Retorna:
        t_array           : vector de tiempos
        positions_record  : posiciones de todos los vehículos en el tiempo
        velocities_record : velocidades de todos los vehículos en el tiempo
    """
    params = create_default_params()

    # Tiempo y pasos
    t_array = np.arange(0.0, final_time + dt, dt)
    n_steps = len(t_array) - 1

    # Posiciones iniciales uniformes
    init_positions = np.linspace(0.0, road_length * 0.9, n_cars)

    # Velocidades iniciales: mayoría cerca de v0, algunos más lentos
    v0 = params["v0"]
    init_velocities = np.ones(n_cars, dtype=float) * (0.9 * v0)

    # Introducir perturbación: primeros vehículos más lentos
    k_slow = max(3, n_cars // 5)
    init_velocities[:k_slow] = 0.5 * v0

    model = MicroscopicModel(
        n_cars=n_cars,
        road_length=road_length,
        params=params,
        init_positions=init_positions,
        init_velocities=init_velocities,
    )

    positions_record, velocities_record = simulate(
        model,
        dt=dt,
        n_steps=n_steps,
        periodic=False,
        road_length=road_length,
        record=True,
    )

    _ensure_output_dir(output_path)
    plot_spacetime_diagram_micro(
        positions_record,
        t_array,
        filename=output_path,
    )

    return t_array, positions_record, velocities_record


if __name__ == "__main__":
    # Ejecuta ambos escenarios como demostración rápida
    run_stable_flow_scenario()
    run_perturbation_scenario()
