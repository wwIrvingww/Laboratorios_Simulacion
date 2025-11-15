# src/solvers/runge_kutta.py
"""
Integrador Runge-Kutta 4 (RK4) para el modelo microscopico.
"""
from typing import Tuple, Optional
import numpy as np
from src.models.microscopic import idm_acceleration


def _compute_accelerations(positions: np.ndarray,
                           velocities: np.ndarray,
                           params: dict,
                           periodic: bool = False,
                           road_length: Optional[float] = None) -> np.ndarray:
    n = positions.shape[0]
    acc = np.zeros(n, dtype=float)

    for i in range(n):
        if periodic:
            j = (i + 1) % n
            s = positions[j] - positions[i]
            if s <= 0:
                s += road_length
            v_lead = velocities[j]
            acc[i] = idm_acceleration(float(velocities[i]), float(v_lead), float(s), params)
        else:
            if i < n - 1:
                j = i + 1
                s = positions[j] - positions[i]
                if s <= 0:
                    s = 1e-3
                v_lead = velocities[j]
                acc[i] = idm_acceleration(float(velocities[i]), float(v_lead), float(s), params)
            else:
                v = float(velocities[i])
                v0 = params.get("v0", 30.0)
                a = params.get("a", 1.2)
                acc[i] = a * (1.0 - (v / v0) ** 4)

    return acc


def rk4_step(model, dt: float, periodic: bool = False, road_length: Optional[float] = None) -> None:
    """
    Avanza el estado del `model` un paso dt usando RK4.
    Modifica in-place model.positions y model.velocities.
    """
    pos0, vel0 = model.get_state()
    params = model.params

    # k1
    dpos1 = vel0
    dvel1 = _compute_accelerations(pos0, vel0, params, periodic=periodic, road_length=road_length)

    # k2
    pos_k2 = pos0 + 0.5 * dt * dpos1
    vel_k2 = vel0 + 0.5 * dt * dvel1
    dpos2 = vel_k2
    dvel2 = _compute_accelerations(pos_k2, vel_k2, params, periodic=periodic, road_length=road_length)

    # k3
    pos_k3 = pos0 + 0.5 * dt * dpos2
    vel_k3 = vel0 + 0.5 * dt * dvel2
    dpos3 = vel_k3
    dvel3 = _compute_accelerations(pos_k3, vel_k3, params, periodic=periodic, road_length=road_length)

    # k4
    pos_k4 = pos0 + dt * dpos3
    vel_k4 = vel0 + dt * dvel3
    dpos4 = vel_k4
    dvel4 = _compute_accelerations(pos_k4, vel_k4, params, periodic=periodic, road_length=road_length)

    # combinar incrementos (formula clasica RK4)
    pos_new = pos0 + (dt / 6.0) * (dpos1 + 2.0 * dpos2 + 2.0 * dpos3 + dpos4)
    vel_new = vel0 + (dt / 6.0) * (dvel1 + 2.0 * dvel2 + 2.0 * dvel3 + dvel4)

    # evitar velocidades negativas
    vel_new = np.maximum(vel_new, 0.0)

    # aplicar estado nuevo al modelo (mutacion)
    model.positions = pos_new
    model.velocities = vel_new

    # --- aqui esta el paso 3: aplicar correccion "hard-core" para evitar solapamientos ---
    # elegir s_min: si el usuario lo especifico en params lo usamos; si no, usamos s0 (o una fraccion)
    s_min = model.params.get("s_min", model.params.get("s0", 2.0))
    try:
        # llama al metodo que implementamos en MicroscopicModel
        model.enforce_min_spacing(s_min, periodic=periodic, road_length=road_length)
    except AttributeError:
        # si el modelo no tiene enforce_min_spacing no hacemos nada (compatibilidad hacia atras)
        pass
    # -------------------------------------------------------------------------------


def simulate(model,
             dt: float,
             n_steps: int,
             periodic: bool = False,
             road_length: Optional[float] = None,
             record: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Corre la simulacion n_steps pasos con rk4_step.
    """
    if record:
        positions_record = np.zeros((n_steps + 1, model.n_cars), dtype=float)
        velocities_record = np.zeros((n_steps + 1, model.n_cars), dtype=float)
        p0, v0 = model.get_state()
        positions_record[0, :] = p0
        velocities_record[0, :] = v0

    for step in range(1, n_steps + 1):
        rk4_step(model, dt, periodic=periodic, road_length=road_length)
        if record:
            p, v = model.get_state()
            positions_record[step, :] = p
            velocities_record[step, :] = v

    if record:
        return positions_record, velocities_record
    return None
