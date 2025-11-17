"""
Modelo Microscópico - Intelligent Driver Model (IDM)
"""
from typing import Optional, Dict
import numpy as np


def idm_acceleration(v: float, v_lead: float, s: float, params: Dict[str, float]) -> float:
    v0 = params.get("v0", 30.0)
    a = params.get("a", 1.2)
    b = params.get("b", 1.5)
    T = params.get("T", 1.5)
    s0 = params.get("s0", 2.0)

    s_eff = max(s, 0.1)
    s_star = s0 + v * T + (v * (v - v_lead)) / (2.0 * np.sqrt(a * b))
    acc = a * (1.0 - (v / v0) ** 4 - (s_star / s_eff) ** 2)
    return float(acc)



class MicroscopicModel:
    def __init__(
        self,
        n_cars: int,
        road_length: float,
        params: Dict[str, float],
        init_positions: Optional[np.ndarray] = None,
        init_velocities: Optional[np.ndarray] = None,
    ):
        self.n_cars = int(n_cars)
        self.road_length = float(road_length)
        self.params = params

        if init_positions is None:
            self.positions = np.linspace(0.0, self.road_length * 0.9, self.n_cars)
        else:
            self.positions = np.asarray(init_positions, dtype=float).copy()

        if init_velocities is None:
            v0 = params.get("v0", 30.0)
            self.velocities = np.ones(self.n_cars, dtype=float) * (0.8 * v0)
        else:
            self.velocities = np.asarray(init_velocities, dtype=float).copy()

    def compute_spacing(self) -> np.ndarray:
        s = np.empty(self.n_cars, dtype=float)
        for i in range(self.n_cars - 1):
            s[i] = self.positions[i + 1] - self.positions[i]
        s[-1] = np.inf
        return s

    def compute_accelerations(self) -> np.ndarray:
        acc = np.zeros(self.n_cars, dtype=float)
        s = self.compute_spacing()

        for i in range(self.n_cars - 1):
            v = float(self.velocities[i])
            v_lead = float(self.velocities[i + 1])
            acc[i] = idm_acceleration(v, v_lead, s[i], self.params)

        v_lead_idx = self.n_cars - 1
        v_lead = float(self.velocities[v_lead_idx])
        v0 = self.params.get("v0", 30.0)
        a = self.params.get("a", 1.2)
        acc[v_lead_idx] = a * (1.0 - (v_lead / v0) ** 4)

        return acc

    def enforce_min_spacing(self, s_min: float, periodic: bool = False, road_length: Optional[float] = None) -> None:
        if s_min <= 0:
            return

        n = self.n_cars
        pos = self.positions
        vel = self.velocities

        if periodic:
            if road_length is None:
                raise ValueError("road_length must be provided for periodic enforcement")
            for i in range(n - 1, -1, -1):
                j = (i + 1) % n
                s = pos[j] - pos[i]
                if s <= 0:
                    s += road_length
                if s < s_min:
                    new_pos_i = pos[j] - s_min
                    new_pos_i = new_pos_i % road_length
                    pos[i] = new_pos_i
                    vel[i] = min(vel[i], vel[j])
        else:
            for i in range(n - 2, -1, -1):
                j = i + 1
                s = pos[j] - pos[i]
                if s < s_min:
                    pos[i] = pos[j] - s_min
                    vel[i] = min(vel[i], vel[j])

    def step_euler(self, dt: float) -> None:
        acc = self.compute_accelerations()
        self.velocities += acc * dt
        self.velocities = np.maximum(self.velocities, 0.0)
        self.positions += self.velocities * dt

    def get_state(self):
        return self.positions.copy(), self.velocities.copy()
