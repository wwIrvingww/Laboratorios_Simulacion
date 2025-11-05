# src/models/microscopic.py
"""
Modelo microscópico (IDM) - implementación básica de la
función de aceleración y una clase que mantiene el estado
(posiciones/velocidades) de los vehículos.
"""
from typing import Optional, Dict
import numpy as np

# ----- Función IDM -----
def idm_acceleration(v: float, v_lead: float, s: float, params: Dict[str, float]) -> float:
    """
    Calcula la aceleración según el Intelligent Driver Model (IDM).
    Parámetros:
      v       : velocidad del vehículo (m/s)
      v_lead  : velocidad del vehículo líder (m/s)
      s       : separación al vehículo líder (m) -> s = x_lead - x
      params  : diccionario con claves: v0, a, b, T, s0
    Retorna:
      aceleración en m/s^2
    """
    v0 = params.get("v0", 30.0)
    a = params.get("a", 1.2)
    b = params.get("b", 1.5)
    T = params.get("T", 1.5)
    s0 = params.get("s0", 2.0)

    # Evitar división por cero / distancias no físicas
    s_eff = max(s, 0.1)

    # término de distancia segura dinámica (s*)
    s_star = s0 + v * T + (v * (v - v_lead)) / (2.0 * np.sqrt(a * b))

    # aceleración IDM
    acc = a * (1.0 - (v / v0) ** 4 - (s_star / s_eff) ** 2)
    return float(acc)


# ----- Clase que representa el sistema de vehículos -----
class MicroscopicModel:
    """
    Modelo microscópico simple que guarda posiciones y velocidades.
    No contiene el integrador RK4: sólo la dinámica (evaluación de aceleraciones).
    """

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

        # inicializar posiciones (por defecto: uniformes en [0, road_length*0.9] ordenados de adelante hacia atrás)
        if init_positions is None:
            self.positions = np.linspace(0.0, self.road_length * 0.9, self.n_cars)
        else:
            self.positions = np.asarray(init_positions, dtype=float).copy()

        # inicializar velocidades (por defecto: 80% de v0)
        if init_velocities is None:
            v0 = params.get("v0", 30.0)
            self.velocities = np.ones(self.n_cars, dtype=float) * (0.8 * v0)
        else:
            self.velocities = np.asarray(init_velocities, dtype=float).copy()

    def compute_spacing(self) -> np.ndarray:
        """
        Calcula la separación s_i = x_{i+1} - x_i para i=0..n-2.
        Para el último vehículo (lead) devolvemos +inf (no hay líder).
        """
        s = np.empty(self.n_cars, dtype=float)
        for i in range(self.n_cars - 1):
            s[i] = self.positions[i + 1] - self.positions[i]
        s[-1] = np.inf
        return s

    def compute_accelerations(self) -> np.ndarray:
        """
        Evalúa la aceleración IDM para cada vehículo en el estado actual.
        """
        acc = np.zeros(self.n_cars, dtype=float)
        s = self.compute_spacing()

        for i in range(self.n_cars - 1):
            v = float(self.velocities[i])
            v_lead = float(self.velocities[i + 1])
            acc[i] = idm_acceleration(v, v_lead, s[i], self.params)

        # Vehículo líder (último índice): acelera hacia v0 con el término libre del IDM
        v_lead_idx = self.n_cars - 1
        v_lead = float(self.velocities[v_lead_idx])
        v0 = self.params.get("v0", 30.0)
        a = self.params.get("a", 1.2)
        acc[v_lead_idx] = a * (1.0 - (v_lead / v0) ** 4)

        return acc

    def enforce_min_spacing(self, s_min: float, periodic: bool = False, road_length: Optional[float] = None) -> None:
        """
        Enforce a minimum spacing s_min between vehicles (hard-core).
        - Si hay solapamiento, colocamos el vehículo de atrás en pos = pos_delante - s_min.
        - Iteramos de adelante hacia atrás (índices mayores a menores) para propagar la corrección hacia atrás.
        - Para carretera periódica, se respeta el wrap-around usando road_length.
        Mutates self.positions and self.velocities in-place.
        """
        if s_min <= 0:
            return

        n = self.n_cars
        pos = self.positions
        vel = self.velocities

        if periodic:
            if road_length is None:
                raise ValueError("road_length must be provided for periodic enforcement")
            # recorremos desde líder (n-1) hacia atrás (0)
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
            # frontera abierta: corregimos de n-2 -> 0
            for i in range(n - 2, -1, -1):
                j = i + 1
                s = pos[j] - pos[i]
                if s < s_min:
                    pos[i] = pos[j] - s_min
                    vel[i] = min(vel[i], vel[j])

    def step_euler(self, dt: float) -> None:
        """
        Paso explícito simple (Euler) para avanzar el estado un dt.
        Útil para pruebas rápidas; en producción usar RK4.
        """
        acc = self.compute_accelerations()
        self.velocities += acc * dt
        self.velocities = np.maximum(self.velocities, 0.0)  # evitar velocidades negativas
        self.positions += self.velocities * dt

    def get_state(self):
        """Devuelve posiciones y velocidades actuales (copias)."""
        return self.positions.copy(), self.velocities.copy()
