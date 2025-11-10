"""
Modelo Macroscópico de Tráfico Vehicular.

Implementa el modelo de flujo continuo basado en la ecuación de conservación
y el modelo de Greenshields para la relación velocidad-densidad.

Ecuación de conservación: ∂ρ/∂t + ∂q/∂x = 0
Modelo de Greenshields: v(ρ) = V_max * (1 - ρ/ρ_max)
Flujo: q(ρ) = ρ * v(ρ)
"""

import numpy as np
from ..utils.parameters import V_MAX, RHO_MAX, get_velocity, get_flux
from ..solvers.lax_friedrichs import lax_friedrichs_solve, adaptive_lax_friedrichs_solve


def greenshields_flux(rho):
    """
    Calcula el flujo vehicular según el modelo de Greenshields.
    
    q(ρ) = ρ * V_max * (1 - ρ/ρ_max)
    
    Parámetros:
        rho (np.ndarray): Densidad vehicular (veh/km)
    
    Retorna:
        np.ndarray: Flujo vehicular (veh/h)
    """
    return get_flux(rho)


def greenshields_velocity(rho):
    """
    Calcula la velocidad según el modelo de Greenshields.
    
    v(ρ) = V_max * (1 - ρ/ρ_max)
    
    Parámetros:
        rho (np.ndarray): Densidad vehicular (veh/km)
    
    Retorna:
        np.ndarray: Velocidad (km/h)
    """
    return get_velocity(rho)


def simulate_traffic_flow(rho0, x, t, boundary='periodic', adaptive=False, cfl_target=0.8):
    """
    Simula el flujo de tráfico macroscópico usando el modelo de Greenshields.
    
    Resuelve la ecuación de conservación:
        ∂ρ/∂t + ∂q/∂x = 0
    
    con q(ρ) = ρ * V_max * (1 - ρ/ρ_max)
    
    Parámetros:
        rho0 (np.ndarray): Condición inicial de densidad (veh/km)
        x (np.ndarray): Malla espacial (km)
        t (np.ndarray or float): Malla temporal (h) o tiempo final si adaptive=True
        boundary (str): Tipo de condición de frontera ('periodic' o 'outflow')
        adaptive (bool): Si True, usa paso temporal adaptativo
        cfl_target (float): Número CFL objetivo si adaptive=True
    
    Retorna:
        dict: Diccionario con:
            - 'rho': Densidad ρ(x,t) con shape (n_time, n_space)
            - 'flux': Flujo q(x,t) con shape (n_time, n_space)
            - 'velocity': Velocidad v(x,t) con shape (n_time, n_space)
            - 'x': Malla espacial
            - 't': Malla temporal
    """
    if adaptive:
        rho, t_array = adaptive_lax_friedrichs_solve(
            rho0, greenshields_flux, x, t_final=t, 
            cfl_target=cfl_target, boundary=boundary
        )
    else:
        t_array = t
        rho = lax_friedrichs_solve(rho0, greenshields_flux, x, t, boundary=boundary)
    
    # Calcular flujo y velocidad para cada punto espacio-temporal
    flux = np.zeros_like(rho)
    velocity = np.zeros_like(rho)
    
    for n in range(len(t_array)):
        flux[n, :] = greenshields_flux(rho[n, :])
        velocity[n, :] = greenshields_velocity(rho[n, :])
    
    return {
        'rho': rho,
        'flux': flux,
        'velocity': velocity,
        'x': x,
        't': t_array
    }


def compute_fundamental_diagram(rho_range=None):
    """
    Calcula el diagrama fundamental de tráfico (relación flujo-densidad).
    
    Retorna los valores teóricos según el modelo de Greenshields.
    
    Parámetros:
        rho_range (np.ndarray, opcional): Rango de densidades a evaluar.
            Si None, usa np.linspace(0, RHO_MAX, 200)
    
    Retorna:
        dict: Diccionario con:
            - 'rho': Densidades evaluadas (veh/km)
            - 'flux': Flujos correspondientes (veh/h)
            - 'velocity': Velocidades correspondientes (km/h)
            - 'rho_critical': Densidad crítica (máximo flujo)
            - 'flux_max': Flujo máximo (capacidad)
    """
    if rho_range is None:
        rho_range = np.linspace(0, RHO_MAX, 200)
    
    flux = greenshields_flux(rho_range)
    velocity = greenshields_velocity(rho_range)
    
    # Densidad crítica (donde el flujo es máximo): ρ_c = ρ_max / 2
    rho_critical = RHO_MAX / 2.0
    flux_max = greenshields_flux(rho_critical)
    
    return {
        'rho': rho_range,
        'flux': flux,
        'velocity': velocity,
        'rho_critical': rho_critical,
        'flux_max': flux_max
    }


def compute_wave_speeds(rho):
    """
    Calcula las velocidades características (de onda) en cada punto.
    
    Para el modelo de Greenshields:
        c(ρ) = dq/dρ = V_max * (1 - 2*ρ/ρ_max)
    
    La velocidad de onda indica cómo se propagan las perturbaciones.
    
    Parámetros:
        rho (np.ndarray): Densidad vehicular (veh/km)
    
    Retorna:
        np.ndarray: Velocidades características (km/h)
    """
    return V_MAX * (1 - 2 * rho / RHO_MAX)


def detect_shock_waves(rho, x, t, threshold_gradient=50.0):
    """
    Detecta ondas de choque en la solución.
    
    Una onda de choque es una discontinuidad que se propaga en el flujo,
    caracterizada por gradientes espaciales grandes de densidad.
    
    Parámetros:
        rho (np.ndarray): Densidad ρ(x,t) con shape (n_time, n_space)
        x (np.ndarray): Malla espacial (km)
        t (np.ndarray): Malla temporal (h)
        threshold_gradient (float): Umbral de gradiente para detectar shock (veh/km²)
    
    Retorna:
        dict: Diccionario con:
            - 'gradient': Gradiente espacial ∂ρ/∂x
            - 'shock_mask': Máscara booleana donde se detectan shocks
            - 'shock_positions': Lista de posiciones de shocks en cada tiempo
    """
    dx = x[1] - x[0]
    
    # Calcular gradiente espacial
    gradient = np.gradient(rho, dx, axis=1)
    
    # Detectar shocks (gradientes grandes)
    shock_mask = np.abs(gradient) > threshold_gradient
    
    # Encontrar posiciones de shocks para cada tiempo
    shock_positions = []
    for n in range(len(t)):
        shock_indices = np.where(shock_mask[n, :])[0]
        positions = x[shock_indices] if len(shock_indices) > 0 else []
        shock_positions.append(positions)
    
    return {
        'gradient': gradient,
        'shock_mask': shock_mask,
        'shock_positions': shock_positions
    }


def compute_travel_time(rho, x, t):
    """
    Calcula el tiempo de viaje promedio para atravesar el dominio completo.
    
    Integra el tiempo necesario para recorrer cada elemento dx a la velocidad local.
    
    Parámetros:
        rho (np.ndarray): Densidad ρ(x,t) con shape (n_time, n_space)
        x (np.ndarray): Malla espacial (km)
        t (np.ndarray): Malla temporal (h)
    
    Retorna:
        np.ndarray: Tiempo de viaje en cada instante temporal (h)
    """
    velocity = greenshields_velocity(rho)
    dx = x[1] - x[0]
    
    # Evitar división por cero
    velocity_safe = np.maximum(velocity, 1e-6)
    
    # Tiempo de viaje: integral de dx/v(x,t)
    travel_time = np.sum(dx / velocity_safe, axis=1)
    
    return travel_time


def compute_congestion_level(rho, threshold=75.0):
    """
    Calcula el nivel de congestión en función de un umbral de densidad.
    
    Parámetros:
        rho (np.ndarray): Densidad ρ(x,t) con shape (n_time, n_space)
        threshold (float): Umbral de densidad para considerar congestión (veh/km)
    
    Retorna:
        dict: Diccionario con:
            - 'congestion_fraction': Fracción del dominio congestionado en cada tiempo
            - 'congestion_mask': Máscara booleana de zonas congestionadas
    """
    congestion_mask = rho > threshold
    congestion_fraction = np.mean(congestion_mask, axis=1)
    
    return {
        'congestion_fraction': congestion_fraction,
        'congestion_mask': congestion_mask
    }


def compute_total_vehicles(rho, x):
    """
    Calcula el número total de vehículos en el dominio.
    
    Integra la densidad sobre el espacio: N(t) = ∫ρ(x,t)dx
    
    Parámetros:
        rho (np.ndarray): Densidad ρ(x,t) con shape (n_time, n_space)
        x (np.ndarray): Malla espacial (km)
    
    Retorna:
        np.ndarray: Número total de vehículos en cada instante (vehículos)
    """
    dx = x[1] - x[0]
    total_vehicles = np.sum(rho, axis=1) * dx
    return total_vehicles


def compute_average_density(rho, x):
    """
    Calcula la densidad promedio en el dominio.
    
    ρ_avg(t) = (1/L) * ∫ρ(x,t)dx
    
    Parámetros:
        rho (np.ndarray): Densidad ρ(x,t) con shape (n_time, n_space)
        x (np.ndarray): Malla espacial (km)
    
    Retorna:
        np.ndarray: Densidad promedio en cada instante (veh/km)
    """
    L = x[-1] - x[0]
    dx = x[1] - x[0]
    avg_density = np.sum(rho, axis=1) * dx / L
    return avg_density


def compute_average_velocity(rho):
    """
    Calcula la velocidad promedio en el dominio.
    
    Promedio espacial de v(ρ(x,t)) en cada tiempo.
    
    Parámetros:
        rho (np.ndarray): Densidad ρ(x,t) con shape (n_time, n_space)
    
    Retorna:
        np.ndarray: Velocidad promedio en cada instante (km/h)
    """
    velocity = greenshields_velocity(rho)
    avg_velocity = np.mean(velocity, axis=1)
    return avg_velocity
