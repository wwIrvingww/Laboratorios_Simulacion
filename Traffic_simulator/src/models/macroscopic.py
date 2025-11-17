"""
Modelo Macroscópico - Greenshields

Ecuación de conservación: ∂ρ/∂t + ∂q/∂x = 0
Modelo de Greenshields: v(ρ) = V_max * (1 - ρ/ρ_max)
Flujo: q(ρ) = ρ * v(ρ)
"""

import numpy as np
from ..utils.parameters import V_MAX, RHO_MAX, get_velocity, get_flux
from ..solvers.lax_friedrichs import lax_friedrichs_solve, adaptive_lax_friedrichs_solve


def greenshields_flux(rho):
    return get_flux(rho)


def greenshields_velocity(rho):
    return get_velocity(rho)


def simulate_traffic_flow(rho0, x, t, boundary='periodic', adaptive=False, cfl_target=0.8):
    if adaptive:
        rho, t_array = adaptive_lax_friedrichs_solve(
            rho0, greenshields_flux, x, t_final=t,
            cfl_target=cfl_target, boundary=boundary
        )
    else:
        t_array = t
        rho = lax_friedrichs_solve(rho0, greenshields_flux, x, t, boundary=boundary)

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
    if rho_range is None:
        rho_range = np.linspace(0, RHO_MAX, 200)

    flux = greenshields_flux(rho_range)
    velocity = greenshields_velocity(rho_range)

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
    # c(ρ) = dq/dρ = V_max * (1 - 2*ρ/ρ_max)
    return V_MAX * (1 - 2 * rho / RHO_MAX)


def detect_shock_waves(rho, x, t, threshold_gradient=50.0):
    dx = x[1] - x[0]
    gradient = np.gradient(rho, dx, axis=1)
    shock_mask = np.abs(gradient) > threshold_gradient

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
    velocity = greenshields_velocity(rho)
    dx = x[1] - x[0]
    velocity_safe = np.maximum(velocity, 1e-6)
    travel_time = np.sum(dx / velocity_safe, axis=1)
    return travel_time


def compute_congestion_level(rho, threshold=75.0):
    congestion_mask = rho > threshold
    congestion_fraction = np.mean(congestion_mask, axis=1)
    return {
        'congestion_fraction': congestion_fraction,
        'congestion_mask': congestion_mask
    }


def compute_total_vehicles(rho, x):
    dx = x[1] - x[0]
    total_vehicles = np.sum(rho, axis=1) * dx
    return total_vehicles


def compute_average_density(rho, x):
    L = x[-1] - x[0]
    dx = x[1] - x[0]
    avg_density = np.sum(rho, axis=1) * dx / L
    return avg_density


def compute_average_velocity(rho):
    velocity = greenshields_velocity(rho)
    avg_velocity = np.mean(velocity, axis=1)
    return avg_velocity
