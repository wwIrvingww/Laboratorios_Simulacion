import numpy as np
from .parameters import RHO_MAX


def uniform_density(x, rho_value=50.0):
    return np.full_like(x, rho_value)


def gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5):
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * width ** 2))


def step_function(x, x_step=5.0, rho_left=20.0, rho_right=120.0):
    rho = np.zeros_like(x)
    rho[x < x_step] = rho_left
    rho[x >= x_step] = rho_right
    return rho


def sinusoidal_perturbation(x, rho_base=50.0, amplitude=30.0, wavelength=2.0):
    k = 2 * np.pi / wavelength
    return rho_base + amplitude * np.sin(k * x)


def shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0):
    return step_function(x, x_step=x_shock, rho_left=rho_upstream, rho_right=rho_downstream)


def two_pulse_scenario(x, x1=3.0, x2=7.0, amplitude1=80.0, amplitude2=100.0, width=0.5):
    pulse1 = gaussian_pulse(x, x0=x1, amplitude=amplitude1, width=width)
    pulse2 = gaussian_pulse(x, x0=x2, amplitude=amplitude2, width=width)
    return pulse1 + pulse2


def linear_gradient(x, rho_start=20.0, rho_end=120.0):
    L = x[-1] - x[0]
    return rho_start + (rho_end - rho_start) * (x - x[0]) / L


def random_fluctuations(x, rho_mean=60.0, std_dev=15.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    rho = rho_mean + std_dev * np.random.randn(len(x))
    rho = np.clip(rho, 0, RHO_MAX)
    return rho
