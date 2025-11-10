"""
Módulo para generar condiciones iniciales para simulaciones de tráfico.

Este módulo proporciona funciones para crear diversas distribuciones iniciales
de densidad vehicular que representan diferentes escenarios de tráfico.
"""

import numpy as np
from .parameters import RHO_MAX


def uniform_density(x, rho_value=50.0):
    """
    Genera una densidad uniforme en todo el dominio.
    
    Parámetros:
        x (np.ndarray): Arreglo de posiciones espaciales (km)
        rho_value (float): Valor de densidad constante (veh/km)
    
    Retorna:
        np.ndarray: Arreglo de densidad con mismo tamaño que x
    """
    return np.full_like(x, rho_value)


def gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5):
    """
    Genera un pulso gaussiano de densidad centrado en x0.
    
    Representa una perturbación localizada, como un grupo de vehículos
    concentrados en una zona específica de la carretera.
    
    Parámetros:
        x (np.ndarray): Arreglo de posiciones espaciales (km)
        x0 (float): Posición central del pulso (km)
        amplitude (float): Amplitud máxima de la densidad (veh/km)
        width (float): Ancho del pulso (km)
    
    Retorna:
        np.ndarray: Arreglo de densidad con forma gaussiana
    """
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * width ** 2))


def step_function(x, x_step=5.0, rho_left=20.0, rho_right=120.0):
    """
    Genera una función escalón (discontinuidad) en x_step.
    
    Simula una transición abrupta entre dos regiones de distinta densidad,
    útil para estudiar la formación de ondas de choque.
    
    Parámetros:
        x (np.ndarray): Arreglo de posiciones espaciales (km)
        x_step (float): Posición de la discontinuidad (km)
        rho_left (float): Densidad a la izquierda del escalón (veh/km)
        rho_right (float): Densidad a la derecha del escalón (veh/km)
    
    Retorna:
        np.ndarray: Arreglo de densidad con discontinuidad
    """
    rho = np.zeros_like(x)
    rho[x < x_step] = rho_left
    rho[x >= x_step] = rho_right
    return rho


def sinusoidal_perturbation(x, rho_base=50.0, amplitude=30.0, wavelength=2.0):
    """
    Genera una perturbación sinusoidal sobre una densidad base.
    
    Representa variaciones periódicas en la densidad, como las que podrían
    surgir por semáforos o patrones de llegada de vehículos.
    
    Parámetros:
        x (np.ndarray): Arreglo de posiciones espaciales (km)
        rho_base (float): Densidad base promedio (veh/km)
        amplitude (float): Amplitud de la oscilación (veh/km)
        wavelength (float): Longitud de onda de la perturbación (km)
    
    Retorna:
        np.ndarray: Arreglo de densidad con variación sinusoidal
    """
    k = 2 * np.pi / wavelength
    return rho_base + amplitude * np.sin(k * x)


def shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0):
    """
    Escenario diseñado para generar una onda de choque.
    
    Alta densidad aguas arriba (tráfico lento/detenido) y baja densidad
    aguas abajo (tráfico fluido). La interfaz se propaga como onda de choque.
    
    Parámetros:
        x (np.ndarray): Arreglo de posiciones espaciales (km)
        x_shock (float): Posición inicial de la interfaz (km)
        rho_upstream (float): Densidad aguas arriba (veh/km)
        rho_downstream (float): Densidad aguas abajo (veh/km)
    
    Retorna:
        np.ndarray: Arreglo de densidad para escenario de onda de choque
    """
    return step_function(x, x_step=x_shock, rho_left=rho_upstream, rho_right=rho_downstream)


def two_pulse_scenario(x, x1=3.0, x2=7.0, amplitude1=80.0, amplitude2=100.0, width=0.5):
    """
    Genera dos pulsos gaussianos separados.
    
    Útil para estudiar la interacción y fusión de dos grupos de vehículos.
    
    Parámetros:
        x (np.ndarray): Arreglo de posiciones espaciales (km)
        x1 (float): Posición del primer pulso (km)
        x2 (float): Posición del segundo pulso (km)
        amplitude1 (float): Amplitud del primer pulso (veh/km)
        amplitude2 (float): Amplitud del segundo pulso (veh/km)
        width (float): Ancho de ambos pulsos (km)
    
    Retorna:
        np.ndarray: Arreglo de densidad con dos pulsos
    """
    pulse1 = gaussian_pulse(x, x0=x1, amplitude=amplitude1, width=width)
    pulse2 = gaussian_pulse(x, x0=x2, amplitude=amplitude2, width=width)
    return pulse1 + pulse2


def linear_gradient(x, rho_start=20.0, rho_end=120.0):
    """
    Genera un gradiente lineal de densidad.
    
    La densidad aumenta o disminuye linealmente a lo largo de la carretera.
    
    Parámetros:
        x (np.ndarray): Arreglo de posiciones espaciales (km)
        rho_start (float): Densidad al inicio (x=0) (veh/km)
        rho_end (float): Densidad al final (x=L) (veh/km)
    
    Retorna:
        np.ndarray: Arreglo de densidad con variación lineal
    """
    L = x[-1] - x[0]
    return rho_start + (rho_end - rho_start) * (x - x[0]) / L


def random_fluctuations(x, rho_mean=60.0, std_dev=15.0, seed=None):
    """
    Genera fluctuaciones aleatorias alrededor de una densidad media.
    
    Simula variabilidad natural en la distribución de vehículos.
    
    Parámetros:
        x (np.ndarray): Arreglo de posiciones espaciales (km)
        rho_mean (float): Densidad media (veh/km)
        std_dev (float): Desviación estándar de las fluctuaciones (veh/km)
        seed (int, opcional): Semilla para reproducibilidad
    
    Retorna:
        np.ndarray: Arreglo de densidad con fluctuaciones aleatorias
    """
    if seed is not None:
        np.random.seed(seed)
    
    rho = rho_mean + std_dev * np.random.randn(len(x))
    # Asegurar que la densidad sea no negativa y no exceda rho_max
    rho = np.clip(rho, 0, RHO_MAX)
    return rho
