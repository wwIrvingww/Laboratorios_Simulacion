"""
Estrategias de Control para Simulación de Tráfico Vehicular.

Este módulo implementa diferentes estrategias de control activo para mejorar
el flujo vehicular y reducir la congestión en modelos macroscópicos de tráfico.

Estrategias implementadas:
1. Variable Speed Limits (VSL) - Límites de velocidad variable
2. Ramp Metering - Control de rampas de entrada
3. Feedback Control - Control por retroalimentación (PID)
4. Predictive Control - Control predictivo basado en densidad
5. Zone-based Control - Control por zonas

Autor: Sistema de Simulación de Tráfico
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
import warnings


class VariableSpeedLimit:
    """
    Control de Límites de Velocidad Variable (VSL).
    
    Ajusta dinámicamente la velocidad máxima permitida según la densidad
    de tráfico para prevenir congestión y ondas de choque.
    """
    
    def __init__(self, v_max_default: float = 100.0, rho_max: float = 150.0):
        """
        Inicializa el controlador VSL.
        
        Parámetros:
            v_max_default (float): Velocidad máxima sin control (km/h)
            rho_max (float): Densidad máxima (veh/km)
        """
        self.v_max_default = v_max_default
        self.rho_max = rho_max
        self.rho_critical = rho_max / 2.0
        
        # Umbrales de activación
        self.rho_warning = 0.6 * rho_max  # 90 veh/km - activar control suave
        self.rho_danger = 0.8 * rho_max   # 120 veh/km - control agresivo
        
    def compute_controlled_velocity(self, rho: np.ndarray, 
                                    aggressive: bool = False) -> np.ndarray:
        """
        Calcula la velocidad máxima controlada según la densidad.
        
        Parámetros:
            rho (np.ndarray): Densidad en cada punto espacial
            aggressive (bool): Si True, aplica control más agresivo
            
        Retorna:
            np.ndarray: Velocidad máxima ajustada para cada punto
        """
        v_controlled = np.ones_like(rho) * self.v_max_default
        
        if aggressive:
            # Control agresivo: reducción proporcional a la congestión
            for i in range(len(rho)):
                if rho[i] > self.rho_danger:
                    # Zona de peligro: reducir hasta 40 km/h
                    reduction_factor = 0.4 + 0.2 * (1 - (rho[i] - self.rho_danger) / 
                                                    (self.rho_max - self.rho_danger))
                    v_controlled[i] = self.v_max_default * reduction_factor
                elif rho[i] > self.rho_warning:
                    # Zona de advertencia: reducir hasta 70 km/h
                    reduction_factor = 0.7 + 0.3 * (1 - (rho[i] - self.rho_warning) / 
                                                    (self.rho_danger - self.rho_warning))
                    v_controlled[i] = self.v_max_default * reduction_factor
        else:
            # Control suave: transición gradual
            for i in range(len(rho)):
                if rho[i] > self.rho_critical:
                    # Reducción lineal después de densidad crítica
                    reduction_factor = 0.5 + 0.5 * (1 - (rho[i] - self.rho_critical) / 
                                                    (self.rho_max - self.rho_critical))
                    v_controlled[i] = self.v_max_default * max(0.4, reduction_factor)
        
        return np.clip(v_controlled, 30.0, self.v_max_default)  # Mínimo 30 km/h
    
    def apply_control(self, rho: np.ndarray, x: np.ndarray, 
                     control_zone: Optional[Tuple[float, float]] = None,
                     aggressive: bool = False) -> Dict:
        """
        Aplica VSL en una zona específica o en toda la vía.
        
        Parámetros:
            rho (np.ndarray): Densidad actual
            x (np.ndarray): Malla espacial
            control_zone (tuple): (x_start, x_end) zona de control
            aggressive (bool): Tipo de control
            
        Retorna:
            dict: Información del control aplicado
        """
        v_controlled = self.compute_controlled_velocity(rho, aggressive)
        
        # Aplicar solo en zona específica si se especifica
        if control_zone is not None:
            x_start, x_end = control_zone
            mask = (x >= x_start) & (x <= x_end)
            v_result = np.ones_like(rho) * self.v_max_default
            v_result[mask] = v_controlled[mask]
        else:
            v_result = v_controlled
        
        # Calcular métricas de control
        avg_reduction = np.mean((self.v_max_default - v_result) / self.v_max_default * 100)
        max_reduction = np.max((self.v_max_default - v_result) / self.v_max_default * 100)
        active_points = np.sum(v_result < self.v_max_default)
        
        return {
            'v_controlled': v_result,
            'avg_reduction_percent': avg_reduction,
            'max_reduction_percent': max_reduction,
            'active_points': active_points,
            'total_points': len(x)
        }


class RampMetering:
    """
    Control de Rampas de Entrada (Ramp Metering).
    
    Regula el flujo de vehículos que entran a la autopista para mantener
    condiciones óptimas de flujo y evitar sobresaturación.
    """
    
    def __init__(self, target_density: float = 75.0, max_inflow: float = 2000.0):
        """
        Inicializa el controlador de rampa.
        
        Parámetros:
            target_density (float): Densidad objetivo (veh/km)
            max_inflow (float): Flujo máximo de entrada (veh/h)
        """
        self.target_density = target_density
        self.max_inflow = max_inflow
        self.min_inflow = 200.0  # Flujo mínimo para no bloquear completamente
        
    def compute_optimal_inflow(self, rho_mainline: float, 
                               current_inflow: float = 1000.0) -> float:
        """
        Calcula el flujo óptimo de entrada basado en densidad de la vía principal.
        
        Parámetros:
            rho_mainline (float): Densidad promedio en vía principal
            current_inflow (float): Flujo actual de entrada
            
        Retorna:
            float: Flujo de entrada óptimo (veh/h)
        """
        # Control proporcional simple
        error = self.target_density - rho_mainline
        Kp = 20.0  # Ganancia proporcional
        
        adjustment = Kp * error
        optimal_inflow = current_inflow + adjustment
        
        # Limitar entre min y max
        return np.clip(optimal_inflow, self.min_inflow, self.max_inflow)
    
    def compute_green_time(self, optimal_inflow: float, 
                          cycle_time: float = 60.0) -> float:
        """
        Calcula el tiempo en verde del semáforo de rampa.
        
        Parámetros:
            optimal_inflow (float): Flujo óptimo (veh/h)
            cycle_time (float): Tiempo de ciclo del semáforo (s)
            
        Retorna:
            float: Tiempo en verde (s)
        """
        # Asumiendo flujo de saturación de 1800 veh/h por carril
        saturation_flow = 1800.0
        vehicles_per_cycle = optimal_inflow / 3600.0 * cycle_time
        green_time = (vehicles_per_cycle / (saturation_flow / 3600.0)) * cycle_time
        
        return np.clip(green_time, 5.0, cycle_time - 5.0)


class FeedbackController:
    """
    Controlador PID por Retroalimentación.
    
    Implementa un controlador Proporcional-Integral-Derivativo para mantener
    la densidad cerca de un valor objetivo.
    """
    
    def __init__(self, setpoint: float = 75.0, 
                 Kp: float = 1.0, Ki: float = 0.1, Kd: float = 0.05):
        """
        Inicializa el controlador PID.
        
        Parámetros:
            setpoint (float): Densidad objetivo (veh/km)
            Kp (float): Ganancia proporcional
            Ki (float): Ganancia integral
            Kd (float): Ganancia derivativa
        """
        self.setpoint = setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        # Estado del controlador
        self.integral_error = 0.0
        self.previous_error = 0.0
        
    def compute_control_action(self, current_density: float, dt: float = 0.01) -> float:
        """
        Calcula la acción de control PID.
        
        Parámetros:
            current_density (float): Densidad actual
            dt (float): Paso de tiempo (h)
            
        Retorna:
            float: Señal de control (ajuste de velocidad en km/h)
        """
        # Error actual
        error = self.setpoint - current_density
        
        # Término proporcional
        P = self.Kp * error
        
        # Término integral (con anti-windup)
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -50, 50)
        I = self.Ki * self.integral_error
        
        # Término derivativo
        derivative_error = (error - self.previous_error) / dt if dt > 0 else 0
        D = self.Kd * derivative_error
        
        # Actualizar error previo
        self.previous_error = error
        
        # Señal de control total
        control_signal = P + I + D
        
        return control_signal
    
    def reset(self):
        """Reinicia el estado del controlador."""
        self.integral_error = 0.0
        self.previous_error = 0.0


class PredictiveControl:
    """
    Control Predictivo basado en detección anticipada de congestión.
    
    Analiza gradientes de densidad para predecir formación de ondas de choque
    y aplicar control preventivo.
    """
    
    def __init__(self, prediction_horizon: int = 5, 
                 gradient_threshold: float = 50.0):
        """
        Inicializa el controlador predictivo.
        
        Parámetros:
            prediction_horizon (int): Número de pasos temporales a predecir
            gradient_threshold (float): Umbral de gradiente para activar control
        """
        self.prediction_horizon = prediction_horizon
        self.gradient_threshold = gradient_threshold
        
    def detect_congestion_formation(self, rho: np.ndarray, 
                                   x: np.ndarray) -> Dict:
        """
        Detecta zonas donde se está formando congestión.
        
        Parámetros:
            rho (np.ndarray): Densidad actual
            x (np.ndarray): Malla espacial
            
        Retorna:
            dict: Información sobre zonas críticas
        """
        # Calcular gradiente espacial de densidad
        dx = x[1] - x[0]
        gradient = np.gradient(rho, dx)
        
        # Detectar gradientes altos (posibles ondas de choque)
        critical_indices = np.where(np.abs(gradient) > self.gradient_threshold)[0]
        
        # Identificar zonas upstream (antes de la congestión)
        upstream_zones = []
        for idx in critical_indices:
            if idx > 0 and gradient[idx] > 0:  # Gradiente positivo = congestión adelante
                # Zona upstream: aplicar control preventivo
                upstream_start = max(0, idx - 10)
                upstream_end = idx
                upstream_zones.append((x[upstream_start], x[upstream_end]))
        
        return {
            'critical_points': critical_indices,
            'gradient': gradient,
            'upstream_zones': upstream_zones,
            'max_gradient': np.max(np.abs(gradient)) if len(gradient) > 0 else 0
        }
    
    def compute_preventive_control(self, rho: np.ndarray, x: np.ndarray,
                                   v_max: float = 100.0) -> Dict:
        """
        Calcula control preventivo basado en predicción de congestión.
        
        Parámetros:
            rho (np.ndarray): Densidad actual
            x (np.ndarray): Malla espacial
            v_max (float): Velocidad máxima sin control
            
        Retorna:
            dict: Velocidad controlada y zonas de actuación
        """
        detection = self.detect_congestion_formation(rho, x)
        v_controlled = np.ones_like(rho) * v_max
        
        # Aplicar reducción de velocidad en zonas upstream
        for zone_start, zone_end in detection['upstream_zones']:
            mask = (x >= zone_start) & (x <= zone_end)
            # Reducir velocidad gradualmente: 80% en inicio, 60% cerca de congestión
            reduction = np.linspace(0.8, 0.6, np.sum(mask))
            v_controlled[mask] = v_max * reduction
        
        return {
            'v_controlled': v_controlled,
            'upstream_zones': detection['upstream_zones'],
            'max_gradient': detection['max_gradient'],
            'control_active': len(detection['upstream_zones']) > 0
        }


class ZoneBasedControl:
    """
    Control por Zonas Geográficas.
    
    Divide la vía en zonas y aplica estrategias de control específicas
    según las características de cada zona.
    """
    
    def __init__(self, x: np.ndarray, n_zones: int = 3):
        """
        Inicializa el control por zonas.
        
        Parámetros:
            x (np.ndarray): Malla espacial
            n_zones (int): Número de zonas
        """
        self.x = x
        self.n_zones = n_zones
        self.zone_boundaries = np.linspace(x[0], x[-1], n_zones + 1)
        
    def assign_zones(self) -> np.ndarray:
        """
        Asigna cada punto espacial a una zona.
        
        Retorna:
            np.ndarray: Índice de zona para cada punto (0 a n_zones-1)
        """
        zones = np.zeros(len(self.x), dtype=int)
        for i, x_val in enumerate(self.x):
            for z in range(self.n_zones):
                if self.zone_boundaries[z] <= x_val < self.zone_boundaries[z + 1]:
                    zones[i] = z
                    break
            if x_val >= self.zone_boundaries[-1]:
                zones[i] = self.n_zones - 1
        return zones
    
    def compute_zone_metrics(self, rho: np.ndarray) -> Dict:
        """
        Calcula métricas agregadas por zona.
        
        Parámetros:
            rho (np.ndarray): Densidad actual
            
        Retorna:
            dict: Métricas por zona
        """
        zones = self.assign_zones()
        zone_metrics = {}
        
        for z in range(self.n_zones):
            mask = zones == z
            zone_metrics[f'zone_{z}'] = {
                'avg_density': np.mean(rho[mask]),
                'max_density': np.max(rho[mask]),
                'min_density': np.min(rho[mask]),
                'std_density': np.std(rho[mask]),
                'n_points': np.sum(mask)
            }
        
        return zone_metrics
    
    def apply_zone_specific_control(self, rho: np.ndarray, 
                                    v_max: float = 100.0) -> Dict:
        """
        Aplica control diferenciado por zona según sus características.
        
        Parámetros:
            rho (np.ndarray): Densidad actual
            v_max (float): Velocidad máxima base
            
        Retorna:
            dict: Velocidad controlada por zona
        """
        zones = self.assign_zones()
        zone_metrics = self.compute_zone_metrics(rho)
        v_controlled = np.ones_like(rho) * v_max
        
        # Aplicar diferentes niveles de control por zona
        for z in range(self.n_zones):
            mask = zones == z
            avg_rho = zone_metrics[f'zone_{z}']['avg_density']
            
            if avg_rho > 100:  # Zona muy congestionada
                v_controlled[mask] = v_max * 0.5
            elif avg_rho > 75:  # Zona moderadamente congestionada
                v_controlled[mask] = v_max * 0.7
            elif avg_rho > 50:  # Zona con tráfico medio
                v_controlled[mask] = v_max * 0.85
            # else: mantener v_max
        
        return {
            'v_controlled': v_controlled,
            'zone_metrics': zone_metrics,
            'zones': zones
        }


def apply_integrated_control(rho: np.ndarray, x: np.ndarray, t: float,
                             v_max: float = 100.0, rho_max: float = 150.0,
                             strategy: str = 'vsl') -> Dict:
    """
    Aplica estrategia de control integrada según el tipo especificado.
    
    Parámetros:
        rho (np.ndarray): Densidad actual
        x (np.ndarray): Malla espacial
        t (float): Tiempo actual
        v_max (float): Velocidad máxima sin control
        rho_max (float): Densidad máxima
        strategy (str): Tipo de estrategia ('vsl', 'predictive', 'zone', 'hybrid')
        
    Retorna:
        dict: Resultado del control aplicado
    """
    if strategy == 'vsl':
        controller = VariableSpeedLimit(v_max, rho_max)
        result = controller.apply_control(rho, x, aggressive=False)
        result['strategy'] = 'Variable Speed Limit'
        
    elif strategy == 'vsl_aggressive':
        controller = VariableSpeedLimit(v_max, rho_max)
        result = controller.apply_control(rho, x, aggressive=True)
        result['strategy'] = 'VSL Aggressive'
        
    elif strategy == 'predictive':
        controller = PredictiveControl()
        result = controller.compute_preventive_control(rho, x, v_max)
        result['strategy'] = 'Predictive Control'
        
    elif strategy == 'zone':
        controller = ZoneBasedControl(x, n_zones=3)
        result = controller.apply_zone_specific_control(rho, v_max)
        result['strategy'] = 'Zone-Based Control'
        
    elif strategy == 'hybrid':
        # Combinar VSL + Predictivo
        vsl_controller = VariableSpeedLimit(v_max, rho_max)
        pred_controller = PredictiveControl()
        
        vsl_result = vsl_controller.apply_control(rho, x, aggressive=False)
        pred_result = pred_controller.compute_preventive_control(rho, x, v_max)
        
        # Tomar el control más restrictivo (menor velocidad)
        v_controlled = np.minimum(vsl_result['v_controlled'], 
                                 pred_result['v_controlled'])
        
        result = {
            'v_controlled': v_controlled,
            'strategy': 'Hybrid (VSL + Predictive)',
            'vsl_active': vsl_result['active_points'],
            'predictive_zones': len(pred_result['upstream_zones'])
        }
    else:
        # Sin control
        result = {
            'v_controlled': np.ones_like(rho) * v_max,
            'strategy': 'No Control'
        }
    
    return result


def compare_control_strategies(rho: np.ndarray, x: np.ndarray,
                              strategies: list = None) -> Dict:
    """
    Compara múltiples estrategias de control sobre el mismo estado de tráfico.
    
    Parámetros:
        rho (np.ndarray): Densidad actual
        x (np.ndarray): Malla espacial
        strategies (list): Lista de estrategias a comparar
        
    Retorna:
        dict: Resultados comparativos
    """
    if strategies is None:
        strategies = ['vsl', 'vsl_aggressive', 'predictive', 'zone', 'hybrid']
    
    results = {}
    for strategy in strategies:
        results[strategy] = apply_integrated_control(rho, x, 0.0, strategy=strategy)
    
    return results
