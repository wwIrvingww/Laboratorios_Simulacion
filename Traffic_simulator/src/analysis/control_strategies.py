"""
Control Strategies for Traffic Flow
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
import warnings


class VariableSpeedLimit:
    def __init__(self, v_max_default: float = 100.0, rho_max: float = 150.0):
        self.v_max_default = v_max_default
        self.rho_max = rho_max
        self.rho_critical = rho_max / 2.0
        self.rho_warning = 0.6 * rho_max
        self.rho_danger = 0.8 * rho_max

    def compute_controlled_velocity(self, rho: np.ndarray,
                                    aggressive: bool = False) -> np.ndarray:
        v_controlled = np.ones_like(rho) * self.v_max_default

        if aggressive:
            for i in range(len(rho)):
                if rho[i] > self.rho_danger:
                    reduction_factor = 0.4 + 0.2 * (1 - (rho[i] - self.rho_danger) /
                                                    (self.rho_max - self.rho_danger))
                    v_controlled[i] = self.v_max_default * reduction_factor
                elif rho[i] > self.rho_warning:
                    reduction_factor = 0.7 + 0.3 * (1 - (rho[i] - self.rho_warning) /
                                                    (self.rho_danger - self.rho_warning))
                    v_controlled[i] = self.v_max_default * reduction_factor
        else:
            for i in range(len(rho)):
                if rho[i] > self.rho_critical:
                    reduction_factor = 0.5 + 0.5 * (1 - (rho[i] - self.rho_critical) /
                                                    (self.rho_max - self.rho_critical))
                    v_controlled[i] = self.v_max_default * max(0.4, reduction_factor)

        return np.clip(v_controlled, 30.0, self.v_max_default)

    def apply_control(self, rho: np.ndarray, x: np.ndarray,
                     control_zone: Optional[Tuple[float, float]] = None,
                     aggressive: bool = False) -> Dict:
        v_controlled = self.compute_controlled_velocity(rho, aggressive)

        if control_zone is not None:
            x_start, x_end = control_zone
            mask = (x >= x_start) & (x <= x_end)
            v_result = np.ones_like(rho) * self.v_max_default
            v_result[mask] = v_controlled[mask]
        else:
            v_result = v_controlled

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
    def __init__(self, target_density: float = 75.0, max_inflow: float = 2000.0):
        self.target_density = target_density
        self.max_inflow = max_inflow
        self.min_inflow = 200.0

    def compute_optimal_inflow(self, rho_mainline: float,
                               current_inflow: float = 1000.0) -> float:
        error = self.target_density - rho_mainline
        Kp = 20.0
        adjustment = Kp * error
        optimal_inflow = current_inflow + adjustment
        return np.clip(optimal_inflow, self.min_inflow, self.max_inflow)

    def compute_green_time(self, optimal_inflow: float,
                          cycle_time: float = 60.0) -> float:
        saturation_flow = 1800.0
        vehicles_per_cycle = optimal_inflow / 3600.0 * cycle_time
        green_time = (vehicles_per_cycle / (saturation_flow / 3600.0)) * cycle_time
        return np.clip(green_time, 5.0, cycle_time - 5.0)


class FeedbackController:
    def __init__(self, setpoint: float = 75.0,
                 Kp: float = 1.0, Ki: float = 0.1, Kd: float = 0.05):
        self.setpoint = setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0.0
        self.previous_error = 0.0

    def compute_control_action(self, current_density: float, dt: float = 0.01) -> float:
        error = self.setpoint - current_density
        P = self.Kp * error

        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -50, 50)
        I = self.Ki * self.integral_error

        derivative_error = (error - self.previous_error) / dt if dt > 0 else 0
        D = self.Kd * derivative_error

        self.previous_error = error
        control_signal = P + I + D
        return control_signal

    def reset(self):
        self.integral_error = 0.0
        self.previous_error = 0.0


class PredictiveControl:
    def __init__(self, prediction_horizon: int = 5,
                 gradient_threshold: float = 50.0):
        self.prediction_horizon = prediction_horizon
        self.gradient_threshold = gradient_threshold

    def detect_congestion_formation(self, rho: np.ndarray,
                                   x: np.ndarray) -> Dict:
        dx = x[1] - x[0]
        gradient = np.gradient(rho, dx)
        critical_indices = np.where(np.abs(gradient) > self.gradient_threshold)[0]

        upstream_zones = []
        for idx in critical_indices:
            if idx > 0 and gradient[idx] > 0:
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
        detection = self.detect_congestion_formation(rho, x)
        v_controlled = np.ones_like(rho) * v_max

        for zone_start, zone_end in detection['upstream_zones']:
            mask = (x >= zone_start) & (x <= zone_end)
            reduction = np.linspace(0.8, 0.6, np.sum(mask))
            v_controlled[mask] = v_max * reduction

        return {
            'v_controlled': v_controlled,
            'upstream_zones': detection['upstream_zones'],
            'max_gradient': detection['max_gradient'],
            'control_active': len(detection['upstream_zones']) > 0
        }


class ZoneBasedControl:
    def __init__(self, x: np.ndarray, n_zones: int = 3):
        self.x = x
        self.n_zones = n_zones
        self.zone_boundaries = np.linspace(x[0], x[-1], n_zones + 1)

    def assign_zones(self) -> np.ndarray:
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
        zones = self.assign_zones()
        zone_metrics = self.compute_zone_metrics(rho)
        v_controlled = np.ones_like(rho) * v_max

        for z in range(self.n_zones):
            mask = zones == z
            avg_rho = zone_metrics[f'zone_{z}']['avg_density']

            if avg_rho > 100:
                v_controlled[mask] = v_max * 0.5
            elif avg_rho > 75:
                v_controlled[mask] = v_max * 0.7
            elif avg_rho > 50:
                v_controlled[mask] = v_max * 0.85

        return {
            'v_controlled': v_controlled,
            'zone_metrics': zone_metrics,
            'zones': zones
        }


def apply_integrated_control(rho: np.ndarray, x: np.ndarray, t: float,
                             v_max: float = 100.0, rho_max: float = 150.0,
                             strategy: str = 'vsl') -> Dict:
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
        vsl_controller = VariableSpeedLimit(v_max, rho_max)
        pred_controller = PredictiveControl()

        vsl_result = vsl_controller.apply_control(rho, x, aggressive=False)
        pred_result = pred_controller.compute_preventive_control(rho, x, v_max)

        v_controlled = np.minimum(vsl_result['v_controlled'],
                                 pred_result['v_controlled'])

        result = {
            'v_controlled': v_controlled,
            'strategy': 'Hybrid (VSL + Predictive)',
            'vsl_active': vsl_result['active_points'],
            'predictive_zones': len(pred_result['upstream_zones'])
        }
    else:
        result = {
            'v_controlled': np.ones_like(rho) * v_max,
            'strategy': 'No Control'
        }

    return result


def compare_control_strategies(rho: np.ndarray, x: np.ndarray,
                              strategies: list = None) -> Dict:
    if strategies is None:
        strategies = ['vsl', 'vsl_aggressive', 'predictive', 'zone', 'hybrid']

    results = {}
    for strategy in strategies:
        results[strategy] = apply_integrated_control(rho, x, 0.0, strategy=strategy)

    return results
