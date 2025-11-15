#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisis Comparativo entre Modelos Macroscopico y Microscopico.

Este script ejecuta simulaciones de los dos modelos para los mismos escenarios
y compara sus resultados, generando graficas comparativas de desempeno.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Anadir el directorio raiz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.macroscopic import simulate_traffic_flow
from src.models.microscopic import MicroscopicModel
from src.solvers.runge_kutta import simulate as simulate_microscopic
from src.utils.parameters import (
    get_spatial_grid,
    get_temporal_grid,
    V_MAX,
    RHO_MAX
)
from src.utils.initial_conditions import (
    uniform_density,
    gaussian_pulse,
    shock_wave_scenario
)
from src.visualization.density_maps import plot_density_heatmap
from src.visualization.spacetime_diagrams import plot_spacetime_diagram_macro, plot_spacetime_diagram_micro


def create_output_directory(base_dir='results'):
    """Crea estructura de directorios para resultados comparativos."""
    dirs = {
        'figures': os.path.join(base_dir, 'figures', 'comparative'),
        'metrics': os.path.join(base_dir, 'metrics')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def run_macroscopic_scenario(scenario_name, rho0, x, t):
    """Ejecuta un escenario macroscopico y retorna resultados."""
    print(f"  - Ejecutando modelo macroscopico...")

    results = simulate_traffic_flow(rho0, x, t, boundary='periodic')

    return {
        'name': scenario_name,
        'rho': results['rho'],
        'flux': results['flux'],
        'x': x,
        't': t,
        'model': 'macroscopico'
    }


def run_microscopic_scenario(scenario_name, n_cars, road_length, final_time, dt):
    print(f"  - Ejecutando modelo microscopico ({n_cars} vehiculos)...")

    params = {"v0": 30.0, "a": 1.2, "b": 1.5, "T": 1.5, "s0": 2.0, "s_min": 2.0}

    # Densidad real microscópica
    km_length = road_length / 1000.0  
    density = n_cars / km_length
    print(f"    Densidad microscópica real: {density:.2f} veh/km")

    # Posiciones iniciales cercanas para forzar congestión
    spacing = road_length / n_cars * 0.5
    init_positions = np.array([i * spacing for i in range(n_cars)], dtype=float)

    # Velocidad inicial depende de la densidad
    if density < 20:            # flujo libre
        init_velocities = np.ones(n_cars) * 28.0
    elif density < 60:          # moderado
        init_velocities = np.linspace(18, 10, n_cars)
    else:                       # congestion severa
        init_velocities = np.linspace(6, 1, n_cars)

    model = MicroscopicModel(
        n_cars=n_cars,
        road_length=road_length,
        params=params,
        init_positions=init_positions,
        init_velocities=init_velocities
    )

    t_array = np.arange(0, final_time + dt, dt)
    n_steps = len(t_array) - 1

    positions_record, velocities_record = simulate_microscopic(
        model, dt=dt, n_steps=n_steps,
        periodic=False, road_length=road_length,
        record=True
    )

    return {
        'name': scenario_name,
        'positions': positions_record,
        'velocities': velocities_record,
        't': t_array,
        'n_cars': n_cars,
        'road_length': road_length,
        'model': 'microscopico'
    }


def compute_metrics(macro_result, micro_result):
    macro_rho = macro_result['rho']
    micro_vels = micro_result['velocities']

    metrics = {
        'scenario': macro_result['name'],

        # MACROSCÓPICO
        'macro_avg_density': float(np.mean(macro_rho)),
        'macro_max_density': float(np.max(macro_rho)),
        'macro_min_density': float(np.min(macro_rho)),

        # MICROSCÓPICO
        'micro_avg_velocity': float(np.mean(micro_vels)),
        'micro_max_velocity': float(np.max(micro_vels)),
        'micro_min_velocity': float(np.min(micro_vels)),
        'micro_n_cars': micro_result['n_cars']
    }

    return metrics



def plot_comparative_density_velocity(macro_result, micro_result, output_dirs, show_plots=True):
    """
    Genera graficas comparativas entre modelos macroscopico y microscopico.
    Muestra metricas equivalentes para comparacion directa.

    Parametros:
        show_plots: Si True, muestra la grafica; si False, solo guarda
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ===== FILA 1: Densidad/Velocidad promedio temporal =====

    # 1. Densidad promedio espacial (macro)
    ax = axes[0, 0]
    t = macro_result['t']
    rho = macro_result['rho']
    rho_spatial_avg = np.mean(rho, axis=1)  # promedio espacial en el tiempo

    ax.plot(t, rho_spatial_avg, 'b-', linewidth=2.5)
    ax.fill_between(t, 0, rho_spatial_avg, alpha=0.3, color='blue')
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Umbral (75)')
    ax.set_xlabel('Tiempo (h)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Densidad Promedio (veh/km)', fontsize=9, fontweight='bold')
    ax.set_title('Macroscopico: Densidad Promedio Temporal', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Velocidad promedio temporal (micro)
    ax = axes[0, 1]
    t_micro = micro_result['t']
    vels = micro_result['velocities']
    vel_temporal_avg = np.mean(vels, axis=1)  # promedio en los vehiculos

    ax.plot(t_micro, vel_temporal_avg, 'g-', linewidth=2.5)
    ax.fill_between(t_micro, 0, vel_temporal_avg, alpha=0.3, color='green')
    ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Velocidad Promedio (m/s)', fontsize=9, fontweight='bold')
    ax.set_title('Microscopico: Velocidad Promedio Temporal', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ===== FILA 2: Distribucion espacial/por vehiculo =====

    # 3. Densidad promedio espacial (macro)
    ax = axes[1, 0]
    x = macro_result['x']
    rho_spatial = np.mean(rho, axis=0)  # promedio temporal en el espacio

    ax.plot(x, rho_spatial, 'b-', linewidth=2.5)
    ax.fill_between(x, 0, rho_spatial, alpha=0.3, color='blue')
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Umbral (75)')
    ax.set_xlabel('Posicion (km)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Densidad Promedio (veh/km)', fontsize=9, fontweight='bold')
    ax.set_title('Macroscopico: Densidad Promedio Espacial', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Velocidad promedio por vehiculo (micro)
    ax = axes[1, 1]
    vel_avg_per_car = np.mean(vels, axis=0)  # promedio temporal por vehiculo
    n_cars = vels.shape[1]

    ax.bar(range(n_cars), vel_avg_per_car, alpha=0.7, color='green', edgecolor='black')
    ax.axhline(y=np.mean(vel_avg_per_car), color='r', linestyle='--', linewidth=2,
              label=f'Promedio: {np.mean(vel_avg_per_car):.2f} m/s')
    ax.set_xlabel('Numero de Vehiculo', fontsize=9, fontweight='bold')
    ax.set_ylabel('Velocidad Promedio (m/s)', fontsize=9, fontweight='bold')
    ax.set_title(f'Microscopico: Velocidad por Vehiculo ({n_cars} autos)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12, hspace=0.35, wspace=0.30)
    filename = os.path.join(output_dirs['figures'], f"{macro_result['name'].replace(' ', '_').lower()}_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()

    plt.close()

    print(f"    OK Grafica comparativa guardada")


def plot_scenario_summary(all_metrics, output_dirs, show_plots=True):
    """
    Genera resumen comparativo de todos los escenarios.
    Muestra 3 graficas claras y entendibles comparando ambos modelos.

    Parametros:
        show_plots: Si True, muestra la grafica; si False, solo guarda
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    scenarios = [m['scenario'] for m in all_metrics]
    x_pos = np.arange(len(scenarios))
    width = 0.35  # Ancho de las barras

    # Parámetros comunes
    v_max = 100  # km/h
    rho_max = 150  # veh/km
    macro_densities = [m['macro_avg_density'] for m in all_metrics]
    macro_velocities = [v_max * (1 - rho / rho_max) for rho in macro_densities]
    micro_velocities = [m['micro_avg_velocity'] * 3.6 for m in all_metrics]

    # Crear etiquetas seguras
    labels = []
    for i, scenario in enumerate(scenarios):
        if ':' in scenario:
            label = scenario.split(':')[1].strip()
        else:
            label = scenario
        labels.append(f"E{i+1}: {label}")

    # ===== GRAFICA 1: VELOCIDAD PROMEDIO =====
    ax = axes[0]
    bars1 = ax.bar(x_pos - width/2, macro_velocities, width, label='Macroscopico',
                   alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, micro_velocities, width, label='Microscopico',
                   alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1.5)

    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Velocidad Promedio (km/h)', fontsize=10, fontweight='bold')
    ax.set_title('1. Velocidad Promedio por Escenario', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 120)

    # ===== GRAFICA 2: DENSIDAD PROMEDIO (Macro vs Micro estimada) =====
    ax = axes[1]

    # Micro: estimar densidad equivalente usando Greenshields inverso
    # rho = rho_max * (1 - v / v_max)
    micro_densities = [rho_max * (1 - v / v_max) for v in micro_velocities]

    bars1 = ax.bar(x_pos - width/2, macro_densities, width, label='Macroscopico',
                   alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, micro_densities, width, label='Microscopico (estimada)',
                   alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1.5)

    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=75, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Umbral congestión (75)')
    ax.set_ylabel('Densidad Promedio (veh/km)', fontsize=10, fontweight='bold')
    ax.set_title('2. Densidad Promedio por Escenario', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 160)

    # ===== GRAFICA 3: FLUJO VEHICULAR (Densidad × Velocidad) =====
    ax = axes[2]

    # Flujo = densidad × velocidad (en unidades consistentes)
    # Macro: q = rho (veh/km) * v (km/h) = veh/h
    macro_flow = [rho * v for rho, v in zip(macro_densities, macro_velocities)]

    # Micro: q estimado = rho (veh/km) * v (km/h) = veh/h
    micro_flow = [rho * v for rho, v in zip(micro_densities, micro_velocities)]

    bars1 = ax.bar(x_pos - width/2, macro_flow, width, label='Macroscopico',
                   alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, micro_flow, width, label='Microscopico',
                   alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1.5)

    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Flujo Vehicular (veh/h)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Escenario', fontsize=10, fontweight='bold')
    ax.set_title('3. Flujo Vehicular por Escenario', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Resumen Comparativo: Modelos Macroscópico vs Microscópico',
                 fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    filename = os.path.join(output_dirs['figures'], 'comparative_summary.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()

    plt.close()

    print(f"OK Resumen comparativo guardado")


def main():
    print("\n================ COMPARATIVE ANALYSIS (MODIFIED) ================")

    output_dirs = create_output_directory()
    print(f"\nResultados se guardarán en: {output_dirs['figures']}")

    # Macro params
    L = 10.0
    T = 1.0
    dx = 0.1
    dt = 0.001

    x = get_spatial_grid(L=L, dx=dx)
    t = get_temporal_grid(T=T, dt=dt)

    # Micro params
    road_length = 10000.0
    final_time = 600.0
    dt_micro = 0.5

    print(f"\nParametros micro:")
    print(f"  Longitud carretera: {road_length} m")
    print(f"  Tiempo final: {final_time} s")
    print(f"  Paso dt: {dt_micro} s")

    scenarios = [
        {'name': 'Flujo Libre', 'macro_rho0': uniform_density(x, 30.0), 'micro_cars': 50},
        {'name': 'Flujo Moderado', 'macro_rho0': uniform_density(x, 75.0), 'micro_cars': 400},
        {'name': 'Congestion Severa', 'macro_rho0': shock_wave_scenario(x, 5.0, 140.0, 30.0), 'micro_cars': 900}
    ]

    all_metrics = []

    for sc in scenarios:
        print(f"\n--- {sc['name']} ---")

        macro_res = run_macroscopic_scenario(sc['name'], sc['macro_rho0'], x, t)
        micro_res = run_microscopic_scenario(sc['name'], sc['micro_cars'], road_length, final_time, dt_micro)

        metrics = compute_metrics(macro_res, micro_res)
        all_metrics.append(metrics)

        print(f"  Velocidad MICRO promedio: {metrics['micro_avg_velocity']:.2f} m/s")

    print("\n===================== RESUMEN DE MÉTRICAS =====================")

    for m in all_metrics:
        print(f"\n{m['scenario']}")
        print("-" * 60)

        print(f"  MODELO MACROSCÓPICO:")
        print(f"    Densidad promedio: {m['macro_avg_density']:.2f} veh/km")
        print(f"    Densidad máxima:   {m['macro_max_density']:.2f} veh/km")
        print(f"    Densidad mínima:   {m['macro_min_density']:.2f} veh/km")

        print(f"\n  MODELO MICROSCÓPICO:")
        print(f"    Velocidad promedio: {m['micro_avg_velocity']:.2f} m/s")
        print(f"    Velocidad máxima:   {m['micro_max_velocity']:.2f} m/s")
        print(f"    Velocidad mínima:   {m['micro_min_velocity']:.2f} m/s")
        print(f"    Número vehículos:   {m['micro_n_cars']}")


if __name__ == "__main__":
    main()