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
    """Ejecuta un escenario microscopico y retorna resultados."""
    print(f"  - Ejecutando modelo microscopico ({n_cars} vehiculos)...")

    params = {
        "v0": 30.0,
        "a": 1.2,
        "b": 1.5,
        "T": 1.5,
        "s0": 2.0,
        "s_min": 2.0
    }

    init_positions = np.linspace(0.0, road_length * 0.9, n_cars)
    v0 = params["v0"]
    init_velocities = np.ones(n_cars, dtype=float) * (0.8 * v0)

    model = MicroscopicModel(
        n_cars=n_cars,
        road_length=road_length,
        params=params,
        init_positions=init_positions,
        init_velocities=init_velocities
    )

    t_array = np.arange(0.0, final_time + dt, dt)
    n_steps = len(t_array) - 1

    positions_record, velocities_record = simulate_microscopic(
        model,
        dt=dt,
        n_steps=n_steps,
        periodic=False,
        road_length=road_length,
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
    """Calcula metricas comparativas entre ambos modelos."""
    macro_rho = macro_result['rho']
    micro_vels = micro_result['velocities']

    # Calcular congestión (porcentaje de puntos con densidad > 75 veh/km)
    congestion_threshold = 75.0
    congestion_fraction = np.sum(macro_rho > congestion_threshold) / macro_rho.size
    max_congestion_fraction = np.max(np.mean(macro_rho > congestion_threshold, axis=1))

    metrics = {
        'scenario': macro_result['name'],
        'macro_avg_density': np.mean(macro_rho),
        'macro_max_density': np.max(macro_rho),
        'macro_min_density': np.min(macro_rho),
        'congestion_fraction': congestion_fraction,
        'max_congestion_fraction': max_congestion_fraction,
        'micro_avg_velocity': np.mean(micro_vels),
        'micro_max_velocity': np.max(micro_vels),
        'micro_min_velocity': np.min(micro_vels),
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
    """
    Ejecuta analisis comparativo entre modelos.

    El modo de analisis comparativo SIEMPRE ejecuta silenciosamente sin mostrar
    graficos individuales, solo muestra el resumen comparativo final.
    """
    print("\n" + "="*80)
    print("ANALISIS COMPARATIVO: MODELO MACROSCOPICO vs MICROSCOPICO")
    print("="*80)

    output_dirs = create_output_directory()
    print(f"\nResultados se guardaran en: {output_dirs['figures']}")

    # Parametros para modelo macroscopico
    L = 10.0
    T = 1.0
    dx = 0.1
    dt = 0.001

    x = get_spatial_grid(L=L, dx=dx)
    t = get_temporal_grid(T=T, dt=dt)

    # Parametros para modelo microscopico (convertir a segundos)
    road_length = 10000.0  # 10 km en metros
    final_time = 3600.0    # 1 hora en segundos
    dt_micro = 1.0         # paso temporal en segundos

    print(f"\nParametros de simulacion:")
    print(f"  Macroscopico: L={L} km, T={T} h, dx={dx} km, dt={dt} h")
    print(f"  Microscopico: L={road_length} m, T={final_time} s, dt={dt_micro} s")

    # Definir escenarios comparativos
    print("\nEjecutando escenarios comparativos (modo silencioso)...\n")

    scenarios = [
        {
            'name': 'Flujo Libre',
            'macro_rho0': uniform_density(x, rho_value=30.0),
            'micro_cars': 10
        },
        {
            'name': 'Flujo Moderado',
            'macro_rho0': uniform_density(x, rho_value=75.0),
            'micro_cars': 20
        },
        {
            'name': 'Congestion Severa',
            'macro_rho0': shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0),
            'micro_cars': 30
        }
    ]

    all_macro_results = []
    all_micro_results = []
    all_metrics = []

    for scenario in scenarios:
        print(f"\nEscenario: {scenario['name']}")
        print("-" * 60)

        # Ejecutar modelo macroscopico
        macro_result = run_macroscopic_scenario(
            scenario['name'],
            scenario['macro_rho0'],
            x, t
        )
        all_macro_results.append(macro_result)

        # Ejecutar modelo microscopico
        micro_result = run_microscopic_scenario(
            scenario['name'],
            scenario['micro_cars'],
            road_length,
            final_time,
            dt_micro
        )
        all_micro_results.append(micro_result)

        # Calcular metricas
        metrics = compute_metrics(macro_result, micro_result)
        all_metrics.append(metrics)

        print(f"    Densidad macro promedio: {metrics['macro_avg_density']:.2f} veh/km")
        print(f"    Densidad macro maxima:   {metrics['macro_max_density']:.2f} veh/km")
        print(f"    Velocidad micro promedio: {metrics['micro_avg_velocity']:.2f} m/s")

        # Generar graficas comparativas (sin mostrar, solo guardar)
        plot_comparative_density_velocity(macro_result, micro_result, output_dirs, show_plots=False)

    # Generar resumen final (ESTE SI SE MUESTRA)
    print("\nGenerando resumen comparativo...")
    print("-" * 60)
    plot_scenario_summary(all_metrics, output_dirs, show_plots=True)

    # Resumen de metricas
    print("\n" + "="*80)
    print("RESUMEN DE METRICAS")
    print("="*80)

    for metrics in all_metrics:
        print(f"\n{metrics['scenario']}")
        print("-" * 60)
        print(f"  Densidad promedio (macro):  {metrics['macro_avg_density']:.2f} veh/km")
        print(f"  Densidad maxima (macro):    {metrics['macro_max_density']:.2f} veh/km")
        print(f"  Densidad minima (macro):    {metrics['macro_min_density']:.2f} veh/km")
        print(f"  Velocidad promedio (micro): {metrics['micro_avg_velocity']:.2f} m/s")
        print(f"  Velocidad maxima (micro):   {metrics['micro_max_velocity']:.2f} m/s")
        print(f"  Velocidad minima (micro):   {metrics['micro_min_velocity']:.2f} m/s")
        print(f"  Numero de vehiculos:        {metrics['micro_n_cars']}")

    print("\n" + "="*80)
    print("OK Analisis comparativo completado")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
