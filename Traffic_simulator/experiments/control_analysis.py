"""
Análisis de Estrategias de Control de Tráfico.

Este script ejecuta simulaciones comparando diferentes estrategias de control
para evaluar su efectividad en diversos escenarios de tráfico.

Comparaciones realizadas:
1. Sin control vs. con control (baseline)
2. VSL suave vs. VSL agresivo
3. Control predictivo vs. reactivo
4. Control por zonas
5. Estrategia híbrida

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.macroscopic import simulate_traffic_flow, greenshields_flux
from src.analysis.control_strategies import (
    VariableSpeedLimit,
    PredictiveControl,
    ZoneBasedControl,
    apply_integrated_control,
    compare_control_strategies
)
from src.utils.parameters import get_spatial_grid, get_temporal_grid, V_MAX, RHO_MAX
from src.utils.initial_conditions import (
    shock_wave_scenario,
    gaussian_pulse,
    sinusoidal_perturbation
)
from src.visualization.density_maps import plot_density_heatmap
from src.visualization.spacetime_diagrams import plot_spacetime_diagram_macro


def create_output_directory(base_dir='results'):
    """Crea estructura de directorios para resultados de control."""
    dirs = {
        'figures': os.path.join(base_dir, 'figures', 'control_analysis'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'comparisons': os.path.join(base_dir, 'figures', 'control_analysis', 'comparisons')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def simulate_with_control(rho0, x, t, control_strategy='vsl', 
                         apply_at_step=0, boundary='periodic'):
    """
    Simula flujo de tráfico aplicando estrategia de control.
    
    Parámetros:
        rho0 (np.ndarray): Condición inicial
        x (np.ndarray): Malla espacial
        t (np.ndarray): Malla temporal
        control_strategy (str): Tipo de control a aplicar
        apply_at_step (int): Paso temporal donde activar control
        boundary (str): Condiciones de frontera
        
    Retorna:
        dict: Resultados de simulación con información de control
    """
    from src.solvers.lax_friedrichs import lax_friedrichs_step
    
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    n_t = len(t)
    n_x = len(x)
    
    # Inicializar arreglos
    rho = np.zeros((n_t, n_x))
    flux = np.zeros((n_t, n_x))
    velocity = np.zeros((n_t, n_x))
    v_controlled = np.zeros((n_t, n_x))
    
    rho[0, :] = rho0
    v_controlled[0, :] = V_MAX
    
    # Inicializar controladores
    control_history = []
    
    # Evolución temporal
    for n in range(n_t - 1):
        # Aplicar control si está activo
        if n >= apply_at_step and control_strategy != 'none':
            control_result = apply_integrated_control(
                rho[n, :], x, t[n], V_MAX, RHO_MAX, control_strategy
            )
            v_max_effective = control_result['v_controlled']
            v_controlled[n, :] = v_max_effective
            control_history.append(control_result)
        else:
            v_max_effective = np.ones(n_x) * V_MAX
            v_controlled[n, :] = V_MAX
        
        # Calcular flujo con velocidad controlada
        velocity_local = v_max_effective * (1 - rho[n, :] / RHO_MAX)
        flux_local = rho[n, :] * velocity_local
        
        # Paso de Lax-Friedrichs
        if boundary == 'periodic':
            rho[n + 1, :] = lax_friedrichs_step(rho[n, :], flux_local, dx, dt, boundary='periodic')
        else:
            rho[n + 1, :] = lax_friedrichs_step(rho[n, :], flux_local, dx, dt, boundary='outflow')
        
        # Calcular velocidad y flujo resultantes
        velocity[n + 1, :] = v_max_effective * (1 - rho[n + 1, :] / RHO_MAX)
        flux[n + 1, :] = rho[n + 1, :] * velocity[n + 1, :]
    
    v_controlled[-1, :] = v_controlled[-2, :]
    
    return {
        'rho': rho,
        'flux': flux,
        'velocity': velocity,
        'v_controlled': v_controlled,
        'control_history': control_history,
        'x': x,
        't': t
    }


def compute_performance_metrics(results):
    """
    Calcula métricas de desempeño de la simulación.
    
    Parámetros:
        results (dict): Resultados de simulación
        
    Retorna:
        dict: Métricas de desempeño
    """
    rho = results['rho']
    velocity = results['velocity']
    x = results['x']
    t = results['t']
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Tiempo de viaje promedio (integral de 1/v sobre espacio)
    travel_times = []
    for n in range(len(t)):
        v_avg = np.mean(velocity[n, :])
        if v_avg > 1.0:  # Evitar división por cero
            tt = (x[-1] - x[0]) / v_avg
            travel_times.append(tt)
        else:
            travel_times.append(10.0)  # Valor máximo arbitrario
    
    avg_travel_time = np.mean(travel_times)
    
    # Densidad promedio
    avg_density = np.mean(rho)
    
    # Velocidad promedio
    avg_velocity = np.mean(velocity)
    
    # Total Vehicle-Hours (TVH)
    total_vehicle_hours = np.sum(rho) * dx * dt
    
    # Total Vehicle-Kilometers (TVK)
    total_vehicle_km = np.sum(rho * velocity) * dx * dt
    
    # Nivel de congestión (% de puntos con ρ > 75 veh/km)
    congestion_level = np.sum(rho > 75) / rho.size * 100
    
    # Throughput (flujo promedio en punto de medición)
    throughput = np.mean(results['flux'][:, len(x)//2])
    
    return {
        'avg_travel_time': avg_travel_time,
        'avg_density': avg_density,
        'avg_velocity': avg_velocity,
        'total_vehicle_hours': total_vehicle_hours,
        'total_vehicle_km': total_vehicle_km,
        'congestion_level': congestion_level,
        'throughput': throughput
    }


def scenario_control_1_shock_wave(x, t, output_dirs):
    """
    Escenario 1: Control de onda de choque.
    Compara simulación sin control vs. con VSL.
    """
    print("\n" + "="*70)
    print("ESCENARIO 1: Control de Onda de Choque")
    print("="*70)
    
    # Condición inicial: onda de choque
    rho0 = shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0)
    
    # Simular sin control
    print("  - Simulando sin control...")
    results_no_control = simulate_with_control(rho0, x, t, control_strategy='none')
    metrics_no_control = compute_performance_metrics(results_no_control)
    
    # Simular con VSL suave
    print("  - Simulando con VSL suave...")
    results_vsl = simulate_with_control(rho0, x, t, control_strategy='vsl', apply_at_step=0)
    metrics_vsl = compute_performance_metrics(results_vsl)
    
    # Simular con VSL agresivo
    print("  - Simulando con VSL agresivo...")
    results_vsl_agg = simulate_with_control(rho0, x, t, control_strategy='vsl_aggressive', apply_at_step=0)
    metrics_vsl_agg = compute_performance_metrics(results_vsl_agg)
    
    # Visualización comparativa
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Fila 1: Mapas de calor de densidad
    scenarios = [
        (results_no_control, "Sin Control"),
        (results_vsl, "VSL Suave"),
        (results_vsl_agg, "VSL Agresivo")
    ]
    
    for i, (result, title) in enumerate(scenarios):
        ax = fig.add_subplot(gs[0, i])
        im = ax.contourf(result['x'], result['t'], result['rho'], levels=20, cmap='YlOrRd')
        ax.set_xlabel('Posición (km)')
        ax.set_ylabel('Tiempo (h)')
        ax.set_title(f'{title}\nDensidad ρ(x,t)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='ρ (veh/km)')
    
    # Fila 2: Velocidad controlada
    for i, (result, title) in enumerate(scenarios):
        ax = fig.add_subplot(gs[1, i])
        im = ax.contourf(result['x'], result['t'], result['v_controlled'], 
                        levels=20, cmap='RdYlGn')
        ax.set_xlabel('Posición (km)')
        ax.set_ylabel('Tiempo (h)')
        ax.set_title(f'Velocidad Controlada', fontweight='bold')
        plt.colorbar(im, ax=ax, label='v (km/h)')
    
    # Fila 3: Comparación de métricas
    ax = fig.add_subplot(gs[2, :])
    metrics_all = [metrics_no_control, metrics_vsl, metrics_vsl_agg]
    labels = ['Sin Control', 'VSL Suave', 'VSL Agresivo']
    
    x_pos = np.arange(len(labels))
    width = 0.2
    
    # Normalizar métricas para comparación visual
    travel_times = [m['avg_travel_time'] * 60 for m in metrics_all]  # en minutos
    velocities = [m['avg_velocity'] for m in metrics_all]
    congestion = [m['congestion_level'] for m in metrics_all]
    
    ax.bar(x_pos - width, travel_times, width, label='Tiempo de Viaje (min)', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x_pos, velocities, width, label='Velocidad Promedio (km/h)', alpha=0.8, color='green')
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.bar(x_pos + width, congestion, width, label='Congestión (%)', alpha=0.8, color='red')
    
    ax.set_xlabel('Estrategia de Control', fontsize=12)
    ax.set_ylabel('Tiempo de Viaje (min)', fontsize=11, color='blue')
    ax2.set_ylabel('Velocidad Promedio (km/h)', fontsize=11, color='green')
    ax3.set_ylabel('Congestión (%)', fontsize=11, color='red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('Comparación de Métricas de Desempeño', fontsize=13, fontweight='bold')
    
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')
    ax3.tick_params(axis='y', labelcolor='red')
    
    # Leyenda combinada
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
             loc='upper left', fontsize=10)
    
    plt.savefig(os.path.join(output_dirs['comparisons'], 'scenario1_shock_control.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Resultados Sin Control:")
    print(f"    - Tiempo de viaje: {metrics_no_control['avg_travel_time']*60:.2f} min")
    print(f"    - Velocidad promedio: {metrics_no_control['avg_velocity']:.2f} km/h")
    print(f"    - Congestión: {metrics_no_control['congestion_level']:.1f}%")
    
    print(f"\n  Resultados VSL Suave:")
    print(f"    - Tiempo de viaje: {metrics_vsl['avg_travel_time']*60:.2f} min")
    print(f"    - Velocidad promedio: {metrics_vsl['avg_velocity']:.2f} km/h")
    print(f"    - Congestión: {metrics_vsl['congestion_level']:.1f}%")
    print(f"    - Mejora en tiempo: {(1 - metrics_vsl['avg_travel_time']/metrics_no_control['avg_travel_time'])*100:.1f}%")
    
    print(f"\n  Resultados VSL Agresivo:")
    print(f"    - Tiempo de viaje: {metrics_vsl_agg['avg_travel_time']*60:.2f} min")
    print(f"    - Velocidad promedio: {metrics_vsl_agg['avg_velocity']:.2f} km/h")
    print(f"    - Congestión: {metrics_vsl_agg['congestion_level']:.1f}%")
    print(f"    - Mejora en tiempo: {(1 - metrics_vsl_agg['avg_travel_time']/metrics_no_control['avg_travel_time'])*100:.1f}%")
    
    return {
        'no_control': (results_no_control, metrics_no_control),
        'vsl': (results_vsl, metrics_vsl),
        'vsl_aggressive': (results_vsl_agg, metrics_vsl_agg)
    }


def scenario_control_2_predictive(x, t, output_dirs):
    """
    Escenario 2: Control Predictivo vs. Reactivo.
    """
    print("\n" + "="*70)
    print("ESCENARIO 2: Control Predictivo vs. Reactivo")
    print("="*70)
    
    # Condición inicial: pulso gaussiano que generará congestión
    rho0 = gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5)
    
    # Simular sin control
    print("  - Simulando sin control...")
    results_no_control = simulate_with_control(rho0, x, t, control_strategy='none')
    metrics_no_control = compute_performance_metrics(results_no_control)
    
    # Simular con control predictivo
    print("  - Simulando con control predictivo...")
    results_predictive = simulate_with_control(rho0, x, t, control_strategy='predictive', apply_at_step=0)
    metrics_predictive = compute_performance_metrics(results_predictive)
    
    # Simular con control híbrido
    print("  - Simulando con control híbrido...")
    results_hybrid = simulate_with_control(rho0, x, t, control_strategy='hybrid', apply_at_step=0)
    metrics_hybrid = compute_performance_metrics(results_hybrid)
    
    # Visualización
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    scenarios = [
        (results_no_control, "Sin Control"),
        (results_predictive, "Predictivo"),
        (results_hybrid, "Híbrido (VSL+Predictivo)")
    ]
    
    for i, (result, title) in enumerate(scenarios):
        # Densidad
        ax = axes[0, i]
        im = ax.contourf(result['x'], result['t'], result['rho'], levels=20, cmap='YlOrRd')
        ax.set_xlabel('Posición (km)')
        ax.set_ylabel('Tiempo (h)')
        ax.set_title(f'{title}\nDensidad', fontweight='bold')
        plt.colorbar(im, ax=ax, label='ρ (veh/km)')
        
        # Velocidad
        ax = axes[1, i]
        im = ax.contourf(result['x'], result['t'], result['velocity'], levels=20, cmap='RdYlGn')
        ax.set_xlabel('Posición (km)')
        ax.set_ylabel('Tiempo (h)')
        ax.set_title(f'Velocidad', fontweight='bold')
        plt.colorbar(im, ax=ax, label='v (km/h)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparisons'], 'scenario2_predictive_control.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Comparación de Métricas:")
    print(f"    {'Estrategia':<20} {'Tiempo (min)':<15} {'Velocidad (km/h)':<20} {'Congestión (%)':<15}")
    print(f"    {'-'*70}")
    print(f"    {'Sin Control':<20} {metrics_no_control['avg_travel_time']*60:<15.2f} "
          f"{metrics_no_control['avg_velocity']:<20.2f} {metrics_no_control['congestion_level']:<15.1f}")
    print(f"    {'Predictivo':<20} {metrics_predictive['avg_travel_time']*60:<15.2f} "
          f"{metrics_predictive['avg_velocity']:<20.2f} {metrics_predictive['congestion_level']:<15.1f}")
    print(f"    {'Híbrido':<20} {metrics_hybrid['avg_travel_time']*60:<15.2f} "
          f"{metrics_hybrid['avg_velocity']:<20.2f} {metrics_hybrid['congestion_level']:<15.1f}")
    
    return {
        'no_control': (results_no_control, metrics_no_control),
        'predictive': (results_predictive, metrics_predictive),
        'hybrid': (results_hybrid, metrics_hybrid)
    }


def scenario_control_3_zone_based(x, t, output_dirs):
    """
    Escenario 3: Control por Zonas.
    """
    print("\n" + "="*70)
    print("ESCENARIO 3: Control por Zonas")
    print("="*70)
    
    # Condición inicial: perturbación sinusoidal
    rho0 = sinusoidal_perturbation(x, rho_base=60.0, amplitude=40.0, wavelength=2.0)
    
    # Simular sin control
    print("  - Simulando sin control...")
    results_no_control = simulate_with_control(rho0, x, t, control_strategy='none')
    metrics_no_control = compute_performance_metrics(results_no_control)
    
    # Simular con control por zonas
    print("  - Simulando con control por zonas...")
    results_zone = simulate_with_control(rho0, x, t, control_strategy='zone', apply_at_step=0)
    metrics_zone = compute_performance_metrics(results_zone)
    
    # Visualizar zonas de control
    zone_controller = ZoneBasedControl(x, n_zones=3)
    zones = zone_controller.assign_zones()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Densidad sin control
    ax = axes[0, 0]
    im = ax.contourf(results_no_control['x'], results_no_control['t'], 
                     results_no_control['rho'], levels=20, cmap='YlOrRd')
    ax.set_xlabel('Posición (km)')
    ax.set_ylabel('Tiempo (h)')
    ax.set_title('Sin Control - Densidad', fontweight='bold')
    plt.colorbar(im, ax=ax, label='ρ (veh/km)')
    
    # Densidad con control por zonas
    ax = axes[0, 1]
    im = ax.contourf(results_zone['x'], results_zone['t'], 
                     results_zone['rho'], levels=20, cmap='YlOrRd')
    ax.set_xlabel('Posición (km)')
    ax.set_ylabel('Tiempo (h)')
    ax.set_title('Control por Zonas - Densidad', fontweight='bold')
    
    # Añadir líneas de separación de zonas
    for boundary in zone_controller.zone_boundaries[1:-1]:
        ax.axvline(boundary, color='white', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.colorbar(im, ax=ax, label='ρ (veh/km)')
    
    # Velocidad controlada
    ax = axes[1, 0]
    im = ax.contourf(results_zone['x'], results_zone['t'], 
                     results_zone['v_controlled'], levels=20, cmap='RdYlGn')
    ax.set_xlabel('Posición (km)')
    ax.set_ylabel('Tiempo (h)')
    ax.set_title('Velocidad Controlada por Zonas', fontweight='bold')
    for boundary in zone_controller.zone_boundaries[1:-1]:
        ax.axvline(boundary, color='black', linestyle='--', linewidth=2, alpha=0.5)
    plt.colorbar(im, ax=ax, label='v (km/h)')
    
    # Comparación de métricas
    ax = axes[1, 1]
    metrics_comparison = {
        'Tiempo de Viaje (min)': [
            metrics_no_control['avg_travel_time'] * 60,
            metrics_zone['avg_travel_time'] * 60
        ],
        'Velocidad (km/h)': [
            metrics_no_control['avg_velocity'],
            metrics_zone['avg_velocity']
        ],
        'Congestión (%)': [
            metrics_no_control['congestion_level'],
            metrics_zone['congestion_level']
        ]
    }
    
    x_pos = np.arange(2)
    width = 0.25
    colors = ['blue', 'green', 'red']
    
    for i, (metric, values) in enumerate(metrics_comparison.items()):
        ax.bar(x_pos + i*width, values, width, label=metric, alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Estrategia')
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Métricas', fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(['Sin Control', 'Control por Zonas'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparisons'], 'scenario3_zone_control.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Mejoras con Control por Zonas:")
    improvement_time = (1 - metrics_zone['avg_travel_time']/metrics_no_control['avg_travel_time']) * 100
    improvement_congestion = (1 - metrics_zone['congestion_level']/metrics_no_control['congestion_level']) * 100
    print(f"    - Reducción en tiempo de viaje: {improvement_time:.1f}%")
    print(f"    - Reducción en congestión: {improvement_congestion:.1f}%")
    print(f"    - Aumento en velocidad promedio: {(metrics_zone['avg_velocity']/metrics_no_control['avg_velocity'] - 1)*100:.1f}%")
    
    return {
        'no_control': (results_no_control, metrics_no_control),
        'zone': (results_zone, metrics_zone)
    }


def generate_comprehensive_report(all_results, output_dirs):
    """
    Genera reporte comprehensivo comparando todas las estrategias.
    """
    print("\n" + "="*70)
    print("REPORTE COMPREHENSIVO - ESTRATEGIAS DE CONTROL")
    print("="*70)
    
    # Extraer todas las métricas
    strategy_names = []
    travel_times = []
    velocities = []
    congestion_levels = []
    
    for scenario_name, scenario_results in all_results.items():
        for strategy_name, (results, metrics) in scenario_results.items():
            full_name = f"{scenario_name}_{strategy_name}"
            strategy_names.append(full_name)
            travel_times.append(metrics['avg_travel_time'] * 60)
            velocities.append(metrics['avg_velocity'])
            congestion_levels.append(metrics['congestion_level'])
    
    # Crear figura resumen
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Tiempo de viaje
    ax = axes[0]
    ax.barh(range(len(strategy_names)), travel_times, alpha=0.8, color='steelblue')
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=8)
    ax.set_xlabel('Tiempo de Viaje (min)', fontsize=11)
    ax.set_title('Tiempo de Viaje por Estrategia', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Velocidad promedio
    ax = axes[1]
    ax.barh(range(len(strategy_names)), velocities, alpha=0.8, color='green')
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=8)
    ax.set_xlabel('Velocidad Promedio (km/h)', fontsize=11)
    ax.set_title('Velocidad Promedio por Estrategia', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Nivel de congestión
    ax = axes[2]
    ax.barh(range(len(strategy_names)), congestion_levels, alpha=0.8, color='red')
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=8)
    ax.set_xlabel('Nivel de Congestión (%)', fontsize=11)
    ax.set_title('Congestión por Estrategia', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['figures'], 'comprehensive_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Guardar reporte en texto
    report_file = os.path.join(output_dirs['metrics'], 'control_analysis_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE ANÁLISIS DE ESTRATEGIAS DE CONTROL DE TRÁFICO\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Estrategia':<40} {'Tiempo (min)':<15} {'Velocidad (km/h)':<20} {'Congestión (%)':<15}\n")
        f.write("-"*90 + "\n")
        
        for i, name in enumerate(strategy_names):
            f.write(f"{name:<40} {travel_times[i]:<15.2f} {velocities[i]:<20.2f} {congestion_levels[i]:<15.1f}\n")
    
    print(f"\nReporte guardado en: {report_file}")
    print(f"Gráficas guardadas en: {output_dirs['figures']}")


def main():
    """
    Función principal para ejecutar análisis de control.
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE ESTRATEGIAS DE CONTROL DE TRÁFICO VEHICULAR")
    print("="*80)
    
    # Crear directorios
    output_dirs = create_output_directory()
    
    # Parámetros de simulación
    L = 10.0
    T = 1.0
    dx = 0.1
    dt = 0.01
    
    x = get_spatial_grid(L=L, dx=dx)
    t = get_temporal_grid(T=T, dt=dt)
    
    print(f"\nParámetros:")
    print(f"  - Longitud: L = {L} km")
    print(f"  - Tiempo: T = {T} h")
    print(f"  - dx = {dx} km, dt = {dt} h")
    print(f"  - V_max = {V_MAX} km/h, ρ_max = {RHO_MAX} veh/km")
    
    # Ejecutar escenarios
    all_results = {}
    
    all_results['scenario1'] = scenario_control_1_shock_wave(x, t, output_dirs)
    all_results['scenario2'] = scenario_control_2_predictive(x, t, output_dirs)
    all_results['scenario3'] = scenario_control_3_zone_based(x, t, output_dirs)
    
    # Generar reporte comprehensivo
    generate_comprehensive_report(all_results, output_dirs)
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print(f"Resultados en: {output_dirs['figures']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
