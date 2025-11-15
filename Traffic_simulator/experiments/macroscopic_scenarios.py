"""
Escenarios de Simulación Macroscópica de Tráfico Vehicular.

Este módulo orquesta múltiples escenarios de simulación para el modelo macroscópico,
generando visualizaciones completas y métricas de desempeño para cada caso.

Escenarios implementados:
1. Flujo libre (densidad baja uniforme)
2. Congestión uniforme (densidad alta)
3. Onda de choque (discontinuidad en densidad)
4. Perturbación localizada (pulso gaussiano)
5. Perturbación sinusoidal
6. Dos pulsos interactuantes
7. Gradiente de densidad

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.macroscopic import (
    simulate_traffic_flow,
    compute_fundamental_diagram,
    detect_shock_waves,
    compute_travel_time,
    compute_congestion_level,
    compute_average_velocity,
    compute_average_density,
    compute_total_vehicles
)
from src.utils.parameters import get_spatial_grid, get_temporal_grid, V_MAX, RHO_MAX
from src.utils.initial_conditions import (
    uniform_density,
    gaussian_pulse,
    shock_wave_scenario,
    sinusoidal_perturbation,
    two_pulse_scenario,
    linear_gradient
)
from src.visualization.density_maps import (
    plot_density_heatmap,
    plot_density_snapshots,
    plot_density_evolution,
    plot_flux_density_relation
)
from src.visualization.spacetime_diagrams import (
    plot_spacetime_diagram_macro,
    plot_shockwave_detection,
    plot_characteristic_curves
)
from src.visualization.travel_time_plots import (
    plot_travel_time_evolution,
    plot_average_velocity,
    plot_congestion_metrics
)
from src.visualization.figure_navigator import FigureNavigator


def create_output_directory(base_dir='results'):
    """
    Crea la estructura de directorios para guardar resultados.
    
    Retorna:
        dict: Rutas a los subdirectorios creados
    """
    dirs = {
        'figures': os.path.join(base_dir, 'figures', 'macroscopic'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'animations': os.path.join(base_dir, 'animations')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def run_scenario(scenario_name, rho0, x, t, output_dirs, boundary='periodic', show_plots=True):
    """
    Ejecuta un escenario completo de simulación y genera todas las visualizaciones.

    Parámetros:
        scenario_name (str): Nombre del escenario
        rho0 (np.ndarray): Condición inicial de densidad
        x (np.ndarray): Malla espacial
        t (np.ndarray): Malla temporal
        output_dirs (dict): Diccionario con rutas de salida
        boundary (str): Tipo de condiciones de frontera
        show_plots (bool): Si True, muestra el navegador interactivo; si False, solo guarda

    Retorna:
        dict: Resultados de la simulación y métricas
    """
    print(f"\n{'='*60}")
    print(f"Ejecutando escenario: {scenario_name}")
    print(f"{'='*60}")

    # Simular flujo de tráfico
    results = simulate_traffic_flow(rho0, x, t, boundary=boundary)
    rho = results['rho']
    flux = results['flux']
    velocity = results['velocity']

    # Crear subdirectorio para este escenario (remover caracteres inválidos de Windows)
    safe_name = scenario_name.replace(' ', '_').replace(':', '').replace('/', '_').lower()
    scenario_dir = os.path.join(output_dirs['figures'], safe_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # Crear UNA SOLA figura con 5 axes para las 5 graficas mas importantes
    print("  - Generando graficas...")
    fig = plt.figure(figsize=(10, 6))

    # Crear 5 axes individuales con márgenes adecuados
    axes_list = []
    titles_list = []

    for i in range(5):
        # Cada axis ocupa todo el espacio, pero solo uno será visible
        ax = fig.add_axes([0.12, 0.15, 0.85, 0.70])
        ax.set_visible(False)
        axes_list.append(ax)

    # Llenar cada subplot con su grafica correspondiente
    # 1. Diagrama espacio-tiempo (densidad)
    ax = axes_list[0]
    cf = ax.contourf(x, t, rho, levels=15, cmap='coolwarm')
    ax.set_xlabel('Posicion (km)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Tiempo (h)', fontsize=9, fontweight='bold')
    ax.set_title('Diagrama Espacio-Tiempo (Densidad)', fontsize=10, fontweight='bold')
    cbar1 = fig.colorbar(cf, ax=ax, label='Densidad (veh/km)')
    ax.colorbar = cbar1
    ax.set_visible(False)
    titles_list.append("Diagrama Espacio-Tiempo")

    # 2. Densidad promedio espacial
    ax = axes_list[1]
    density_spatial_avg = np.mean(rho, axis=0)
    ax.plot(x, density_spatial_avg, 'b-', linewidth=2.5)
    ax.fill_between(x, 0, density_spatial_avg, alpha=0.3, color='blue')
    ax.set_xlabel('Posicion (km)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Densidad Promedio (veh/km)', fontsize=9, fontweight='bold')
    ax.set_title('Densidad Promedio Espacial', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_visible(False)
    titles_list.append("Densidad Espacial Promedio")

    # 3. Velocidad promedio
    ax = axes_list[2]
    avg_velocity = compute_average_velocity(rho)
    ax.plot(t, avg_velocity, 'g-', linewidth=2)
    ax.fill_between(t, 0, avg_velocity, alpha=0.3, color='green')
    ax.set_xlabel('Tiempo (h)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Velocidad Promedio (km/h)', fontsize=9, fontweight='bold')
    ax.set_title('Velocidad Promedio Temporal', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_visible(False)
    titles_list.append("Velocidad Promedio")

    # 4. Tiempo de viaje
    ax = axes_list[3]
    travel_time = compute_travel_time(rho, x, t)
    ax.plot(t, travel_time * 60, 'purple', linewidth=2.5)
    ax.fill_between(t, 0, travel_time * 60, alpha=0.3, color='purple')
    ax.set_xlabel('Tiempo (h)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Tiempo de Viaje (min)', fontsize=9, fontweight='bold')
    ax.set_title('Evolucion del Tiempo de Viaje', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_visible(False)
    titles_list.append("Tiempo de Viaje")

    # 5. Metricas de congestión
    ax = axes_list[4]
    congestion_info = compute_congestion_level(rho, threshold=75.0)
    congestion_frac = congestion_info['congestion_fraction']
    ax.plot(t, congestion_frac * 100, 'r-', linewidth=2)
    ax.fill_between(t, 0, congestion_frac * 100, alpha=0.3, color='red')
    ax.set_xlabel('Tiempo (h)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Porcentaje Congestionado (%)', fontsize=9, fontweight='bold')
    ax.set_title('Metricas de Congestion', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_visible(False)
    titles_list.append("Metricas de Congestion")

    # Guardar figura completa PRIMERO
    print("  - Guardando graficas...")
    for i, ax in enumerate(axes_list):
        ax.set_visible(True)
    plt.savefig(os.path.join(scenario_dir, 'all_plots.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # LUEGO mostrar navegador interactivo (si aplica)
    if show_plots:
        # Recrear la figura para mostrar interactivamente
        fig = plt.figure(figsize=(10, 6))
        axes_list_display = []

        # Recrear los 5 axes con márgenes adecuados
        for i in range(5):
            ax = fig.add_axes([0.12, 0.15, 0.85, 0.70])
            ax.set_visible(False)
            axes_list_display.append(ax)

        # 1. Diagrama espacio-tiempo
        ax = axes_list_display[0]
        cf = ax.contourf(x, t, rho, levels=15, cmap='coolwarm')
        ax.set_xlabel('Posicion (km)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Tiempo (h)', fontsize=9, fontweight='bold')
        ax.set_title('Diagrama Espacio-Tiempo', fontsize=10, fontweight='bold')
        cbar1 = fig.colorbar(cf, ax=ax, label='Densidad (veh/km)')
        ax.colorbar = cbar1
        ax.set_visible(False)

        # 2. Densidad promedio espacial
        ax = axes_list_display[1]
        density_spatial_avg = np.mean(rho, axis=0)
        ax.plot(x, density_spatial_avg, 'b-', linewidth=2.5)
        ax.fill_between(x, 0, density_spatial_avg, alpha=0.3, color='blue')
        ax.set_xlabel('Posicion (km)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Densidad Promedio (veh/km)', fontsize=9, fontweight='bold')
        ax.set_title('Densidad Promedio Espacial', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_visible(False)

        # 3. Velocidad promedio
        ax = axes_list_display[2]
        avg_velocity_plot = compute_average_velocity(rho)
        ax.plot(t, avg_velocity_plot, 'g-', linewidth=2)
        ax.fill_between(t, 0, avg_velocity_plot, alpha=0.3, color='green')
        ax.set_xlabel('Tiempo (h)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Velocidad Promedio (km/h)', fontsize=9, fontweight='bold')
        ax.set_title('Velocidad Promedio Temporal', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_visible(False)

        # 4. Tiempo de viaje
        ax = axes_list_display[3]
        travel_time_plot = compute_travel_time(rho, x, t)
        ax.plot(t, travel_time_plot * 60, 'purple', linewidth=2.5)
        ax.fill_between(t, 0, travel_time_plot * 60, alpha=0.3, color='purple')
        ax.set_xlabel('Tiempo (h)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Tiempo de Viaje (min)', fontsize=9, fontweight='bold')
        ax.set_title('Evolucion del Tiempo de Viaje', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_visible(False)

        # 5. Metricas de congestion
        ax = axes_list_display[4]
        congestion_info_plot = compute_congestion_level(rho, threshold=75.0)
        congestion_frac_plot = congestion_info_plot['congestion_fraction']
        ax.plot(t, congestion_frac_plot * 100, 'r-', linewidth=2)
        ax.fill_between(t, 0, congestion_frac_plot * 100, alpha=0.3, color='red')
        ax.set_xlabel('Tiempo (h)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Porcentaje Congestionado (%)', fontsize=9, fontweight='bold')
        ax.set_title('Metricas de Congestion', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_visible(False)

        print(f"\n  Abriendo navegador interactivo (5 graficas)...")
        navigator = FigureNavigator(axes_list_display, title_prefix=f"Escenario: {scenario_name}")
        navigator.display()
    
    # Calcular métricas resumen
    avg_density = compute_average_density(rho, x)
    avg_velocity = compute_average_velocity(rho)
    congestion_info = compute_congestion_level(rho, threshold=75.0)
    total_vehicles = compute_total_vehicles(rho, x)
    travel_time = compute_travel_time(rho, x, t)
    shock_info = detect_shock_waves(rho, x, t)

    metrics = {
        'scenario_name': scenario_name,
        'avg_density_initial': avg_density[0],
        'avg_density_final': avg_density[-1],
        'avg_velocity_initial': avg_velocity[0],
        'avg_velocity_final': avg_velocity[-1],
        'travel_time_initial': travel_time[0],
        'travel_time_final': travel_time[-1],
        'max_congestion_fraction': np.max(congestion_info['congestion_fraction']),
        'avg_congestion_fraction': np.mean(congestion_info['congestion_fraction']),
        'total_vehicles_initial': total_vehicles[0],
        'total_vehicles_final': total_vehicles[-1],
        'shock_waves_detected': np.sum([len(pos) for pos in shock_info['shock_positions']])
    }
    
    print(f"\n  Métricas del escenario:")
    print(f"    - Densidad promedio inicial: {metrics['avg_density_initial']:.2f} veh/km")
    print(f"    - Densidad promedio final: {metrics['avg_density_final']:.2f} veh/km")
    print(f"    - Velocidad promedio inicial: {metrics['avg_velocity_initial']:.2f} km/h")
    print(f"    - Velocidad promedio final: {metrics['avg_velocity_final']:.2f} km/h")
    print(f"    - Tiempo de viaje inicial: {metrics['travel_time_initial']:.4f} h")
    print(f"    - Tiempo de viaje final: {metrics['travel_time_final']:.4f} h")
    print(f"    - Fracción máxima congestionada: {metrics['max_congestion_fraction']:.2%}")
    print(f"    - Ondas de choque detectadas: {metrics['shock_waves_detected']}")
    
    return {
        'results': results,
        'metrics': metrics,
        'shock_info': shock_info,
        'congestion_info': congestion_info
    }


def scenario_1_free_flow(x, t, output_dirs, show_plots=True):
    """Escenario 1: Flujo libre con densidad baja uniforme."""
    rho0 = uniform_density(x, rho_value=30.0)
    return run_scenario("Escenario 1: Flujo Libre", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_2_uniform_congestion(x, t, output_dirs, show_plots=True):
    """Escenario 2: Congestión uniforme con densidad alta."""
    rho0 = uniform_density(x, rho_value=120.0)
    return run_scenario("Escenario 2: Congestión Uniforme", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_3_shock_wave(x, t, output_dirs, show_plots=True):
    """Escenario 3: Formación de onda de choque."""
    rho0 = shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0)
    return run_scenario("Escenario 3: Onda de Choque", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_4_gaussian_pulse(x, t, output_dirs, show_plots=True):
    """Escenario 4: Perturbación localizada (pulso gaussiano)."""
    rho0 = gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5)
    return run_scenario("Escenario 4: Perturbación Gaussiana", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_5_sinusoidal(x, t, output_dirs, show_plots=True):
    """Escenario 5: Perturbación sinusoidal."""
    rho0 = sinusoidal_perturbation(x, rho_base=60.0, amplitude=30.0, wavelength=2.0)
    return run_scenario("Escenario 5: Perturbación Sinusoidal", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_6_two_pulses(x, t, output_dirs, show_plots=True):
    """Escenario 6: Dos pulsos interactuantes."""
    rho0 = two_pulse_scenario(x, x1=3.0, x2=7.0, amplitude1=80.0, amplitude2=100.0, width=0.5)
    return run_scenario("Escenario 6: Dos Pulsos", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_7_linear_gradient(x, t, output_dirs, show_plots=True):
    """Escenario 7: Gradiente lineal de densidad."""
    rho0 = linear_gradient(x, rho_start=20.0, rho_end=120.0)
    return run_scenario("Escenario 7: Gradiente Lineal", rho0, x, t, output_dirs, show_plots=show_plots)


def generate_summary_report(all_metrics, output_dirs):
    """
    Genera un reporte resumen comparando todos los escenarios.
    
    Parámetros:
        all_metrics (list): Lista de diccionarios con métricas de cada escenario
        output_dirs (dict): Rutas de salida
    """
    print(f"\n{'='*60}")
    print("REPORTE RESUMEN - TODOS LOS ESCENARIOS")
    print(f"{'='*60}\n")
    
    # Crear tabla comparativa
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    scenario_names = [m['scenario_name'] for m in all_metrics]
    
    # 1. Comparación de densidad promedio
    ax = axes[0, 0]
    initial_densities = [m['avg_density_initial'] for m in all_metrics]
    final_densities = [m['avg_density_final'] for m in all_metrics]
    x_pos = np.arange(len(scenario_names))
    width = 0.35
    ax.bar(x_pos - width/2, initial_densities, width, label='Inicial', alpha=0.8)
    ax.bar(x_pos + width/2, final_densities, width, label='Final', alpha=0.8)
    ax.set_ylabel('Densidad Promedio (veh/km)', fontsize=9)
    ax.set_title('Comparación de Densidad Promedio', fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"E{i+1}" for i in range(len(scenario_names))], rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Comparación de velocidad promedio
    ax = axes[0, 1]
    initial_velocities = [m['avg_velocity_initial'] for m in all_metrics]
    final_velocities = [m['avg_velocity_final'] for m in all_metrics]
    ax.bar(x_pos - width/2, initial_velocities, width, label='Inicial', alpha=0.8)
    ax.bar(x_pos + width/2, final_velocities, width, label='Final', alpha=0.8)
    ax.set_ylabel('Velocidad Promedio (km/h)', fontsize=9)
    ax.set_title('Comparación de Velocidad Promedio', fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"E{i+1}" for i in range(len(scenario_names))], rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Comparación de tiempo de viaje
    ax = axes[1, 0]
    initial_travel = [m['travel_time_initial'] * 60 for m in all_metrics]  # Convertir a minutos
    final_travel = [m['travel_time_final'] * 60 for m in all_metrics]
    ax.bar(x_pos - width/2, initial_travel, width, label='Inicial', alpha=0.8)
    ax.bar(x_pos + width/2, final_travel, width, label='Final', alpha=0.8)
    ax.set_ylabel('Tiempo de Viaje (min)', fontsize=9)
    ax.set_title('Comparación de Tiempo de Viaje', fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"E{i+1}" for i in range(len(scenario_names))], rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Nivel de congestión y ondas de choque
    ax = axes[1, 1]
    congestion = [m['max_congestion_fraction'] * 100 for m in all_metrics]
    shockwaves = [m['shock_waves_detected'] for m in all_metrics]
    
    ax2 = ax.twinx()
    bar1 = ax.bar(x_pos - width/2, congestion, width, label='Congestión (%)', alpha=0.8, color='orange')
    bar2 = ax2.bar(x_pos + width/2, shockwaves, width, label='Ondas de Choque', alpha=0.8, color='red')
    
    ax.set_ylabel('Congestión Máxima (%)', fontsize=9, color='orange')
    ax2.set_ylabel('Ondas de Choque Detectadas', fontsize=9, color='red')
    ax.set_title('Congestión y Ondas de Choque', fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"E{i+1}" for i in range(len(scenario_names))], rotation=0)
    ax.tick_params(axis='y', labelcolor='orange')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Leyenda combinada
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dirs['figures'], 'summary_comparison.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Reporte resumen guardado en: {summary_path}")
    
    # Guardar métricas en archivo de texto
    metrics_file = os.path.join(output_dirs['metrics'], 'macroscopic_summary.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE RESUMEN - SIMULACIÓN MACROSCÓPICA DE TRÁFICO\n")
        f.write("="*80 + "\n\n")
        
        for i, metrics in enumerate(all_metrics, 1):
            f.write(f"\n{metrics['scenario_name']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Densidad promedio inicial:    {metrics['avg_density_initial']:.2f} veh/km\n")
            f.write(f"  Densidad promedio final:      {metrics['avg_density_final']:.2f} veh/km\n")
            f.write(f"  Velocidad promedio inicial:   {metrics['avg_velocity_initial']:.2f} km/h\n")
            f.write(f"  Velocidad promedio final:     {metrics['avg_velocity_final']:.2f} km/h\n")
            f.write(f"  Tiempo de viaje inicial:      {metrics['travel_time_initial']*60:.2f} min\n")
            f.write(f"  Tiempo de viaje final:        {metrics['travel_time_final']*60:.2f} min\n")
            f.write(f"  Fracción máxima congestionada:{metrics['max_congestion_fraction']:.2%}\n")
            f.write(f"  Fracción promedio congestionada:{metrics['avg_congestion_fraction']:.2%}\n")
            f.write(f"  Vehículos totales (inicial):  {metrics['total_vehicles_initial']:.0f}\n")
            f.write(f"  Vehículos totales (final):    {metrics['total_vehicles_final']:.0f}\n")
            f.write(f"  Ondas de choque detectadas:   {metrics['shock_waves_detected']}\n")
    
    print(f"Métricas detalladas guardadas en: {metrics_file}")


def plot_fundamental_diagram_theory(output_dirs):
    """
    Genera el diagrama fundamental teórico del modelo de Greenshields.
    """
    print("\nGenerando diagrama fundamental teórico...")
    
    fundamental = compute_fundamental_diagram()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Flujo vs Densidad
    ax = axes[0]
    ax.plot(fundamental['rho'], fundamental['flux'], 'b-', linewidth=2)
    ax.axvline(fundamental['rho_critical'], color='r', linestyle='--', 
               label=f'ρ_crítica = {fundamental["rho_critical"]:.1f} veh/km')
    ax.axhline(fundamental['flux_max'], color='g', linestyle='--',
               label=f'q_max = {fundamental["flux_max"]:.1f} veh/h')
    ax.set_xlabel('Densidad ρ (veh/km)', fontsize=10)
    ax.set_ylabel('Flujo q (veh/h)', fontsize=10)
    ax.set_title('Diagrama Fundamental: Flujo vs Densidad', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Velocidad vs Densidad
    ax = axes[1]
    ax.plot(fundamental['rho'], fundamental['velocity'], 'r-', linewidth=2)
    ax.axvline(fundamental['rho_critical'], color='r', linestyle='--',
               label=f'ρ_crítica = {fundamental["rho_critical"]:.1f} veh/km')
    ax.set_xlabel('Densidad ρ (veh/km)', fontsize=10)
    ax.set_ylabel('Velocidad v (km/h)', fontsize=10)
    ax.set_title('Relación Velocidad-Densidad (Greenshields)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    diagram_path = os.path.join(output_dirs['figures'], 'fundamental_diagram_theory.png')
    plt.savefig(diagram_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Diagrama fundamental teórico guardado en: {diagram_path}")


def main():
    """
    Función principal que ejecuta escenarios macroscópicos.

    Parámetros de línea de comandos:
      - Sin argumentos: ejecuta TODOS los escenarios
      - --scenario N: ejecuta solo el escenario N (1-7)
      - --silent: ejecuta sin mostrar gráficos interactivos (solo guarda archivos)
    """
    import argparse

    parser = argparse.ArgumentParser(description='Simulación macroscópica de tráfico vehicular')
    parser.add_argument('--scenario', type=int, default=None, help='Ejecutar escenario específico (1-7)')
    parser.add_argument('--silent', action='store_true', help='Modo silencioso (sin mostrar gráficos)')
    args = parser.parse_args()

    # Determinar si mostrar gráficos interactivos
    show_plots = not args.silent

    print("\n" + "="*80)
    print("SIMULACIÓN MACROSCÓPICA DE TRÁFICO VEHICULAR")
    print("Modelo: Ecuación de Conservación + Greenshields")
    print("Método numérico: Lax-Friedrichs")
    print("="*80)

    # Crear estructura de directorios
    output_dirs = create_output_directory()
    print(f"\nResultados se guardarán en: {output_dirs['figures']}")

    # Definir parámetros de discretización
    L = 10.0    # Longitud de la carretera (km)
    T = 1.0     # Tiempo total de simulación (h)
    dx = 0.1    # Espaciamiento espacial (km)
    dt = 0.001  # Paso temporal (h)

    x = get_spatial_grid(L=L, dx=dx)
    t = get_temporal_grid(T=T, dt=dt)

    print(f"\nParámetros de simulación:")
    print(f"  - Longitud carretera: L = {L} km")
    print(f"  - Tiempo simulación: T = {T} h")
    print(f"  - Espaciamiento: dx = {dx} km ({len(x)} puntos)")
    print(f"  - Paso temporal: dt = {dt} h ({len(t)} pasos)")
    print(f"  - V_max = {V_MAX} km/h")
    print(f"  - ρ_max = {RHO_MAX} veh/km")

    # Generar diagrama fundamental teórico
    plot_fundamental_diagram_theory(output_dirs)

    # Definir todos los escenarios
    scenarios_map = {
        1: ('Flujo Libre', scenario_1_free_flow),
        2: ('Congestión Uniforme', scenario_2_uniform_congestion),
        3: ('Onda de Choque', scenario_3_shock_wave),
        4: ('Perturbación Gaussiana', scenario_4_gaussian_pulse),
        5: ('Perturbación Sinusoidal', scenario_5_sinusoidal),
        6: ('Dos Pulsos', scenario_6_two_pulses),
        7: ('Gradiente Lineal', scenario_7_linear_gradient)
    }

    # Determinar qué escenarios ejecutar
    if args.scenario is not None:
        if args.scenario < 1 or args.scenario > 7:
            print(f"\nError: Escenario {args.scenario} no válido. Debe estar entre 1 y 7.")
            sys.exit(1)
        scenarios_to_run = [args.scenario]
    else:
        scenarios_to_run = list(range(1, 8))

    # Ejecutar escenarios
    all_results = []
    all_metrics = []

    for scenario_num in scenarios_to_run:
        name, scenario_func = scenarios_map[scenario_num]
        print(f"\n[{scenario_num}/7] Ejecutando: {name}")
        result = scenario_func(x, t, output_dirs, show_plots=show_plots)
        all_results.append(result)
        all_metrics.append(result['metrics'])

    # Generar reporte resumen (solo si se ejecutaron todos)
    if len(scenarios_to_run) == 7:
        generate_summary_report(all_metrics, output_dirs)

    print("\n" + "="*80)
    print("SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print(f"Total de escenarios ejecutados: {len(scenarios_to_run)}")
    print(f"Resultados guardados en: {output_dirs['figures']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
