import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
from src.visualization.figure_navigator import FigureNavigator, DashboardNavigator


def create_output_directory(base_dir='results'):
    dirs = {
        'figures': os.path.join(base_dir, 'figures', 'macroscopic'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'animations': os.path.join(base_dir, 'animations')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def run_scenario(scenario_name, rho0, x, t, output_dirs, boundary='periodic', show_plots=True):
    print(f"\n{'='*60}")
    print(f"Ejecutando escenario: {scenario_name}")
    print(f"{'='*60}")

    results = simulate_traffic_flow(rho0, x, t, boundary=boundary)
    rho = results['rho']
    flux = results['flux']
    velocity = results['velocity']

    safe_name = scenario_name.replace(' ', '_').replace(':', '').replace('/', '_').lower()
    scenario_dir = os.path.join(output_dirs['figures'], safe_name)
    os.makedirs(scenario_dir, exist_ok=True)

    print("  - Generando graficas...")
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    axes_list = []
    titles_list = []
    ax1 = fig.add_subplot(gs[0, :2])
    cf = ax1.contourf(x, t, rho, levels=15, cmap='coolwarm')
    ax1.set_xlabel('Posicion (km)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Tiempo (h)', fontsize=10, fontweight='bold')
    ax1.set_title('Diagrama Espacio-Tiempo (Densidad)', fontsize=11, fontweight='bold')
    cbar1 = fig.colorbar(cf, ax=ax1, label='Densidad (veh/km)')
    ax1.colorbar = cbar1
    axes_list.append(ax1)
    titles_list.append("Diagrama Espacio-Tiempo")

    ax2 = fig.add_subplot(gs[0, 2])
    density_spatial_avg = np.mean(rho, axis=0)
    ax2.plot(x, density_spatial_avg, 'b-', linewidth=2.5)
    ax2.fill_between(x, 0, density_spatial_avg, alpha=0.3, color='blue')
    ax2.set_xlabel('Posicion (km)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Densidad Prom. (veh/km)', fontsize=10, fontweight='bold')
    ax2.set_title('Densidad Promedio Espacial', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    axes_list.append(ax2)
    titles_list.append("Densidad Espacial")

    ax3 = fig.add_subplot(gs[1, 0])
    avg_velocity = compute_average_velocity(rho)
    ax3.plot(t, avg_velocity, 'g-', linewidth=2)
    ax3.fill_between(t, 0, avg_velocity, alpha=0.3, color='green')
    ax3.set_xlabel('Tiempo (h)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Velocidad Prom. (km/h)', fontsize=10, fontweight='bold')
    ax3.set_title('Velocidad Promedio Temporal', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    axes_list.append(ax3)
    titles_list.append("Velocidad Promedio")

    ax4 = fig.add_subplot(gs[1, 1])
    travel_time = compute_travel_time(rho, x, t)
    ax4.plot(t, travel_time * 60, 'purple', linewidth=2.5)
    ax4.fill_between(t, 0, travel_time * 60, alpha=0.3, color='purple')
    ax4.set_xlabel('Tiempo (h)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Tiempo de Viaje (min)', fontsize=10, fontweight='bold')
    ax4.set_title('Evolucion del Tiempo de Viaje', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    axes_list.append(ax4)
    titles_list.append("Tiempo de Viaje")

    ax5 = fig.add_subplot(gs[1, 2])
    congestion_info = compute_congestion_level(rho, threshold=75.0)
    congestion_frac = congestion_info['congestion_fraction']
    ax5.plot(t, congestion_frac * 100, 'r-', linewidth=2)
    ax5.fill_between(t, 0, congestion_frac * 100, alpha=0.3, color='red')
    ax5.set_xlabel('Tiempo (h)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('% Congestionado', fontsize=10, fontweight='bold')
    ax5.set_title('Metricas de Congestion', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    axes_list.append(ax5)
    titles_list.append("Congestión")

    print("  - Guardando graficas...")
    plt.savefig(os.path.join(scenario_dir, 'all_plots.png'), dpi=150, bbox_inches='tight')

    if show_plots:
        print(f"\n  Abriendo dashboard (todas las metricas visibles)...")
        navigator = DashboardNavigator(axes_list, titles=titles_list, title_prefix=f"Escenario: {scenario_name}")
        navigator.display()
    else:
        plt.close(fig)

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
    rho0 = uniform_density(x, rho_value=30.0)
    return run_scenario("Escenario 1: Flujo Libre", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_2_uniform_congestion(x, t, output_dirs, show_plots=True):
    rho0 = uniform_density(x, rho_value=120.0)
    return run_scenario("Escenario 2: Congestión Uniforme", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_3_shock_wave(x, t, output_dirs, show_plots=True):
    rho0 = shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0)
    return run_scenario("Escenario 3: Onda de Choque", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_4_gaussian_pulse(x, t, output_dirs, show_plots=True):
    rho0 = gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5)
    return run_scenario("Escenario 4: Perturbación Gaussiana", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_5_sinusoidal(x, t, output_dirs, show_plots=True):
    rho0 = sinusoidal_perturbation(x, rho_base=60.0, amplitude=30.0, wavelength=2.0)
    return run_scenario("Escenario 5: Perturbación Sinusoidal", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_6_two_pulses(x, t, output_dirs, show_plots=True):
    rho0 = two_pulse_scenario(x, x1=3.0, x2=7.0, amplitude1=80.0, amplitude2=100.0, width=0.5)
    return run_scenario("Escenario 6: Dos Pulsos", rho0, x, t, output_dirs, show_plots=show_plots)


def scenario_7_linear_gradient(x, t, output_dirs, show_plots=True):
    rho0 = linear_gradient(x, rho_start=20.0, rho_end=120.0)
    return run_scenario("Escenario 7: Gradiente Lineal", rho0, x, t, output_dirs, show_plots=show_plots)


def generate_summary_report(all_metrics, output_dirs):
    print(f"\n{'='*60}")
    print("REPORTE RESUMEN - TODOS LOS ESCENARIOS")
    print(f"{'='*60}\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    scenario_names = [m['scenario_name'] for m in all_metrics]

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

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dirs['figures'], 'summary_comparison.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Reporte resumen guardado en: {summary_path}")

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
    print("\nGenerando diagrama fundamental teórico...")
    
    fundamental = compute_fundamental_diagram()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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
    import argparse

    parser = argparse.ArgumentParser(description='Simulación macroscópica de tráfico vehicular')
    parser.add_argument('--scenario', type=int, default=None, help='Ejecutar escenario específico (1-7)')
    parser.add_argument('--silent', action='store_true', help='Modo silencioso (sin mostrar gráficos)')
    args = parser.parse_args()

    show_plots = not args.silent

    print("\n" + "="*80)
    print("SIMULACIÓN MACROSCÓPICA DE TRÁFICO VEHICULAR")
    print("Modelo: Ecuación de Conservación + Greenshields")
    print("Método numérico: Lax-Friedrichs")
    print("="*80)

    output_dirs = create_output_directory()
    print(f"\nResultados se guardarán en: {output_dirs['figures']}")

    L = 10.0
    T = 1.0
    dx = 0.1
    dt = 0.001

    x = get_spatial_grid(L=L, dx=dx)
    t = get_temporal_grid(T=T, dt=dt)

    print(f"\nParámetros de simulación:")
    print(f"  - Longitud carretera: L = {L} km")
    print(f"  - Tiempo simulación: T = {T} h")
    print(f"  - Espaciamiento: dx = {dx} km ({len(x)} puntos)")
    print(f"  - Paso temporal: dt = {dt} h ({len(t)} pasos)")
    print(f"  - V_max = {V_MAX} km/h")
    print(f"  - ρ_max = {RHO_MAX} veh/km")

    plot_fundamental_diagram_theory(output_dirs)

    scenarios_map = {
        1: ('Flujo Libre', scenario_1_free_flow),
        2: ('Congestión Uniforme', scenario_2_uniform_congestion),
        3: ('Onda de Choque', scenario_3_shock_wave),
        4: ('Perturbación Gaussiana', scenario_4_gaussian_pulse),
        5: ('Perturbación Sinusoidal', scenario_5_sinusoidal),
        6: ('Dos Pulsos', scenario_6_two_pulses),
        7: ('Gradiente Lineal', scenario_7_linear_gradient)
    }

    if args.scenario is not None:
        if args.scenario < 1 or args.scenario > 7:
            print(f"\nError: Escenario {args.scenario} no válido. Debe estar entre 1 y 7.")
            sys.exit(1)
        scenarios_to_run = [args.scenario]
    else:
        scenarios_to_run = list(range(1, 8))

    all_results = []
    all_metrics = []

    for scenario_num in scenarios_to_run:
        name, scenario_func = scenarios_map[scenario_num]
        print(f"\n[{scenario_num}/7] Ejecutando: {name}")
        result = scenario_func(x, t, output_dirs, show_plots=show_plots)
        all_results.append(result)
        all_metrics.append(result['metrics'])

    if len(scenarios_to_run) == 7:
        generate_summary_report(all_metrics, output_dirs)

    print("\n" + "="*80)
    print("SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print(f"Total de escenarios ejecutados: {len(scenarios_to_run)}")
    print(f"Resultados guardados en: {output_dirs['figures']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
