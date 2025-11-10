"""
Ejemplo de uso del modelo macroscópico con mejor estabilidad numérica.

Este script demuestra cómo ejecutar un escenario individual con parámetros
que satisfacen la condición CFL para mayor estabilidad.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.macroscopic import (
    simulate_traffic_flow,
    compute_travel_time,
    compute_average_velocity,
    detect_shock_waves
)
from src.utils.parameters import V_MAX, RHO_MAX
from src.utils.initial_conditions import shock_wave_scenario
from src.visualization.density_maps import plot_density_heatmap
from src.visualization.spacetime_diagrams import plot_spacetime_diagram_macro


def example_stable_simulation():
    """
    Ejecuta una simulación con parámetros que satisfacen la condición CFL.
    """
    print("="*80)
    print("EJEMPLO: Simulación Estable con CFL < 1.0")
    print("="*80)
    
    # Parámetros de discretización (CFL seguro)
    L = 10.0        # km
    dx = 0.1        # km
    T = 1.0         # h
    dt = 0.001      # h (3.6 segundos) → CFL = 1.0
    
    # Crear mallas
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, T + dt, dt)
    
    print(f"\nParámetros:")
    print(f"  Espacial: dx = {dx} km ({len(x)} puntos)")
    print(f"  Temporal: dt = {dt} h = {dt*3600:.1f} s ({len(t)} pasos)")
    print(f"  CFL = V_max * dt / dx = {V_MAX * dt / dx:.3f}")
    
    # Condición inicial: onda de choque
    print(f"\nCondición inicial: Onda de choque en x = 5 km")
    rho0 = shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0)
    
    print(f"  ρ aguas arriba: 140 veh/km")
    print(f"  ρ aguas abajo: 30 veh/km")
    
    # Simular
    print(f"\nSimulando... (esto puede tomar 1-2 minutos)")
    results = simulate_traffic_flow(rho0, x, t, boundary='periodic')
    
    rho = results['rho']
    flux = results['flux']
    velocity = results['velocity']
    
    print(f"✓ Simulación completada")
    
    # Analizar resultados
    print(f"\n" + "-"*80)
    print("ANÁLISIS DE RESULTADOS")
    print("-"*80)
    
    # Densidad promedio
    avg_rho = np.mean(rho, axis=1)
    print(f"\nDensidad promedio:")
    print(f"  Inicial: {avg_rho[0]:.2f} veh/km")
    print(f"  Final: {avg_rho[-1]:.2f} veh/km")
    print(f"  Variación: {abs(avg_rho[-1] - avg_rho[0]):.2f} veh/km")
    
    # Velocidad promedio
    avg_vel = compute_average_velocity(rho)
    print(f"\nVelocidad promedio:")
    print(f"  Inicial: {avg_vel[0]:.2f} km/h")
    print(f"  Final: {avg_vel[-1]:.2f} km/h")
    
    # Tiempo de viaje
    travel_time = compute_travel_time(rho, x, t)
    print(f"\nTiempo de viaje:")
    print(f"  Inicial: {travel_time[0]*60:.2f} min")
    print(f"  Final: {travel_time[-1]*60:.2f} min")
    
    # Ondas de choque
    shock_info = detect_shock_waves(rho, x, t, threshold_gradient=50.0)
    total_shocks = sum(len(pos) for pos in shock_info['shock_positions'])
    print(f"\nOndas de choque detectadas: {total_shocks}")
    
    # Verificar estabilidad (no debe haber NaN o valores negativos)
    has_nan = np.any(np.isnan(rho))
    has_negative = np.any(rho < 0)
    has_overflow = np.any(rho > RHO_MAX * 1.5)  # Permitir 50% de overrun
    
    print(f"\nVerificación de estabilidad:")
    print(f"  NaN detectados: {'❌ SÍ' if has_nan else '✓ NO'}")
    print(f"  Valores negativos: {'❌ SÍ' if has_negative else '✓ NO'}")
    print(f"  Overflow extremo: {'❌ SÍ' if has_overflow else '✓ NO'}")
    
    if not (has_nan or has_negative or has_overflow):
        print(f"\n✓ La simulación es numéricamente estable")
    else:
        print(f"\n⚠ Advertencia: Se detectaron inestabilidades numéricas")
    
    # Crear directorio de salida
    output_dir = os.path.join('results', 'examples')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar visualizaciones
    print(f"\n" + "-"*80)
    print("GENERANDO VISUALIZACIONES")
    print("-"*80)
    
    # 1. Mapa de calor
    print(f"  - Mapa de calor de densidad...")
    fig1, _ = plot_density_heatmap(
        rho, x, t,
        title="Onda de Choque - Simulación Estable (CFL = 1.0)",
        filename=os.path.join(output_dir, 'stable_density_heatmap.png')
    )
    plt.close(fig1)
    
    # 2. Diagrama espacio-tiempo
    print(f"  - Diagrama espacio-tiempo...")
    fig2, _ = plot_spacetime_diagram_macro(
        rho, x, t, levels=20,
        filename=os.path.join(output_dir, 'stable_spacetime_diagram.png')
    )
    plt.close(fig2)
    
    # 3. Evolución de métricas
    print(f"  - Evolución de métricas...")
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Densidad promedio vs tiempo
    axes[0, 0].plot(t, avg_rho, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Tiempo (h)', fontsize=11)
    axes[0, 0].set_ylabel('Densidad Promedio (veh/km)', fontsize=11)
    axes[0, 0].set_title('Densidad Promedio', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocidad promedio vs tiempo
    axes[0, 1].plot(t, avg_vel, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Tiempo (h)', fontsize=11)
    axes[0, 1].set_ylabel('Velocidad Promedio (km/h)', fontsize=11)
    axes[0, 1].set_title('Velocidad Promedio', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Tiempo de viaje vs tiempo
    axes[1, 0].plot(t, travel_time * 60, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Tiempo (h)', fontsize=11)
    axes[1, 0].set_ylabel('Tiempo de Viaje (min)', fontsize=11)
    axes[1, 0].set_title('Tiempo de Viaje', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Flujo vs densidad (diagrama fundamental)
    rho_flat = rho.flatten()
    flux_flat = flux.flatten()
    axes[1, 1].scatter(rho_flat, flux_flat, alpha=0.3, s=5, label='Simulación')
    rho_theory = np.linspace(0, RHO_MAX, 100)
    flux_theory = rho_theory * V_MAX * (1 - rho_theory / RHO_MAX)
    axes[1, 1].plot(rho_theory, flux_theory, 'r-', linewidth=2, label='Teoría')
    axes[1, 1].set_xlabel('Densidad (veh/km)', fontsize=11)
    axes[1, 1].set_ylabel('Flujo (veh/h)', fontsize=11)
    axes[1, 1].set_title('Diagrama Fundamental', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'stable_metrics_evolution.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"\n✓ Visualizaciones guardadas en: {output_dir}")
    print(f"="*80)
    print(f"EJEMPLO COMPLETADO")
    print(f"="*80)
    
    return results


def example_adaptive_simulation():
    """
    Demuestra el uso del modo adaptativo para CFL seguro automático.
    """
    print("\n" + "="*80)
    print("EJEMPLO: Simulación con Paso Temporal Adaptativo")
    print("="*80)
    
    # Parámetros
    L = 10.0
    dx = 0.1
    T_final = 1.0
    cfl_target = 0.8
    
    x = np.arange(0, L + dx, dx)
    
    print(f"\nParámetros:")
    print(f"  Espacial: dx = {dx} km")
    print(f"  Tiempo final: T = {T_final} h")
    print(f"  CFL objetivo: {cfl_target}")
    
    # Condición inicial
    rho0 = shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0)
    
    print(f"\nSimulando con paso temporal adaptativo...")
    
    # El modo adaptativo calcula dt automáticamente
    results = simulate_traffic_flow(
        rho0, x, t=T_final,  # Nota: t es un escalar (tiempo final)
        boundary='periodic',
        adaptive=True,
        cfl_target=cfl_target
    )
    
    t_adaptive = results['t']
    dt_adaptive = t_adaptive[1] - t_adaptive[0]
    
    print(f"\n✓ Simulación completada")
    print(f"  dt calculado: {dt_adaptive:.6f} h = {dt_adaptive*3600:.2f} s")
    print(f"  Pasos temporales: {len(t_adaptive)}")
    print(f"  CFL efectivo: {V_MAX * dt_adaptive / dx:.3f}")
    
    # Verificar estabilidad
    rho = results['rho']
    has_nan = np.any(np.isnan(rho))
    
    print(f"\nEstabilidad: {'✓ Estable' if not has_nan else '❌ Inestable'}")
    
    print(f"="*80)
    
    return results


if __name__ == "__main__":
    # Ejecutar ejemplo con CFL fijo
    results_stable = example_stable_simulation()
    
    # Ejecutar ejemplo con CFL adaptativo
    results_adaptive = example_adaptive_simulation()
    
    print("\n" + "="*80)
    print("AMBOS EJEMPLOS COMPLETADOS EXITOSAMENTE")
    print("Consulte el directorio 'results/examples/' para las visualizaciones")
    print("="*80 + "\n")
