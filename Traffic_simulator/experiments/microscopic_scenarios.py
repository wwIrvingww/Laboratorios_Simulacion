#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escenarios de simulacion para el modelo microscopico (IDM).

Este modulo define escenarios reproducibles para:
  - Flujo estable de trafico
  - Flujo congestionado
  - Flujo mixto
  - Flujo con perturbacion gaussiana
  - Flujo con perturbacion sinusoidal
  - Flujo con dos pulsos
  - Flujo con gradiente lineal

Los resultados se guardan como diagramas espacio-tiempo en la carpeta
results/figures/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Anadir el directorio raiz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.microscopic import MicroscopicModel
from src.solvers.runge_kutta import simulate
from src.visualization.spacetime_diagrams import plot_spacetime_diagram_micro
from src.visualization.figure_navigator import FigureNavigator


def create_output_directory(base_dir='results'):
    """Crea estructura de directorios para resultados."""
    dirs = {
        'figures': os.path.join(base_dir, 'figures', 'microscopic'),
        'metrics': os.path.join(base_dir, 'metrics')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def create_default_params():
    """
    Parametros por defecto para el modelo IDM en los escenarios microscopicos.
    """
    return {
        "v0": 30.0,   # velocidad deseada (m/s)
        "a": 1.2,     # aceleracion maxima (m/s^2)
        "b": 1.5,     # deceleracion comoda (m/s^2)
        "T": 1.5,     # tiempo de reaccion (s)
        "s0": 2.0,    # distancia minima (m)
        "s_min": 2.0  # separacion minima numerica (hard-core)
    }


def run_scenario(scenario_name, n_cars, road_length, final_time, dt, init_velocities_func, output_dirs, show_plots=True):
    """
    Ejecuta un escenario microscopico completo con visualizaciones interactivas.

    Parametros:
        scenario_name: Nombre del escenario
        n_cars: Numero de vehiculos
        road_length: Longitud de la carretera (m)
        final_time: Tiempo total de simulacion (s)
        dt: Paso temporal (s)
        init_velocities_func: Funcion que genera velocidades iniciales
        output_dirs: Diccionario con rutas de salida
        show_plots: Si True, muestra navegador interactivo; si False, solo guarda
    """
    print(f"\n{'='*60}")
    print(f"Ejecutando escenario: {scenario_name}")
    print(f"{'='*60}")

    params = create_default_params()

    # Tiempo y pasos
    t_array = np.arange(0.0, final_time + dt, dt)
    n_steps = len(t_array) - 1

    # Posiciones iniciales: distribucion uniforme en la carretera
    init_positions = np.linspace(0.0, road_length * 0.9, n_cars)

    # Velocidades iniciales
    init_velocities = init_velocities_func(n_cars, params["v0"])

    model = MicroscopicModel(
        n_cars=n_cars,
        road_length=road_length,
        params=params,
        init_positions=init_positions,
        init_velocities=init_velocities,
    )

    print("  - Simulando...")
    # Simulacion con RK4
    positions_record, velocities_record = simulate(
        model,
        dt=dt,
        n_steps=n_steps,
        periodic=False,
        road_length=road_length,
        record=True,
    )

    # Crear subdirectorio para este escenario
    safe_name = scenario_name.replace(' ', '_').replace(':', '').replace('/', '_').lower()
    scenario_dir = os.path.join(output_dirs['figures'], safe_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # Crear UNA SOLA figura con 5 axes para las 5 graficas mas importantes
    print("  - Generando graficas...")
    fig = plt.figure(figsize=(10, 6))
    axes_list = []
    titles_list = []

    # Crear 5 axes individuales con márgenes adecuados
    for i in range(5):
        # Cada axis ocupa todo el espacio, pero solo uno será visible
        ax = fig.add_axes([0.12, 0.15, 0.85, 0.70])
        ax.set_visible(False)
        axes_list.append(ax)

    # 1. Diagrama espacio-tiempo (posiciones)
    ax = axes_list[0]
    for i in range(n_cars):
        ax.plot(t_array, positions_record[:, i], linewidth=1, alpha=0.7, label=f'Vehiculo {i+1}' if i < 5 else '')
    ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Posicion (m)', fontsize=9, fontweight='bold')
    ax.set_title('Diagrama Espacio-Tiempo: Posiciones de Vehiculos', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if n_cars <= 5:
        ax.legend(fontsize=9)
    ax.set_visible(False)
    titles_list.append("Posiciones Vehiculos")

    # 2. Velocidad promedio
    ax = axes_list[1]
    avg_velocity = np.mean(velocities_record, axis=1)
    ax.plot(t_array, avg_velocity, 'b-', linewidth=2.5)
    ax.fill_between(t_array, 0, avg_velocity, alpha=0.3, color='blue')
    ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Velocidad Promedio (m/s)', fontsize=9, fontweight='bold')
    ax.set_title('Velocidad Promedio Temporal', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_visible(False)
    titles_list.append("Velocidad Promedio")

    # 3. Velocidad maxima y minima
    ax = axes_list[2]
    max_velocity = np.max(velocities_record, axis=1)
    min_velocity = np.min(velocities_record, axis=1)
    ax.plot(t_array, max_velocity, 'r-', linewidth=2, label='Maxima')
    ax.plot(t_array, min_velocity, 'b-', linewidth=2, label='Minima')
    ax.fill_between(t_array, min_velocity, max_velocity, alpha=0.2, color='gray')
    ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Velocidad (m/s)', fontsize=9, fontweight='bold')
    ax.set_title('Velocidad Maxima y Minima', fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_visible(False)
    titles_list.append("Vel Max/Min")

    # 4. Distribucion de velocidades (final)
    ax = axes_list[3]
    final_velocities = velocities_record[-1, :]
    ax.hist(final_velocities, bins=max(5, n_cars//4), color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(final_velocities), color='r', linestyle='--', linewidth=2, label=f'Promedio: {np.mean(final_velocities):.2f} m/s')
    ax.set_xlabel('Velocidad (m/s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Numero de Vehiculos', fontsize=9, fontweight='bold')
    ax.set_title('Distribucion de Velocidades (Final)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_visible(False)
    titles_list.append("Distribucion Velocidades")

    # 5. Espaciamiento promedio entre vehiculos
    ax = axes_list[4]
    spacing_array = []
    for t_idx in range(n_steps):
        spacings = np.diff(np.sort(positions_record[t_idx, :]))
        spacing_array.append(np.mean(spacings) if len(spacings) > 0 else 0)
    ax.plot(t_array[:len(spacing_array)], spacing_array, 'g-', linewidth=2.5)
    ax.fill_between(t_array[:len(spacing_array)], 0, spacing_array, alpha=0.3, color='green')
    ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Espaciamiento Promedio (m)', fontsize=9, fontweight='bold')
    ax.set_title('Espaciamiento Promedio entre Vehiculos', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_visible(False)
    titles_list.append("Espaciamiento")

    # Guardar figura completa PRIMERO
    print("  - Guardando graficas...")
    for i, ax in enumerate(axes_list):
        ax.set_visible(True)
    plt.savefig(os.path.join(scenario_dir, 'all_plots.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Guardar diagrama espacio-tiempo tradicional
    print("  - Guardando diagrama espacio-tiempo...")
    plot_spacetime_diagram_micro(
        positions_record,
        t_array,
        filename=os.path.join(scenario_dir, 'spacetime_diagram.png'),
    )

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

        # 1. Diagrama espacio-tiempo (posiciones)
        ax = axes_list_display[0]
        for i in range(n_cars):
            ax.plot(t_array, positions_record[:, i], linewidth=1, alpha=0.7, label=f'Vehiculo {i+1}' if i < 5 else '')
        ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Posicion (m)', fontsize=9, fontweight='bold')
        ax.set_title('Diagrama Espacio-Tiempo: Posiciones de Vehiculos', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if n_cars <= 5:
            ax.legend(fontsize=9)
        ax.set_visible(False)

        # 2. Velocidad promedio
        ax = axes_list_display[1]
        avg_velocity = np.mean(velocities_record, axis=1)
        ax.plot(t_array, avg_velocity, 'b-', linewidth=2.5)
        ax.fill_between(t_array, 0, avg_velocity, alpha=0.3, color='blue')
        ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Velocidad Promedio (m/s)', fontsize=9, fontweight='bold')
        ax.set_title('Velocidad Promedio Temporal', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_visible(False)

        # 3. Velocidad maxima y minima
        ax = axes_list_display[2]
        max_velocity = np.max(velocities_record, axis=1)
        min_velocity = np.min(velocities_record, axis=1)
        ax.plot(t_array, max_velocity, 'r-', linewidth=2, label='Maxima')
        ax.plot(t_array, min_velocity, 'b-', linewidth=2, label='Minima')
        ax.fill_between(t_array, min_velocity, max_velocity, alpha=0.2, color='gray')
        ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Velocidad (m/s)', fontsize=9, fontweight='bold')
        ax.set_title('Velocidad Maxima y Minima', fontsize=10, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_visible(False)

        # 4. Distribucion de velocidades (final)
        ax = axes_list_display[3]
        final_velocities = velocities_record[-1, :]
        ax.hist(final_velocities, bins=max(5, n_cars//4), color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(final_velocities), color='r', linestyle='--', linewidth=2, label=f'Promedio: {np.mean(final_velocities):.2f} m/s')
        ax.set_xlabel('Velocidad (m/s)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Numero de Vehiculos', fontsize=9, fontweight='bold')
        ax.set_title('Distribucion de Velocidades (Final)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_visible(False)

        # 5. Espaciamiento promedio entre vehiculos
        ax = axes_list_display[4]
        spacing_array = []
        for t_idx in range(n_steps):
            spacings = np.diff(np.sort(positions_record[t_idx, :]))
            spacing_array.append(np.mean(spacings) if len(spacings) > 0 else 0)
        ax.plot(t_array[:len(spacing_array)], spacing_array, 'g-', linewidth=2.5)
        ax.fill_between(t_array[:len(spacing_array)], 0, spacing_array, alpha=0.3, color='green')
        ax.set_xlabel('Tiempo (s)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Espaciamiento Promedio (m)', fontsize=9, fontweight='bold')
        ax.set_title('Espaciamiento Promedio entre Vehiculos', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_visible(False)

        print(f"\n  Abriendo navegador interactivo (5 graficas)...")
        navigator = FigureNavigator(axes_list_display, title_prefix=f"Escenario: {scenario_name}")
        navigator.display()

    # Calcular y mostrar metricas
    print(f"\n  Metricas del escenario:")
    print(f"    - Numero de vehiculos: {n_cars}")
    print(f"    - Longitud carretera: {road_length} m")
    print(f"    - Tiempo simulacion: {final_time} s")
    print(f"    - Velocidad promedio inicial: {np.mean(init_velocities):.2f} m/s")
    print(f"    - Velocidad promedio final: {np.mean(velocities_record[-1, :]):.2f} m/s")
    print(f"    - Velocidad maxima alcanzada: {np.max(velocities_record):.2f} m/s")
    print(f"    - Velocidad minima alcanzada: {np.min(velocities_record):.2f} m/s")

    return {
        'positions': positions_record,
        'velocities': velocities_record,
        't': t_array,
        'n_cars': n_cars
    }


def main():
    """
    Ejecuta escenarios microscopicos.

    Parametros de linea de comandos:
      - Sin argumentos: ejecuta TODOS los escenarios
      - --scenario N: ejecuta solo el escenario N (1-7)
      - --silent: ejecuta sin mostrar graficos interactivos (solo guarda archivos)
    """
    import argparse

    parser = argparse.ArgumentParser(description='Simulacion microscopica de trafico vehicular')
    parser.add_argument('--scenario', type=int, default=None, help='Ejecutar escenario especifico (1-7)')
    parser.add_argument('--silent', action='store_true', help='Modo silencioso (sin mostrar graficos)')
    args = parser.parse_args()

    # Determinar si mostrar graficos interactivos
    show_plots = not args.silent

    print("\n" + "="*80)
    print("SIMULACION MICROSCOPICA DE TRAFICO VEHICULAR")
    print("Modelo: Intelligent Driver Model (IDM)")
    print("Metodo numerico: Runge-Kutta 4 (RK4)")
    print("="*80)

    output_dirs = create_output_directory()
    print(f"\nResultados se guardaran en: {output_dirs['figures']}")

    # Parametros comunes
    road_length = 800.0
    final_time = 60.0
    dt = 0.1

    print(f"\nParametros de simulacion:")
    print(f"  - Longitud carretera: {road_length} m")
    print(f"  - Tiempo final: {final_time} s")
    print(f"  - Paso temporal: {dt} s")

    # Definir escenarios con funciones lambda para velocidades iniciales
    scenarios_list = [
        {
            'name': 'Escenario 1: Flujo Libre',
            'n_cars': 10,
            'init_vel_func': lambda n, v0: np.ones(n) * (0.8 * v0)
        },
        {
            'name': 'Escenario 2: Flujo Moderado',
            'n_cars': 20,
            'init_vel_func': lambda n, v0: np.ones(n) * (0.7 * v0)
        },
        {
            'name': 'Escenario 3: Flujo Congestionado',
            'n_cars': 30,
            'init_vel_func': lambda n, v0: np.ones(n) * (0.5 * v0)
        },
        {
            'name': 'Escenario 4: Perturbacion Gaussiana',
            'n_cars': 25,
            'init_vel_func': lambda n, v0: np.ones(n) * (0.7 * v0) + np.random.normal(0, 2, n)
        },
        {
            'name': 'Escenario 5: Perturbacion Sinusoidal',
            'n_cars': 20,
            'init_vel_func': lambda n, v0: (0.7 * v0) + 5 * np.sin(np.linspace(0, 2*np.pi, n))
        },
        {
            'name': 'Escenario 6: Dos Grupos',
            'n_cars': 20,
            'init_vel_func': lambda n, v0: np.concatenate([np.ones(n//2) * (0.9 * v0), np.ones(n - n//2) * (0.4 * v0)])
        },
        {
            'name': 'Escenario 7: Gradiente Lineal',
            'n_cars': 25,
            'init_vel_func': lambda n, v0: np.linspace(0.3 * v0, 0.9 * v0, n)
        }
    ]

    # Determinar que escenarios ejecutar
    if args.scenario is not None:
        if args.scenario < 1 or args.scenario > 7:
            print(f"\nError: Escenario {args.scenario} no valido. Debe estar entre 1 y 7.")
            sys.exit(1)
        scenarios_to_run = [args.scenario]
    else:
        scenarios_to_run = list(range(1, 8))

    print(f"\nEjecutando {len(scenarios_to_run)} escenarios...\n")

    # Ejecutar escenarios
    for scenario_num in scenarios_to_run:
        scenario = scenarios_list[scenario_num - 1]
        print(f"\n[{scenario_num}/7] {scenario['name']}")
        run_scenario(
            scenario['name'],
            scenario['n_cars'],
            road_length,
            final_time,
            dt,
            scenario['init_vel_func'],
            output_dirs,
            show_plots=show_plots
        )

    print("\n" + "="*80)
    print("OK Simulacion microscopica completada")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
