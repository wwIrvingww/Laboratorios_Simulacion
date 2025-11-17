import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.macroscopic import simulate_traffic_flow
from src.utils.parameters import get_spatial_grid, get_temporal_grid
from src.utils.initial_conditions import (
    shock_wave_scenario,
    gaussian_pulse,
    sinusoidal_perturbation,
    two_pulse_scenario,
    uniform_density
)
from src.visualization.animations import animate_macroscopic_traffic


def create_animation_directory():
    anim_dir = os.path.join('results', 'animations', 'macroscopic')
    os.makedirs(anim_dir, exist_ok=True)
    return anim_dir


def generate_animation(scenario_name, rho0, x, t, output_dir, fps=10, subsample=5):
    print(f"\n{'='*60}")
    print(f"Generando animación: {scenario_name}")
    print(f"{'='*60}")

    print("  Ejecutando simulación...")
    results = simulate_traffic_flow(rho0, x, t, boundary='periodic')
    rho = results['rho']

    if subsample > 1:
        rho_sub = rho[::subsample, :]
        t_sub = t[::subsample]
        print(f"  Submuestreado temporal: {len(t)} → {len(t_sub)} frames")
    else:
        rho_sub = rho
        t_sub = t

    print(f"  Generando animación ({len(t_sub)} frames a {fps} fps)...")
    safe_name = scenario_name.replace(' ', '_').replace(':', '').lower()
    filename = os.path.join(output_dir, f'{safe_name}.gif')
    
    anim = animate_macroscopic_traffic(rho_sub, x, t_sub, filename=filename, fps=fps)
    
    print(f"  ✓ Animación guardada: {filename}")

    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"  Tamaño: {size_mb:.2f} MB")
    
    return filename


def main():
    print("\n" + "="*80)
    print("GENERADOR DE ANIMACIONES - MODELO MACROSCÓPICO")
    print("="*80)

    output_dir = create_animation_directory()
    print(f"\nAnimaciones se guardarán en: {output_dir}")

    L = 10.0
    T = 1.0
    dx = 0.2
    dt = 0.01
    
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, T + dt, dt)
    
    print(f"\nParámetros de animación:")
    print(f"  Espacial: dx = {dx} km ({len(x)} puntos)")
    print(f"  Temporal: dt = {dt} h ({len(t)} frames)")
    print(f"  Submuestreo temporal: 5x (cada 5 frames)")
    print(f"  Frames resultantes: ~{len(t)//5}")
    print(f"  FPS: 10")
    print(f"  Duración estimada: ~{len(t)//5/10:.1f} segundos por animación")

    scenarios = [
        {
            'name': 'Escenario 1 Flujo Libre',
            'rho0': uniform_density(x, rho_value=30.0),
            'description': 'Densidad baja uniforme, flujo constante'
        },
        {
            'name': 'Escenario 3 Onda de Choque',
            'rho0': shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0),
            'description': 'Discontinuidad que genera onda de choque'
        },
        {
            'name': 'Escenario 4 Perturbación Gaussiana',
            'rho0': gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5),
            'description': 'Pulso localizado que se dispersa'
        },
        {
            'name': 'Escenario 5 Perturbación Sinusoidal',
            'rho0': sinusoidal_perturbation(x, rho_base=60.0, amplitude=30.0, wavelength=2.0),
            'description': 'Variación periódica que se propaga'
        },
        {
            'name': 'Escenario 6 Dos Pulsos',
            'rho0': two_pulse_scenario(x, x1=3.0, x2=7.0, amplitude1=80.0, amplitude2=100.0, width=0.5),
            'description': 'Interacción entre dos perturbaciones'
        }
    ]
    
    print(f"\nGenerando {len(scenarios)} animaciones...")
    
    generated_files = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario['description']}")
        try:
            filename = generate_animation(
                scenario['name'],
                scenario['rho0'],
                x, t,
                output_dir,
                fps=10,
                subsample=5
            )
            generated_files.append(filename)
        except Exception as e:
            print(f"  ❌ Error generando animación: {e}")
            continue

    print("\n" + "="*80)
    print("GENERACIÓN COMPLETADA")
    print("="*80)
    print(f"\nAnimaciones generadas: {len(generated_files)}/{len(scenarios)}")
    print(f"\nArchivos:")
    for filename in generated_files:
        print(f"  - {os.path.basename(filename)}")
    
    print(f"\nDirectorio: {output_dir}")
    print("\nPuedes abrir los archivos .gif con cualquier navegador o visor de imágenes.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
