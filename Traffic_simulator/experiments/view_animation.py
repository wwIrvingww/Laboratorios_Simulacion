"""
Visor Interactivo de Animaciones Macroscópicas.

Este script muestra una animación interactiva en una ventana de matplotlib
para visualizar la evolución del tráfico en tiempo real.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.macroscopic import simulate_traffic_flow
from src.models.microscopic import MicroscopicModel
from src.solvers.runge_kutta import simulate as simulate_microscopic
from src.utils.parameters import get_spatial_grid, get_temporal_grid
from src.utils.initial_conditions import (
    uniform_density,
    shock_wave_scenario,
    gaussian_pulse,
    sinusoidal_perturbation,
    two_pulse_scenario
)


def create_interactive_animation(rho, x, t, title="Simulación de Tráfico Macroscópico"):
    """
    Crea y muestra una animación interactiva del modelo macroscópico.
    
    Parámetros:
        rho (np.ndarray): Densidad ρ(x,t) con shape (n_time, n_space)
        x (np.ndarray): Malla espacial (km)
        t (np.ndarray): Malla temporal (h)
        title (str): Título de la animación
    """
    # Crear figura con dos subplots
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])  # Gráfica de densidad superior (ancha)
    ax2 = fig.add_subplot(gs[1, 0])  # Diagrama espacio-tiempo
    ax3 = fig.add_subplot(gs[1, 1])  # Velocidad promedio
    
    # Configurar subplot 1: Perfil de densidad
    line_density, = ax1.plot([], [], 'b-', linewidth=2.5, label='Densidad')
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(0, 160)
    ax1.set_xlabel('Posición (km)', fontsize=12)
    ax1.set_ylabel('Densidad (veh/km)', fontsize=12)
    ax1.set_title('Perfil de Densidad Vehicular', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=75, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='ρ crítica')
    ax1.legend(loc='upper right')
    
    # Área sombreada bajo la curva
    fill = ax1.fill_between([], [], alpha=0.3, color='blue')
    
    # Texto con información del tiempo y métricas
    info_text = ax1.text(0.02, 0.97, '', transform=ax1.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        family='monospace')
    
    # Configurar subplot 2: Diagrama espacio-tiempo
    im = ax2.imshow(rho.T, aspect='auto', origin='lower', cmap='RdYlGn_r',
                   extent=[t[0], t[-1], x[0], x[-1]], vmin=0, vmax=150)
    ax2.set_xlabel('Tiempo (h)', fontsize=11)
    ax2.set_ylabel('Posición (km)', fontsize=11)
    ax2.set_title('Diagrama Espacio-Tiempo', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2, label='Densidad (veh/km)')
    
    # Línea vertical que indica el tiempo actual
    vline = ax2.axvline(x=t[0], color='white', linewidth=2.5, linestyle='-', alpha=0.9)
    
    # Configurar subplot 3: Velocidad promedio
    velocity = 100 * (1 - rho / 150)  # Modelo de Greenshields
    avg_velocity = np.mean(velocity, axis=1)
    
    line_velocity, = ax3.plot([], [], 'g-', linewidth=2.5, label='Velocidad promedio')
    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(0, 110)
    ax3.set_xlabel('Tiempo (h)', fontsize=11)
    ax3.set_ylabel('Velocidad (km/h)', fontsize=11)
    ax3.set_title('Evolución de Velocidad Promedio', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=100, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='v máx')
    ax3.legend(loc='upper right')
    
    # Marcador del punto actual
    marker, = ax3.plot([], [], 'ro', markersize=10, label='Actual')
    
    # Título principal
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    
    # Variables para la animación
    frames_count = len(t)
    
    def init():
        """Inicializa la animación."""
        line_density.set_data([], [])
        line_velocity.set_data([], [])
        marker.set_data([], [])
        info_text.set_text('')
        return line_density, line_velocity, marker, vline, info_text
    
    def animate(frame):
        """Actualiza cada frame de la animación."""
        # Actualizar perfil de densidad
        line_density.set_data(x, rho[frame, :])
        
        # Actualizar área sombreada
        # Eliminar todas las colecciones (PolyCollection del fill_between)
        while len(ax1.collections) > 0:
            ax1.collections[0].remove()
        ax1.fill_between(x, 0, rho[frame, :], alpha=0.3, color='blue')
        
        # Calcular métricas actuales
        current_density = rho[frame, :]
        avg_rho = np.mean(current_density)
        avg_v = np.mean(100 * (1 - current_density / 150))
        congestion_fraction = np.sum(current_density > 75) / len(current_density) * 100
        
        # Actualizar texto informativo
        info_text.set_text(
            f't = {t[frame]:.3f} h ({t[frame]*60:.1f} min)\n'
            f'ρ promedio = {avg_rho:.1f} veh/km\n'
            f'v promedio = {avg_v:.1f} km/h\n'
            f'Congestión = {congestion_fraction:.1f}%'
        )
        
        # Actualizar línea vertical en diagrama espacio-tiempo
        vline.set_xdata([t[frame], t[frame]])
        
        # Actualizar gráfica de velocidad
        line_velocity.set_data(t[:frame+1], avg_velocity[:frame+1])
        marker.set_data([t[frame]], [avg_velocity[frame]])
        
        return line_density, line_velocity, marker, vline, info_text
    
    # Crear animación
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=frames_count,
        interval=100,  # 100 ms entre frames = 10 fps
        blit=False,
        repeat=True
    )
    
    return anim


def select_model():
    """Permite al usuario seleccionar un modelo para visualizar."""
    print("\n" + "="*70)
    print("VISOR INTERACTIVO DE ANIMACIONES")
    print("="*70)
    print("\nSeleccionar modelo:")
    print("  1. Macroscopico (Ecuacion de Conservacion + Greenshields)")
    print("  2. Microscopico (Intelligent Driver Model - IDM)")
    print()

    choice = input("Seleccione modelo (1-2) [1]: ").strip()
    if not choice or choice not in ['1', '2']:
        choice = '1'

    return choice


def select_scenario(model='macro'):
    """
    Permite al usuario seleccionar un escenario para visualizar.

    Parametros:
        model: 'macro' o 'micro' para indicar que escenarios mostrar
    """
    if model == 'macro':
        print("\nEscenarios disponibles (Macroscopico):")
        print("  1. Flujo Libre (densidad baja uniforme)")
        print("  2. Onda de Choque (discontinuidad)")
        print("  3. Perturbacion Gaussiana (pulso localizado)")
        print("  4. Perturbacion Sinusoidal (variacion periodica)")
        print("  5. Dos Pulsos (interaccion)")
    else:  # micro
        print("\nEscenarios disponibles (Microscopico):")
        print("  1. Flujo Libre")
        print("  2. Flujo Moderado")
        print("  3. Flujo Congestionado")
        print("  4. Perturbacion Gaussiana")
        print("  5. Perturbacion Sinusoidal")
        print("  6. Dos Grupos")
        print("  7. Gradiente Lineal")
    print()

    max_choice = '5' if model == 'macro' else '7'
    choice = input(f"Seleccione un escenario (1-{max_choice}) [1]: ").strip()
    if not choice:
        choice = '1'

    return choice


def main():
    """Función principal para visualizar animación interactiva."""

    # Seleccionar modelo
    model_choice = select_model()
    model = 'macro' if model_choice == '1' else 'micro'

    if model == 'macro':
        # MODELO MACROSCÓPICO
        # Configurar parámetros
        L = 10.0
        T = 1.0
        dx = 0.1
        dt = 0.01

        x = get_spatial_grid(L=L, dx=dx)
        t = get_temporal_grid(T=T, dt=dt)

        # Seleccionar escenario
        choice = select_scenario(model='macro')

        # Definir condición inicial según elección
        scenarios = {
            '1': ('Flujo Libre', uniform_density(x, rho_value=30.0)),
            '2': ('Onda de Choque', shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0)),
            '3': ('Perturbacion Gaussiana', gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5)),
            '4': ('Perturbacion Sinusoidal', sinusoidal_perturbation(x, rho_base=60.0, amplitude=30.0, wavelength=2.0)),
            '5': ('Dos Pulsos', two_pulse_scenario(x, x1=3.0, x2=7.0, amplitude1=80.0, amplitude2=100.0, width=0.5))
        }

        if choice in scenarios:
            scenario_name, rho0 = scenarios[choice]
        else:
            print("Opcion no valida. Usando escenario por defecto (Flujo Libre).")
            scenario_name, rho0 = scenarios['1']

        print(f"\nSimulando escenario: {scenario_name}")
        print("Esto puede tomar unos segundos...")

        # Simular
        results = simulate_traffic_flow(rho0, x, t, boundary='periodic')
        rho = results['rho']

        print("✓ Simulacion completada")
        print("\nGenerando animacion interactiva...")
        print("Cierre la ventana de matplotlib para terminar.")

        # Crear y mostrar animación
        anim = create_interactive_animation(rho, x, t, title=f"Escenario: {scenario_name} (Modelo Macroscopico)")

        plt.show()

    else:
        # MODELO MICROSCÓPICO
        choice = select_scenario(model='micro')

        # Seleccionar escenario microscópico
        scenarios_micro = {
            '1': ('Flujo Libre', 10),
            '2': ('Flujo Moderado', 20),
            '3': ('Flujo Congestionado', 30),
            '4': ('Perturbacion Gaussiana', 25),
            '5': ('Perturbacion Sinusoidal', 20),
            '6': ('Dos Grupos', 20),
            '7': ('Gradiente Lineal', 25)
        }

        if choice in scenarios_micro:
            scenario_name, n_cars = scenarios_micro[choice]
        else:
            print("Opcion no valida. Usando escenario por defecto (Flujo Libre).")
            scenario_name, n_cars = scenarios_micro['1']

        print(f"\nSimulando escenario: {scenario_name}")
        print("Esto puede tomar unos segundos...")

        # Parametros para microscopico
        road_length = 10000.0
        final_time = 360.0
        dt = 1.0

        params = {
            "v0": 30.0,
            "a": 1.2,
            "b": 1.5,
            "T": 1.5,
            "s0": 2.0,
            "s_min": 2.0
        }

        init_positions = np.linspace(0.0, road_length * 0.9, n_cars)
        init_velocities = np.ones(n_cars) * (0.8 * params["v0"])

        model_micro = MicroscopicModel(
            n_cars=n_cars,
            road_length=road_length,
            params=params,
            init_positions=init_positions,
            init_velocities=init_velocities
        )

        t_array = np.arange(0.0, final_time + dt, dt)
        n_steps = len(t_array) - 1

        positions_record, velocities_record = simulate_microscopic(
            model_micro,
            dt=dt,
            n_steps=n_steps,
            periodic=False,
            road_length=road_length,
            record=True
        )

        print("✓ Simulacion completada")
        print("\nGenerando animacion interactiva...")
        print("Cierre la ventana de matplotlib para terminar.")

        # Para microscopico, crear animacion simple
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

        ax1.set_xlim(0, road_length)
        ax1.set_ylim(-5, 5)
        ax1.set_xlabel('Posicion (m)', fontsize=11)
        ax1.set_ylabel('Carril', fontsize=11)
        ax1.set_title(f'Escenario: {scenario_name} (Modelo Microscopico)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlim(0, final_time)
        ax2.set_ylim(0, params["v0"] * 1.2)
        ax2.set_xlabel('Tiempo (s)', fontsize=11)
        ax2.set_ylabel('Velocidad promedio (m/s)', fontsize=11)
        ax2.set_title('Velocidad Promedio en el Tiempo', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        scatter = ax1.scatter([], [], s=100, c='blue', alpha=0.7, edgecolor='black')
        line_vel, = ax2.plot([], [], 'g-', linewidth=2)
        info_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        def init():
            scatter.set_offsets(np.empty((0, 2)))
            line_vel.set_data([], [])
            return scatter, line_vel, info_text

        def animate(frame):
            positions = positions_record[frame, :]
            offsets = np.column_stack((positions, np.zeros_like(positions)))
            scatter.set_offsets(offsets)

            avg_vel = np.mean(velocities_record[:frame+1, :], axis=1)
            line_vel.set_data(t_array[:frame+1], avg_vel)

            info_text.set_text(f'Tiempo: {t_array[frame]:.1f}s | Vehiculos: {n_cars}')

            return scatter, line_vel, info_text

        anim = FuncAnimation(fig, animate, init_func=init, frames=n_steps, interval=50, blit=True, repeat=True)

        plt.tight_layout()
        plt.show()

    print("\n✓ Animacion cerrada")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
