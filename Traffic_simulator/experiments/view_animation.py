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
    fig = plt.figure(figsize=(14, 8))
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


def select_scenario():
    """Permite al usuario seleccionar un escenario para visualizar."""
    print("\n" + "="*70)
    print("VISOR INTERACTIVO DE ANIMACIONES - MODELO MACROSCÓPICO")
    print("="*70)
    print("\nEscenarios disponibles:")
    print("  1. Flujo Libre (densidad baja uniforme)")
    print("  2. Onda de Choque (discontinuidad)")
    print("  3. Perturbación Gaussiana (pulso localizado)")
    print("  4. Perturbación Sinusoidal (variación periódica)")
    print("  5. Dos Pulsos (interacción)")
    print("  6. Personalizado (ingresar parámetros)")
    print()
    
    choice = input("Seleccione un escenario (1-6) [1]: ").strip()
    if not choice:
        choice = '1'
    
    return choice


def main():
    """Función principal para visualizar animación interactiva."""
    
    # Configurar parámetros
    L = 10.0
    T = 1.0
    dx = 0.1
    dt = 0.01
    
    x = get_spatial_grid(L=L, dx=dx)
    t = get_temporal_grid(T=T, dt=dt)
    
    # Seleccionar escenario
    choice = select_scenario()
    
    # Definir condición inicial según elección
    scenarios = {
        '1': ('Flujo Libre', uniform_density(x, rho_value=30.0)),
        '2': ('Onda de Choque', shock_wave_scenario(x, x_shock=5.0, rho_upstream=140.0, rho_downstream=30.0)),
        '3': ('Perturbación Gaussiana', gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5)),
        '4': ('Perturbación Sinusoidal', sinusoidal_perturbation(x, rho_base=60.0, amplitude=30.0, wavelength=2.0)),
        '5': ('Dos Pulsos', two_pulse_scenario(x, x1=3.0, x2=7.0, amplitude1=80.0, amplitude2=100.0, width=0.5))
    }
    
    if choice in scenarios:
        scenario_name, rho0 = scenarios[choice]
    else:
        print("Opción no válida. Usando escenario por defecto (Flujo Libre).")
        scenario_name, rho0 = scenarios['1']
    
    print(f"\nSimulando escenario: {scenario_name}")
    print("Esto puede tomar unos segundos...")
    
    # Simular
    results = simulate_traffic_flow(rho0, x, t, boundary='periodic')
    rho = results['rho']
    
    print("✓ Simulación completada")
    print("\nGenerando animación interactiva...")
    print("Cierre la ventana de matplotlib para terminar.")
    
    # Crear y mostrar animación
    anim = create_interactive_animation(rho, x, t, title=f"Escenario: {scenario_name}")
    
    plt.show()
    
    print("\n✓ Animación cerrada")
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
