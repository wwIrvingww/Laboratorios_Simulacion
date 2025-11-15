import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

def animate_macroscopic_traffic(rho, x, t, filename=None, fps=10):
    """Animacion macroscopica mejorada - Densidad evoluciona en tiempo real."""
    fig, ax = plt.subplots(figsize=(14, 6))

    line, = ax.plot([], [], 'b-', linewidth=3, label='Densidad')
    fill = ax.fill_between(x, 0, 0, alpha=0.4, color='cyan')

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 160)
    ax.set_xlabel('Posicion (km)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Densidad (veh/km)', fontsize=13, fontweight='bold')
    ax.set_title('SIMULACION MACROSCOPICA: Densidad Vehicular en Tiempo Real',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')

    # Lineas de referencia
    ax.axhline(y=75, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Congestion (75 veh/km)')
    ax.axhline(y=150, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Capacidad Maxima (150 veh/km)')

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    progress_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, fontsize=11,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    def init():
        line.set_data([], [])
        return line, fill, time_text, progress_text

    def animate(frame):
        nonlocal fill

        line.set_data(x, rho[frame, :])

        # Remover fill anterior
        for coll in ax.collections:
            coll.remove()

        # Crear nuevo fill con color segun densidad
        fill = ax.fill_between(x, 0, rho[frame, :], alpha=0.4, color='cyan')

        # Actualizar textos
        time_text.set_text(f'Tiempo: {t[frame]:.4f} h ({t[frame]*60:.2f} min)')
        progress = int((frame / len(t)) * 100)
        progress_text.set_text(f'Progreso: {progress}%')

        return line, fill, time_text, progress_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                        interval=1000/fps, blit=False, repeat=True)

    if filename:
        print(f"Guardando animacion macroscopica: {filename}")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"OK Animacion macroscopica guardada")

    plt.tight_layout()
    return anim


def animate_microscopic_traffic(x_vehicles, v_vehicles, t, L_road=800.0, filename=None, fps=10):
    """Animacion microscopica mejorada - Posiciones de vehiculos + velocidades."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1.5]})

    n_vehicles = x_vehicles.shape[1]
    colors = plt.cm.Set3(np.linspace(0, 1, n_vehicles))

    # Panel superior: Carretera con vehiculos
    ax1.set_xlim(0, L_road)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('Posicion (m)', fontsize=13, fontweight='bold')
    ax1.set_title('SIMULACION MICROSCOPICA: Posiciones de Vehiculos', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_yticks([])

    # Dibujar carretera
    road = Rectangle((0, -0.3), L_road, 0.6, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax1.add_patch(road)

    # Crear circulos para vehiculos
    cars = []
    for i in range(n_vehicles):
        circle = plt.Circle((0, 0), 20, color=colors[i], ec='black', linewidth=1.5)
        ax1.add_patch(circle)
        cars.append(circle)

    time_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                         verticalalignment='top', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Panel inferior: Velocidades individuales
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, 35)
    ax2.set_xlabel('Tiempo (s)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Velocidad (m/s)', fontsize=13, fontweight='bold')
    ax2.set_title('Evolucion de Velocidades Individuales', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.4, linestyle='--')

    velocity_lines = []
    for i in range(n_vehicles):
        line, = ax2.plot([], [], color=colors[i], linewidth=2.5, label=f'Vehiculo {i+1}', alpha=0.8)
        velocity_lines.append(line)

    ax2.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9)

    progress_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=11,
                            verticalalignment='top', fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    def init():
        for car in cars:
            car.center = (0, 0)
        for line in velocity_lines:
            line.set_data([], [])
        return cars + velocity_lines + [time_text1, progress_text]

    def animate(frame):
        # Actualizar posiciones de vehiculos
        for i, car in enumerate(cars):
            car.center = (x_vehicles[frame, i], 0)

        # Actualizar velocidades
        for i, line in enumerate(velocity_lines):
            line.set_data(t[:frame+1], v_vehicles[:frame+1, i])

        # Textos
        time_text1.set_text(f'Tiempo: {t[frame]:.2f} s')
        progress = int((frame / len(t)) * 100)
        progress_text.set_text(f'Progreso: {progress}%')

        return cars + velocity_lines + [time_text1, progress_text]

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                        interval=1000/fps, blit=False, repeat=True)

    if filename:
        print(f"Guardando animacion microscopica: {filename}")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"OK Animacion microscopica guardada")

    plt.tight_layout()
    return anim


def animate_combined_view(rho, x_vehicles, x, t, L_road=10.0, filename=None, fps=10):
    """Animacion combinada: vista macroscopica y microscopica simultaneas."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Densidad (Macroscopico)
    ax1 = fig.add_subplot(gs[0, :])
    line_density, = ax1.plot([], [], 'b-', linewidth=3)
    fill = ax1.fill_between(x, 0, 0, alpha=0.4, color='cyan')
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(0, 160)
    ax1.set_xlabel('Posicion (km)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Densidad (veh/km)', fontsize=11, fontweight='bold')
    ax1.set_title('MODELO MACROSCOPICO: Densidad Vehicular', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=75, color='orange', linestyle='--', alpha=0.5)
    ax1.axhline(y=150, color='red', linestyle='--', alpha=0.5)

    # Vehiculos (Microscopico)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(0, L_road)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel('Posicion (km)', fontsize=11, fontweight='bold')
    ax2.set_title('MODELO MICROSCOPICO: Posiciones', fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')

    road = Rectangle((0, -0.3), L_road, 0.6, linewidth=2, edgecolor='black', facecolor='gray', alpha=0.2)
    ax2.add_patch(road)

    n_vehicles = x_vehicles.shape[1]
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_vehicles, 20)))
    cars = []
    for i in range(min(n_vehicles, 20)):
        circle = plt.Circle((0, 0), 0.03, color=colors[i], ec='black', linewidth=1)
        ax2.add_patch(circle)
        cars.append(circle)

    # Velocidades
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(0, 35)
    ax3.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Velocidad (m/s)', fontsize=11, fontweight='bold')
    ax3.set_title('Velocidades Vehiculos', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    velocity_lines = []
    for i in range(min(n_vehicles, 10)):
        line, = ax3.plot([], [], linewidth=2, alpha=0.7)
        velocity_lines.append(line)

    # Textos de tiempo y progreso
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=13, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    progress_text = fig.text(0.98, 0.02, '', ha='right', fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    def init():
        line_density.set_data([], [])
        for car in cars:
            car.center = (0, 0)
        for line in velocity_lines:
            line.set_data([], [])
        return [line_density, fill] + cars + velocity_lines

    def animate(frame):
        nonlocal fill

        # Densidad
        line_density.set_data(x, rho[frame, :])
        for coll in ax1.collections:
            coll.remove()
        fill = ax1.fill_between(x, 0, rho[frame, :], alpha=0.4, color='cyan')

        # Vehiculos
        for i, car in enumerate(cars):
            car.center = (x_vehicles[frame, i] / 1000, 0)  # Convertir a km

        # Velocidades
        for i, line in enumerate(velocity_lines):
            line.set_data(t[:frame+1], v_vehicles[:frame+1, i])

        # Textos
        time_text.set_text(f'Tiempo: {t[frame]:.2f} s = {t[frame]/60:.4f} h')
        progress = int((frame / len(t)) * 100)
        progress_text.set_text(f'Progreso: {progress}%')

        return [line_density, fill] + cars + velocity_lines

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                        interval=1000/fps, blit=False, repeat=True)

    if filename:
        print(f"Guardando animacion combinada: {filename}")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"OK Animacion combinada guardada")

    plt.tight_layout()
    return anim
