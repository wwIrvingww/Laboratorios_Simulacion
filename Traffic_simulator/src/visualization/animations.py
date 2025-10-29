import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_macroscopic_traffic(rho, x, t, filename=None, fps=10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    line1, = ax1.plot([], [], 'b-', linewidth=2)
    fill = ax1.fill_between(x, 0, 0, alpha=0.3)
    
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(0, 160)
    ax1.set_xlabel('Posición (km)', fontsize=11)
    ax1.set_ylabel('Densidad (veh/km)', fontsize=11)
    ax1.set_title('Densidad Vehicular', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                         fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    im = ax2.imshow(rho[:1, :].T, aspect='auto', origin='lower', cmap='RdYlGn_r',
                    extent=[x[0], x[-1], t[0], t[-1]], vmin=0, vmax=150)
    ax2.set_xlabel('Posición (km)', fontsize=11)
    ax2.set_ylabel('Tiempo (h)', fontsize=11)
    ax2.set_title('Evolución Espacio-Temporal', fontsize=12)
    
    hline = ax2.axhline(y=t[0], color='white', linewidth=2, linestyle='--')
    
    plt.colorbar(im, ax=ax2, label='Densidad (veh/km)')
    
    def init():
        line1.set_data([], [])
        time_text.set_text('')
        return line1, fill, time_text, im, hline
    
    def animate(frame):
        nonlocal fill
        
        line1.set_data(x, rho[frame, :])
        
        for coll in ax1.collections:
            coll.remove()
        fill = ax1.fill_between(x, 0, rho[frame, :], alpha=0.3)
        
        time_text.set_text(f'Tiempo: {t[frame]:.3f} h')
        
        im.set_data(rho[:frame+1, :].T)
        im.set_extent([x[0], x[-1], t[0], t[frame]])
        
        hline.set_ydata([t[frame], t[frame]])
        
        return line1, fill, time_text, im, hline
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), 
                        interval=1000/fps, blit=False, repeat=True)
    
    if filename:
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
    
    plt.tight_layout()
    return anim

def animate_microscopic_traffic(x_vehicles, v_vehicles, t, L_road=10.0, filename=None, fps=10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    n_vehicles = x_vehicles.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_vehicles))
    
    cars = []
    for i in range(n_vehicles):
        car, = ax1.plot([], [], 'o', markersize=10, color=colors[i])
        cars.append(car)
    
    ax1.set_xlim(0, L_road)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Posición (km)', fontsize=11)
    ax1.set_title('Posición de Vehículos', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_yticks([])
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                         fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    velocity_lines = []
    for i in range(n_vehicles):
        line, = ax2.plot([], [], color=colors[i], linewidth=1.5, alpha=0.7)
        velocity_lines.append(line)
    
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, 120)
    ax2.set_xlabel('Tiempo (h)', fontsize=11)
    ax2.set_ylabel('Velocidad (km/h)', fontsize=11)
    ax2.set_title('Evolución de Velocidades', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    def init():
        for car in cars:
            car.set_data([], [])
        for line in velocity_lines:
            line.set_data([], [])
        time_text.set_text('')
        return cars + velocity_lines + [time_text]
    
    def animate(frame):
        for i, car in enumerate(cars):
            car.set_data([x_vehicles[frame, i]], [0])
        
        for i, line in enumerate(velocity_lines):
            line.set_data(t[:frame+1], v_vehicles[:frame+1, i])
        
        time_text.set_text(f'Tiempo: {t[frame]:.3f} h')
        
        return cars + velocity_lines + [time_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), 
                        interval=1000/fps, blit=True, repeat=True)
    
    if filename:
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
    
    plt.tight_layout()
    return anim

def animate_combined_view(rho, x_vehicles, x, t, L_road=10.0, filename=None, fps=10):
    fig = plt.figure(figsize=(14, 8))
    
    ax1 = plt.subplot(2, 2, 1)
    line_density, = ax1.plot([], [], 'b-', linewidth=2)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(0, 160)
    ax1.set_xlabel('Posición (km)', fontsize=10)
    ax1.set_ylabel('Densidad (veh/km)', fontsize=10)
    ax1.set_title('Densidad (Macro)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 2, 2)
    n_vehicles = x_vehicles.shape[1]
    cars = []
    for i in range(n_vehicles):
        car, = ax2.plot([], [], 'o', markersize=8)
        cars.append(car)
    ax2.set_xlim(0, L_road)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('Posición (km)', fontsize=10)
    ax2.set_title('Vehículos (Micro)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_yticks([])
    
    ax3 = plt.subplot(2, 1, 2)
    im = ax3.imshow(rho[:1, :].T, aspect='auto', origin='lower', cmap='RdYlGn_r',
                    extent=[x[0], x[-1], t[0], t[-1]], vmin=0, vmax=150)
    
    for i in range(n_vehicles):
        ax3.plot(x_vehicles[:1, i], t[:1], 'w-', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Posición (km)', fontsize=10)
    ax3.set_ylabel('Tiempo (h)', fontsize=10)
    ax3.set_title('Diagrama Espacio-Tiempo', fontsize=11)
    plt.colorbar(im, ax=ax3, label='Densidad (veh/km)')
    
    time_text = fig.text(0.5, 0.98, '', ha='center', fontsize=12, fontweight='bold')
    
    def init():
        line_density.set_data([], [])
        for car in cars:
            car.set_data([], [])
        time_text.set_text('')
        return [line_density] + cars + [time_text, im]
    
    def animate(frame):
        line_density.set_data(x, rho[frame, :])
        
        for i, car in enumerate(cars):
            car.set_data([x_vehicles[frame, i]], [0])
        
        im.set_data(rho[:frame+1, :].T)
        im.set_extent([x[0], x[-1], t[0], t[frame]])
        
        time_text.set_text(f'Tiempo: {t[frame]:.3f} h')
        
        return [line_density] + cars + [time_text, im]
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), 
                        interval=1000/fps, blit=False, repeat=True)
    
    if filename:
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
    
    plt.tight_layout()
    return anim