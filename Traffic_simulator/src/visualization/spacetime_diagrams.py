import numpy as np
import matplotlib.pyplot as plt

def plot_spacetime_diagram_macro(rho, x, t, levels=10, filename=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(x, t, rho, levels=levels, cmap='RdYlGn_r')
    
    ax.set_xlabel('Posición (km)', fontsize=12)
    ax.set_ylabel('Tiempo (h)', fontsize=12)
    ax.set_title('Diagrama Espacio-Tiempo (Macroscópico)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(contour, ax=ax, label='Densidad (veh/km)')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_spacetime_diagram_micro(x_vehicles, t, vehicle_ids=None, filename=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    n_vehicles = x_vehicles.shape[1]
    
    if vehicle_ids is None:
        vehicle_ids = range(n_vehicles)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(vehicle_ids)))
    
    for i, vid in enumerate(vehicle_ids):
        ax.plot(x_vehicles[:, vid], t, color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Posición (km)', fontsize=12)
    ax.set_ylabel('Tiempo (h)', fontsize=12)
    ax.set_title('Diagrama Espacio-Tiempo (Microscópico)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_shockwave_detection(rho, x, t, threshold=100, filename=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    grad_x = np.gradient(rho, axis=1)
    shockwave_mask = np.abs(grad_x) > threshold
    
    ax.contourf(x, t, rho, levels=15, cmap='RdYlGn_r', alpha=0.6)
    ax.contour(x, t, shockwave_mask.astype(int), levels=[0.5], colors='black', linewidths=2)
    
    ax.set_xlabel('Posición (km)', fontsize=12)
    ax.set_ylabel('Tiempo (h)', fontsize=12)
    ax.set_title('Detección de Ondas de Choque', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_characteristic_curves(rho, x, t, x0_positions, filename=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.contourf(x, t, rho, levels=15, cmap='RdYlGn_r', alpha=0.5)
    
    for x0 in x0_positions:
        x0_idx = np.argmin(np.abs(x - x0))
        
        x_char = [x[x0_idx]]
        t_char = [t[0]]
        
        for i in range(1, len(t)):
            rho_current = rho[i-1, x0_idx]
            v_current = 100 * (1 - rho_current / 150)
            
            dx = v_current * (t[i] - t[i-1])
            x_new = x_char[-1] + dx
            
            if x_new < x[0] or x_new > x[-1]:
                break
            
            x_char.append(x_new)
            t_char.append(t[i])
        
        ax.plot(x_char, t_char, 'k-', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Posición (km)', fontsize=12)
    ax.set_ylabel('Tiempo (h)', fontsize=12)
    ax.set_title('Curvas Características', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax