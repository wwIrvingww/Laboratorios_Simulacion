import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_density_heatmap(rho, x, t, title="Mapa de Densidad", filename=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cmap = LinearSegmentedColormap.from_list('traffic', ['green', 'yellow', 'red'])
    
    im = ax.imshow(rho.T, aspect='auto', origin='lower', cmap=cmap,
                   extent=[t[0], t[-1], x[0], x[-1]], vmin=0, vmax=150)
    
    ax.set_xlabel('Tiempo (h)', fontsize=12)
    ax.set_ylabel('Posición (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Densidad (veh/km)')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_density_snapshots(rho, x, t, time_indices, filename=None):
    n_snapshots = len(time_indices)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(5*n_snapshots, 4))
    
    if n_snapshots == 1:
        axes = [axes]
    
    for idx, (ax, t_idx) in enumerate(zip(axes, time_indices)):
        ax.plot(x, rho[t_idx, :], 'b-', linewidth=2)
        ax.fill_between(x, 0, rho[t_idx, :], alpha=0.3)
        ax.set_xlabel('Posición (km)', fontsize=11)
        ax.set_ylabel('Densidad (veh/km)', fontsize=11)
        ax.set_title(f't = {t[t_idx]:.3f} h', fontsize=12)
        ax.set_ylim([0, 160])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, axes

def plot_density_evolution(rho, x, t, positions, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for pos_idx in positions:
        ax.plot(t, rho[:, pos_idx], label=f'x = {x[pos_idx]:.2f} km', linewidth=2)
    
    ax.set_xlabel('Tiempo (h)', fontsize=12)
    ax.set_ylabel('Densidad (veh/km)', fontsize=12)
    ax.set_title('Evolución Temporal de la Densidad', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_flux_density_relation(rho, flux, filename=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rho_flat = rho.flatten()
    flux_flat = flux.flatten()
    
    ax.scatter(rho_flat, flux_flat, alpha=0.3, s=10)
    
    rho_theory = np.linspace(0, 150, 100)
    v_theory = 100 * (1 - rho_theory / 150)
    flux_theory = rho_theory * v_theory
    ax.plot(rho_theory, flux_theory, 'r-', linewidth=2, label='Teoría')
    
    ax.set_xlabel('Densidad ρ (veh/km)', fontsize=12)
    ax.set_ylabel('Flujo q (veh/h)', fontsize=12)
    ax.set_title('Diagrama Fundamental de Tráfico', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax