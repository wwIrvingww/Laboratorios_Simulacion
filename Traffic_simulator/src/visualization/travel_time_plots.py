import numpy as np
import matplotlib.pyplot as plt

def calculate_travel_time_macro(rho, x, t):
    travel_times = []
    
    for i in range(len(t)):
        v = 100 * (1 - rho[i, :] / 150)
        dt = np.diff(x) / v[:-1]
        total_time = np.sum(dt)
        travel_times.append(total_time)
    
    return np.array(travel_times)

def calculate_travel_time_micro(x_vehicles, t):
    travel_times = []
    n_vehicles = x_vehicles.shape[1]
    
    for i in range(n_vehicles):
        trajectory = x_vehicles[:, i]
        
        start_idx = 0
        end_idx = len(t) - 1
        
        for j in range(len(trajectory) - 1):
            if trajectory[j+1] < trajectory[j]:
                end_idx = j
                break
        
        travel_time = t[end_idx] - t[start_idx]
        travel_times.append(travel_time)
    
    return np.array(travel_times)

def plot_travel_time_evolution(travel_times, t, model_type="Macroscópico", filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, travel_times, 'b-', linewidth=2)
    ax.fill_between(t, travel_times, alpha=0.3)
    
    mean_time = np.mean(travel_times)
    ax.axhline(y=mean_time, color='r', linestyle='--', linewidth=2, 
               label=f'Promedio: {mean_time:.4f} h')
    
    ax.set_xlabel('Tiempo (h)', fontsize=12)
    ax.set_ylabel('Tiempo de Viaje (h)', fontsize=12)
    ax.set_title(f'Evolución del Tiempo de Viaje - {model_type}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_travel_time_histogram(travel_times, model_type="Microscópico", filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(travel_times, bins=20, edgecolor='black', alpha=0.7)
    
    mean_time = np.mean(travel_times)
    std_time = np.std(travel_times)
    
    ax.axvline(x=mean_time, color='r', linestyle='--', linewidth=2, 
               label=f'Media: {mean_time:.4f} h')
    ax.axvline(x=mean_time - std_time, color='orange', linestyle=':', linewidth=2, 
               label=f'μ - σ: {mean_time - std_time:.4f} h')
    ax.axvline(x=mean_time + std_time, color='orange', linestyle=':', linewidth=2, 
               label=f'μ + σ: {mean_time + std_time:.4f} h')
    
    ax.set_xlabel('Tiempo de Viaje (h)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title(f'Distribución de Tiempos de Viaje - {model_type}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_average_velocity(rho, x, t, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_velocities = []
    for i in range(len(t)):
        v = 100 * (1 - rho[i, :] / 150)
        avg_v = np.mean(v)
        avg_velocities.append(avg_v)
    
    avg_velocities = np.array(avg_velocities)
    
    ax.plot(t, avg_velocities, 'g-', linewidth=2)
    ax.fill_between(t, avg_velocities, alpha=0.3)
    
    ax.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.5, 
               label='v_max = 100 km/h')
    
    ax.set_xlabel('Tiempo (h)', fontsize=12)
    ax.set_ylabel('Velocidad Promedio (km/h)', fontsize=12)
    ax.set_title('Evolución de la Velocidad Promedio', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_travel_time_comparison(travel_times_macro, travel_times_micro, t, filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if len(travel_times_macro) == len(t):
        ax1.plot(t, travel_times_macro, 'b-', linewidth=2, label='Macroscópico')
    else:
        ax1.hist(travel_times_macro, bins=15, alpha=0.5, label='Macroscópico', 
                 color='blue', edgecolor='black')
    
    if len(travel_times_micro) == len(t):
        ax1.plot(t, travel_times_micro, 'r-', linewidth=2, label='Microscópico')
    else:
        ax1.hist(travel_times_micro, bins=15, alpha=0.5, label='Microscópico', 
                 color='red', edgecolor='black')
    
    ax1.set_xlabel('Tiempo de Viaje (h)', fontsize=11)
    ax1.set_ylabel('Frecuencia / Valor', fontsize=11)
    ax1.set_title('Comparación de Tiempos de Viaje', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    stats_macro = {
        'Media': np.mean(travel_times_macro),
        'Desv. Std': np.std(travel_times_macro),
        'Mín': np.min(travel_times_macro),
        'Máx': np.max(travel_times_macro)
    }
    
    stats_micro = {
        'Media': np.mean(travel_times_micro),
        'Desv. Std': np.std(travel_times_micro),
        'Mín': np.min(travel_times_micro),
        'Máx': np.max(travel_times_micro)
    }
    
    categories = list(stats_macro.keys())
    macro_values = list(stats_macro.values())
    micro_values = list(stats_micro.values())
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x_pos - width/2, macro_values, width, label='Macroscópico', color='blue', alpha=0.7)
    ax2.bar(x_pos + width/2, micro_values, width, label='Microscópico', color='red', alpha=0.7)
    
    ax2.set_xlabel('Estadística', fontsize=11)
    ax2.set_ylabel('Valor (h)', fontsize=11)
    ax2.set_title('Comparación Estadística', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)

def plot_congestion_metrics(rho, x, t, threshold=75, filename=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    congestion_mask = rho > threshold
    congestion_fraction = np.mean(congestion_mask, axis=1)
    
    ax1.plot(t, congestion_fraction * 100, 'r-', linewidth=2)
    ax1.fill_between(t, congestion_fraction * 100, alpha=0.3)
    ax1.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, 
                label='50% congestionado')
    
    ax1.set_xlabel('Tiempo (h)', fontsize=11)
    ax1.set_ylabel('% de carretera congestionada', fontsize=11)
    ax1.set_title('Nivel de Congestión en el Tiempo', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    congestion_duration = np.sum(congestion_mask, axis=0)
    
    ax2.plot(x, congestion_duration, 'b-', linewidth=2)
    ax2.fill_between(x, congestion_duration, alpha=0.3)
    
    ax2.set_xlabel('Posición (km)', fontsize=11)
    ax2.set_ylabel('Tiempo total congestionado', fontsize=11)
    ax2.set_title('Duración de Congestión por Posición', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)