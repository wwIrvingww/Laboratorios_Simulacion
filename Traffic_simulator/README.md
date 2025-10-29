# Simulación de Tráfico Vehicular: Dinámica y Congestión

## Descripción del Proyecto

Este proyecto implementa dos enfoques complementarios para simular el flujo vehicular y estudiar la formación de atascos:

- **Modelo Macroscópico (EDP)**: Describe el tráfico como un fluido continuo usando ecuaciones de conservación
- **Modelo Microscópico (EDO)**: Simula el comportamiento individual de cada vehículo

**Objetivo**: Diseñar e implementar un simulador mediante solución numérica de EDOs y EDPs para identificar umbrales de congestión y validar estrategias de control.

**Parámetros**:
- v_max = 100 km/h
- ρ_max = 150 veh/km
- Métodos numéricos: Lax-Friedrichs, Runge-Kutta 4

## Integrantes

- **Josué**
- **Irving**
- **Sebastián**

## Estructura del Repositorio
```
traffic-simulation/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── macroscopic.py
│   │   └── microscopic.py
│   │
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── lax_friedrichs.py
│   │   └── runge_kutta.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── density_maps.py
│   │   ├── spacetime_diagrams.py
│   │   ├── animations.py
│   │   └── travel_time_plots.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── parameters.py
│   │   └── initial_conditions.py
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── metrics.py
│       └── control_strategies.py
│
├── experiments/
│   ├── macroscopic_scenarios.py
│   ├── microscopic_scenarios.py
│   └── comparative_analysis.py
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_visualization.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
└── results/
    ├── figures/
    ├── animations/
    └── metrics/
```

## Instalación
```bash
git clone https://github.com/usuario/traffic-simulation.git
cd traffic-simulation
pip install -r requirements.txt
```

## Uso
```bash
python main.py
```

Para ejecutar escenarios específicos:
```bash
python experiments/macroscopic_scenarios.py
python experiments/microscopic_scenarios.py
python experiments/comparative_analysis.py
```

# EJEMPLO DE USO DE utils/parameter.py

## parameters.py

```python
from src.utils.parameters import V_MAX, RHO_MAX, get_spatial_grid, get_temporal_grid, get_velocity, get_flux

x = get_spatial_grid(L=10.0, dx=0.1)
t = get_temporal_grid(T=1.0, dt=0.01)

rho = 75.0
v = get_velocity(rho)
q = get_flux(rho)
```

# EJEMPLOS DE USO visualization/

## density_maps.py

```python
from src.visualization.density_maps import plot_density_heatmap, plot_density_snapshots, plot_density_evolution, plot_flux_density_relation

plot_density_heatmap(rho, x, t, filename='density.png')

time_indices = [0, 25, 50, 75, 100]
plot_density_snapshots(rho, x, t, time_indices, filename='snapshots.png')

positions = [10, 30, 50, 70, 90]
plot_density_evolution(rho, x, t, positions, filename='evolution.png')

plot_flux_density_relation(rho, flux, filename='fundamental_diagram.png')
```

## spacetime_diagrams.py

```python
from src.visualization.spacetime_diagrams import plot_spacetime_diagram_macro, plot_spacetime_diagram_micro, plot_shockwave_detection, plot_characteristic_curves

plot_spacetime_diagram_macro(rho, x, t, filename='spacetime_macro.png')

plot_spacetime_diagram_micro(x_vehicles, t, filename='spacetime_micro.png')

plot_shockwave_detection(rho, x, t, threshold=100, filename='shockwaves.png')

x0_positions = [2.0, 5.0, 8.0]
plot_characteristic_curves(rho, x, t, x0_positions, filename='characteristics.png')
```

## animations.py

```python
from src.visualization.animations import animate_macroscopic_traffic, animate_microscopic_traffic, animate_combined_view

animate_macroscopic_traffic(rho, x, t, filename='animation_macro.gif', fps=10)

animate_microscopic_traffic(x_vehicles, v_vehicles, t, L_road=10.0, filename='animation_micro.gif', fps=10)

animate_combined_view(rho, x_vehicles, x, t, L_road=10.0, filename='animation_combined.gif', fps=10)
```

## travel_time_plots.py

```python
from src.visualization.travel_time_plots import calculate_travel_time_macro, calculate_travel_time_micro, plot_travel_time_evolution, plot_travel_time_histogram, plot_average_velocity, plot_travel_time_comparison, plot_congestion_metrics

travel_times = calculate_travel_time_macro(rho, x, t)
plot_travel_time_evolution(travel_times, t, filename='travel_time_evolution.png')

travel_times = calculate_travel_time_micro(x_vehicles, t)
plot_travel_time_histogram(travel_times, filename='travel_time_hist.png')

plot_average_velocity(rho, x, t, filename='avg_velocity.png')

plot_travel_time_comparison(travel_times_macro, travel_times_micro, t, filename='comparison.png')

plot_congestion_metrics(rho, x, t, threshold=75, filename='congestion.png')
```