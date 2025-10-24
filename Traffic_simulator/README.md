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