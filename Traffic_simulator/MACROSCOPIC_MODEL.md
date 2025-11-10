# Modelo Macroscópico de Tráfico Vehicular

## Descripción

Este documento describe la implementación del modelo macroscópico de tráfico vehicular basado en la ecuación de conservación y el modelo de Greenshields.

## Fundamentos Teóricos

### Ecuación de Conservación

El tráfico se modela como un fluido continuo mediante:

```
∂ρ/∂t + ∂q/∂x = 0
```

donde:
- `ρ(x,t)` = densidad vehicular (veh/km)
- `q(x,t)` = flujo vehicular (veh/h)
- `x` = posición (km)
- `t` = tiempo (h)

### Modelo de Greenshields

La relación velocidad-densidad se describe mediante:

```
v(ρ) = V_max * (1 - ρ/ρ_max)
```

El flujo resulta:

```
q(ρ) = ρ * v(ρ) = ρ * V_max * (1 - ρ/ρ_max)
```

**Parámetros del sistema:**
- `V_max = 100 km/h` (velocidad máxima)
- `ρ_max = 150 veh/km` (densidad máxima)

**Valores críticos:**
- Densidad crítica: `ρ_c = ρ_max/2 = 75 veh/km`
- Flujo máximo (capacidad): `q_max = ρ_c * V_max/2 = 3750 veh/h`

## Método Numérico: Lax-Friedrichs

El esquema de Lax-Friedrichs es un método explícito de diferencias finitas:

```
u_i^(n+1) = 0.5 * (u_{i-1}^n + u_{i+1}^n) - (dt/(2*dx)) * (f_{i+1}^n - f_{i-1}^n)
```

### Condición CFL (Courant-Friedrichs-Lewy)

Para estabilidad numérica se requiere:

```
CFL = V_max * dt/dx ≤ 1.0
```

**Parámetros de discretización recomendados:**
- Espacial: `dx = 0.1 km`
- Temporal: `dt = 0.001 h` (3.6 s) para CFL = 1.0
- Temporal usado: `dt = 0.01 h` (36 s) → CFL = 10.0 (inestable para algunos casos)

**Nota:** El dt usado (0.01 h) viola la condición CFL en algunos escenarios. Para resultados más estables, use dt más pequeño o active el modo adaptativo.

## Estructura de Archivos

### Implementación

```
src/
├── models/
│   └── macroscopic.py          # Modelo principal, funciones de simulación y análisis
├── solvers/
│   └── lax_friedrichs.py       # Esquema numérico de Lax-Friedrichs
├── utils/
│   ├── parameters.py           # Parámetros globales (V_MAX, RHO_MAX, etc.)
│   └── initial_conditions.py  # Condiciones iniciales para diversos escenarios
└── visualization/
    ├── density_maps.py         # Mapas de calor y gráficos de densidad
    ├── spacetime_diagrams.py   # Diagramas espacio-tiempo
    └── travel_time_plots.py    # Métricas de desempeño
```

### Experimentos

```
experiments/
└── macroscopic_scenarios.py    # Orquestador de todos los escenarios
```

## Escenarios Implementados

### 1. Flujo Libre (ρ = 30 veh/km)
- Densidad baja uniforme
- Velocidad alta constante (80 km/h)
- Sin congestión

### 2. Congestión Uniforme (ρ = 120 veh/km)
- Densidad alta uniforme
- Velocidad baja constante (20 km/h)
- 100% del dominio congestionado

### 3. Onda de Choque
- Discontinuidad en x = 5 km
- ρ_upstream = 140 veh/km, ρ_downstream = 30 veh/km
- Propagación de onda de choque

### 4. Perturbación Gaussiana
- Pulso localizado en x = 5 km
- Amplitud: 100 veh/km, ancho: 0.5 km
- Dispersión y propagación del pulso

### 5. Perturbación Sinusoidal
- ρ_base = 60 veh/km, amplitud = 30 veh/km
- Longitud de onda: 2 km
- Patrones periódicos

### 6. Dos Pulsos
- Pulsos en x = 3 km y x = 7 km
- Interacción y fusión de perturbaciones

### 7. Gradiente Lineal
- ρ varía linealmente de 20 a 120 veh/km
- Estudio de transiciones graduales

## Uso

### Ejecución Básica

```bash
cd Traffic_simulator
python experiments/macroscopic_scenarios.py
```

Esto ejecutará los 7 escenarios y generará:
- Figuras en `results/figures/macroscopic/`
- Métricas en `results/metrics/`

### Ejecutar un Escenario Individual

```python
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.models.macroscopic import simulate_traffic_flow
from src.utils.parameters import get_spatial_grid, get_temporal_grid
from src.utils.initial_conditions import gaussian_pulse

# Configurar dominio
x = get_spatial_grid(L=10.0, dx=0.1)
t = get_temporal_grid(T=1.0, dt=0.01)

# Condición inicial
rho0 = gaussian_pulse(x, x0=5.0, amplitude=100.0, width=0.5)

# Simular
results = simulate_traffic_flow(rho0, x, t, boundary='periodic')

rho = results['rho']      # Densidad ρ(x,t)
flux = results['flux']    # Flujo q(x,t)
velocity = results['velocity']  # Velocidad v(x,t)
```

### Modo Adaptativo (CFL seguro)

Para mayor estabilidad, use el modo adaptativo:

```python
results = simulate_traffic_flow(
    rho0, x, t=1.0,  # t_final en lugar de array
    boundary='periodic',
    adaptive=True,
    cfl_target=0.8
)
```

## Visualizaciones Generadas

Para cada escenario se generan automáticamente:

1. **Mapa de Calor de Densidad** (`density_heatmap.png`)
   - Evolución espacio-temporal de ρ(x,t)
   - Escala de color: verde (baja) → amarillo → rojo (alta)

2. **Snapshots Temporales** (`density_snapshots.png`)
   - Perfiles de densidad en 5 instantes
   - Muestra evolución temporal

3. **Evolución Temporal** (`density_evolution.png`)
   - Densidad vs tiempo en 5 posiciones fijas
   - Detecta variaciones locales

4. **Diagrama Fundamental** (`fundamental_diagram.png`)
   - Flujo vs densidad
   - Compara simulación con teoría

5. **Diagrama Espacio-Tiempo** (`spacetime_diagram.png`)
   - Contornos de densidad
   - Visualiza propagación de ondas

6. **Detección de Ondas de Choque** (`shockwave_detection.png`)
   - Identifica discontinuidades
   - Umbral de gradiente: 50 veh/km²

7. **Curvas Características** (`characteristic_curves.png`)
   - Trayectorias de información
   - Velocidades de onda

8. **Tiempo de Viaje** (`travel_time.png`)
   - Evolución del tiempo para cruzar el dominio

9. **Velocidad Promedio** (`average_velocity.png`)
   - Velocidad media espacial vs tiempo

10. **Métricas de Congestión** (`congestion_metrics.png`)
    - % de carretera congestionada vs tiempo
    - Duración de congestión por posición

## Métricas Calculadas

Para cada escenario se reportan:

- **Densidad promedio** (inicial y final)
- **Velocidad promedio** (inicial y final)
- **Tiempo de viaje** (inicial y final)
- **Nivel de congestión**:
  - Fracción máxima de carretera congestionada
  - Fracción promedio
- **Ondas de choque detectadas**
- **Total de vehículos** (conservación)

## Resultados Esperados

### Escenario 1: Flujo Libre
- ✓ Densidad constante (~30 veh/km)
- ✓ Sin congestión (0%)
- ✓ Velocidad alta (~80 km/h)
- ✓ Tiempo de viaje corto (~7.6 min)

### Escenario 2: Congestión Uniforme
- ✓ Densidad constante alta (~120 veh/km)
- ✓ Congestión total (100%)
- ✓ Velocidad baja (~20 km/h)
- ✓ Tiempo de viaje largo (~30 min)

### Escenario 3: Onda de Choque
- ⚠ Inestabilidad numérica (CFL violado)
- ✓ Formación y propagación de onda de choque
- ~ Velocidad de shock: ~43.7 km/h (promedio)
- ⚠ Oscilaciones numéricas (NaN en tiempos finales)

### Escenarios 4-7
- ⚠ Inestabilidad en tiempos largos
- ✓ Patrones iniciales capturados correctamente
- ~ Propagación de perturbaciones visible
- ⚠ Requiere dt más pequeño para estabilidad completa

## Limitaciones y Recomendaciones

### Limitaciones Actuales

1. **Condición CFL**: El dt = 0.01 h usado es demasiado grande para algunos escenarios
2. **Inestabilidad numérica**: Aparecen NaN en escenarios con discontinuidades fuertes
3. **Fronteras periódicas**: Pueden no ser realistas para todos los casos

### Recomendaciones

Para mejorar la estabilidad:

1. **Reducir dt**:
   ```python
   t = get_temporal_grid(T=1.0, dt=0.001)  # CFL = 1.0
   ```

2. **Usar modo adaptativo**:
   ```python
   results = simulate_traffic_flow(rho0, x, t=1.0, adaptive=True, cfl_target=0.5)
   ```

3. **Fronteras de flujo saliente** (más realistas):
   ```python
   results = simulate_traffic_flow(rho0, x, t, boundary='outflow')
   ```

4. **Filtro de suavizado** para estabilizar:
   ```python
   from scipy.ndimage import gaussian_filter1d
   rho0_smooth = gaussian_filter1d(rho0, sigma=2)
   ```

## Interpretación Física

### Diagrama Fundamental

- **Fase I** (ρ < ρ_c): Flujo libre, q aumenta con ρ
- **Fase II** (ρ > ρ_c): Congestión, q disminuye con ρ
- **Punto crítico** (ρ = 75 veh/km): Máximo flujo (3750 veh/h)

### Ondas de Choque

Velocidad de propagación:

```
s = (q2 - q1) / (ρ2 - ρ1)
```

- Shock hacia atrás: tráfico denso alcanza tráfico ligero
- Shock hacia adelante: tráfico ligero entra en zona densa

### Velocidades Características

```
c(ρ) = dq/dρ = V_max * (1 - 2*ρ/ρ_max)
```

- c > 0: ondas se propagan hacia adelante
- c < 0: ondas se propagan hacia atrás
- c = 0: onda estacionaria en ρ_crítica

## Referencias

1. Lighthill, M. J., & Whitham, G. B. (1955). *On kinematic waves II. A theory of traffic flow on long crowded roads*. Proceedings of the Royal Society of London. Series A.

2. Greenshields, B. D. (1935). *A study of traffic capacity*. Highway Research Board Proceedings, 14, 448-477.

3. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.

4. Treiber, M., & Kesting, A. (2013). *Traffic Flow Dynamics: Data, Models and Simulation*. Springer.

## Autores

- Josué
- Irving
- Sebastián

## Fecha

Noviembre 2025
