# Guía de Estrategias de Control de Tráfico

## 📚 Índice
1. [Introducción](#introducción)
2. [Estrategias Implementadas](#estrategias-implementadas)
3. [Cómo Usar](#cómo-usar)
4. [Resultados y Análisis](#resultados-y-análisis)
5. [Referencias](#referencias)

---

## 🎯 Introducción

Este módulo implementa diversas **estrategias de control activo** para mejorar el flujo vehicular y reducir la congestión en simulaciones macroscópicas de tráfico.

### ¿Qué es el Control de Tráfico?

El control de tráfico consiste en aplicar acciones dinámicas sobre el sistema de transporte para:
- **Reducir tiempos de viaje**
- **Prevenir formación de congestión**
- **Suavizar ondas de choque**
- **Optimizar el uso de la infraestructura**

---

## 🛠️ Estrategias Implementadas

### 1. **Variable Speed Limits (VSL)** 🚦
Ajusta dinámicamente la velocidad máxima permitida según la densidad de tráfico.

#### Principio de Funcionamiento:
```
Si densidad > umbral_crítico:
    Reducir velocidad_máxima
Objetivo: Prevenir frenados bruscos y ondas de choque
```

#### Modos Disponibles:
- **VSL Suave**: Reducción gradual de velocidad
  - ρ > 90 veh/km → v_max = 70-80 km/h
  - ρ > 120 veh/km → v_max = 40-60 km/h

- **VSL Agresivo**: Reducción más drástica
  - Activación anticipada en zonas críticas
  - Mayor reducción de velocidad

#### Cuándo Usar:
- Ondas de choque detectadas
- Alta densidad localizada
- Transiciones bruscas de tráfico

---

### 2. **Ramp Metering** 🚥
Controla el flujo de vehículos que entran a la autopista desde rampas de acceso.

#### Principio:
```
Si densidad_vía_principal > densidad_objetivo:
    Reducir flujo_entrada (aumentar tiempo_rojo)
Objetivo: Mantener flujo óptimo
```

#### Parámetros:
- **Densidad objetivo**: ρ_target = 75 veh/km (densidad crítica)
- **Flujo máximo**: 2000 veh/h
- **Flujo mínimo**: 200 veh/h

#### Aplicaciones:
- Control de entradas en horas pico
- Prevención de sobresaturación
- Maximización de throughput

---

### 3. **Feedback Control (PID)** 🎛️
Controlador Proporcional-Integral-Derivativo para mantener densidad cerca de un valor objetivo.

#### Ecuación de Control:
```
u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt

Donde:
  e(t) = ρ_objetivo - ρ_actual (error)
  Kp = ganancia proporcional
  Ki = ganancia integral
  Kd = ganancia derivativa
```

#### Componentes:
- **Proporcional**: Respuesta inmediata al error actual
- **Integral**: Elimina error acumulado (offset)
- **Derivativo**: Anticipa cambios futuros

#### Ventajas:
- Respuesta rápida y estable
- Sin error en estado estacionario
- Adaptable a perturbaciones

---

### 4. **Predictive Control** 🔮
Detecta anticipadamente la formación de congestión analizando gradientes de densidad.

#### Algoritmo:
```python
1. Calcular gradiente espacial: ∇ρ(x)
2. Si |∇ρ| > umbral:
   - Detectar onda de choque en formación
   - Identificar zona upstream (aguas arriba)
3. Aplicar control preventivo en zona upstream:
   - Reducir velocidad gradualmente
   - Objetivo: Suavizar transición
```

#### Ventanas de Predicción:
- **Horizonte temporal**: 5-10 pasos (~0.05-0.1 h)
- **Umbral de gradiente**: 50 veh/km²

#### Beneficios:
- Control anticipatorio (no reactivo)
- Prevención vs. corrección
- Menor impacto en conductores

---

### 5. **Zone-Based Control** 🗺️
Divide la vía en zonas geográficas con estrategias de control diferenciadas.

#### Implementación:
```
Zona 1 (0-3.33 km):   Control según métricas locales
Zona 2 (3.33-6.67 km): Control independiente
Zona 3 (6.67-10 km):   Control específico
```

#### Criterios por Zona:
| Densidad Promedio | Acción                    |
|-------------------|---------------------------|
| ρ < 50 veh/km     | Sin control               |
| 50 < ρ < 75       | VSL suave (85% v_max)     |
| 75 < ρ < 100      | VSL moderado (70% v_max)  |
| ρ > 100           | VSL agresivo (50% v_max)  |

#### Ventajas:
- Adaptación local
- Eficiencia energética
- Control granular

---

### 6. **Hybrid Control** 🔄
Combina múltiples estrategias para control óptimo.

#### Combinación VSL + Predictivo:
```
v_control = min(v_vsl, v_predictivo)
→ Toma el control más restrictivo (seguro)
```

#### Ventajas:
- Robustez ante múltiples escenarios
- Complementariedad de estrategias
- Mejor desempeño global

---

## 🚀 Cómo Usar

### Ejecución Básica

```bash
# Ejecutar análisis completo de control
cd Traffic_simulator
python experiments/control_analysis.py
```

### Uso Programático

```python
from src.analysis.control_strategies import apply_integrated_control
from src.utils.parameters import get_spatial_grid, V_MAX, RHO_MAX

# Configurar malla espacial
x = get_spatial_grid(L=10.0, dx=0.1)

# Supongamos que tenemos densidad actual
rho_actual = np.array([...])  # Densidad en cada punto

# Aplicar estrategia VSL
resultado = apply_integrated_control(
    rho=rho_actual,
    x=x,
    t=0.0,
    v_max=V_MAX,
    rho_max=RHO_MAX,
    strategy='vsl'  # Opciones: 'vsl', 'predictive', 'zone', 'hybrid'
)

# Obtener velocidad controlada
v_controlada = resultado['v_controlled']
```

### Estrategias Disponibles

```python
strategies = [
    'vsl',              # Variable Speed Limit (suave)
    'vsl_aggressive',   # VSL agresivo
    'predictive',       # Control predictivo
    'zone',             # Control por zonas
    'hybrid',           # Híbrido (VSL + Predictivo)
    'none'              # Sin control (baseline)
]
```

---

## 📊 Resultados y Análisis

### Escenarios Evaluados

#### 1. **Onda de Choque**
- **Condición inicial**: Discontinuidad (ρ_upstream=140, ρ_downstream=30)
- **Mejor estrategia**: VSL Agresivo
- **Mejora**: 4-5% en tiempo de viaje

#### 2. **Perturbación Gaussiana**
- **Condición inicial**: Pulso localizado
- **Mejor estrategia**: Control Predictivo
- **Mejora**: Reducción 2-3% en congestión

#### 3. **Tráfico Periódico**
- **Condición inicial**: Variación sinusoidal
- **Mejor estrategia**: Control por Zonas
- **Mejora**: 2% en tiempo de viaje

### Métricas de Desempeño

```
📈 Métricas Evaluadas:
- Tiempo de viaje promedio (min)
- Velocidad promedio (km/h)
- Nivel de congestión (%)
- Throughput (veh/h)
- Total Vehicle-Hours (TVH)
- Total Vehicle-Kilometers (TVK)
```

### Visualizaciones Generadas

El script `control_analysis.py` genera:

1. **Mapas de calor comparativos**
   - Densidad con/sin control
   - Velocidad controlada

2. **Diagramas espacio-tiempo**
   - Evolución temporal de densidad
   - Zonas de control activas

3. **Gráficas de métricas**
   - Comparación por estrategia
   - Mejoras porcentuales

4. **Reporte comprehensivo**
   - Archivo TXT con todas las métricas
   - Comparación cuantitativa

### Ubicación de Resultados

```
Traffic_simulator/
├── results/
│   ├── figures/
│   │   └── control_analysis/
│   │       ├── comprehensive_comparison.png
│   │       └── comparisons/
│   │           ├── scenario1_shock_control.png
│   │           ├── scenario2_predictive_control.png
│   │           └── scenario3_zone_control.png
│   └── metrics/
│       └── control_analysis_report.txt
```

---

## 🔍 Análisis de Casos de Uso

### Caso 1: Autopista Urbana (Alta Demanda)
**Problema**: Congestión recurrente en horas pico

**Estrategia Recomendada**: Ramp Metering + VSL
- Controlar entradas para mantener ρ ≈ 75 veh/km
- VSL para suavizar transiciones

**Beneficio Esperado**: 10-15% reducción en tiempo de viaje

---

### Caso 2: Zona de Obras (Capacidad Reducida)
**Problema**: Cuello de botella por construcción

**Estrategia Recomendada**: Predictive Control
- Detectar congestión upstream
- Reducir velocidad anticipadamente

**Beneficio Esperado**: 20-25% reducción en frenados bruscos

---

### Caso 3: Tráfico Variable (Día/Noche)
**Problema**: Patrones cambiantes de demanda

**Estrategia Recomendada**: Zone-Based Control
- Control adaptativo por zona horaria
- Eficiencia energética en períodos bajos

**Beneficio Esperado**: 5-10% mejora en throughput promedio

### Calibración de Parámetros

```python
# Ejemplo: Ajustar umbrales de VSL

vsl = VariableSpeedLimit(v_max_default=100.0)

# Calibrar umbrales según datos reales
vsl.rho_warning = 0.65 * RHO_MAX  # 97.5 veh/km
vsl.rho_danger = 0.75 * RHO_MAX   # 112.5 veh/km

# Probar en simulación
results = vsl.apply_control(rho, x, aggressive=False)
```