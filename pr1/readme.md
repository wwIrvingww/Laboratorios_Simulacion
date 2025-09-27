# Proyecto 1: Algoritmos Genéticos + LP - TSP

## Descripción del Proyecto

Este proyecto implementa la resolución del **Traveling Salesman Problem (TSP)** utilizando dos enfoques:
- **Algoritmo Genético (GA)**: Implementación personalizada con parámetros configurables
- **Programación Lineal Entera (ILP)**: Utilizando PuLP en Python

## Escenarios Trabajados

1. **eil101**: 101 ciudades del repositorio TSPLIB95
2. **gr229**: 229 ciudades del repositorio TSPLIB95  
3. **3case**: Escenario personalizado con 3 ciudades (caso de prueba)

## Estructura del Proyecto

```
pr1/
├── gn.py              # Implementación del Algoritmo Genético
├── pl.py              # Implementación de Programación Lineal (ILP)
├── pl2.py             # Versión alternativa de ILP
├── analisys.py        # Script de análisis y comparación
├── data/              # Datos de entrada
│   ├── 3case.tsp/
│   ├── eil101.tsp/
│   └── gr229.tsp/
└── data/solutions/    # Resultados generados
```

## Cómo Ejecutar

### 1. Algoritmo Genético
```bash
python gn.py
```

### 2. Programación Lineal
```bash
python pl.py
```

### 3. Análisis Comparativo
```bash
python analisys.py
```

## Resultados Guardados

### Archivos CSV
- `ga_solutions.csv`: Soluciones del algoritmo genético
- `ilp_solutions_detailed.csv`: Soluciones detalladas de ILP
- `ilp_solutions_ga_format.csv`: Soluciones ILP en formato comparable
- `tabla_comparativa_inciso3.csv`: Tabla comparativa final

### Visualizaciones
- `comparacion_ga_vs_lp.png`: Gráfico comparativo de ambos métodos
- `/pngs/`: Visualizaciones individuales de rutas óptimas por escenario

### Resultados Principales
- **GA**: Soluciones sub-óptimas con tiempos de ejecución variables
- **ILP**: Soluciones óptimas garantizadas con mayor costo computacional
- **Comparación**: El ILP encuentra mejores soluciones pero GA es más eficiente en tiempo

## Parámetros del GA Utilizados
- Población: 100 individuos
- Iteraciones máximas: 500
- Selección: 50% de supervivientes
- Cruce: 40% de descendencia
- Mutación: 10% de la población