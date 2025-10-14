import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("="*70)
    print("INCISO 3: TABLA COMPARATIVA GA vs LP")
    print("="*70)
    
    current_dir = os.getcwd()
    print(f"Directorio actual: {current_dir}")
    
    solutions_dir = os.path.join("data", "solutions")
    ga_file = os.path.join(solutions_dir, "ga_solutions.csv")
    lp_file = os.path.join(solutions_dir, "ilp_solutions_ga_format.csv")
    
    print(f"\nBuscando archivos en: {os.path.abspath(solutions_dir)}")
    print(f"GA file: {ga_file} - Existe: {os.path.exists(ga_file)}")
    print(f"LP file: {lp_file} - Existe: {os.path.exists(lp_file)}")
    
    ga_results = None
    lp_results = None
    
    try:
        if os.path.exists(ga_file):
            ga_results = pd.read_csv(ga_file)
            print(f"✓ Resultados GA cargados: {len(ga_results)} filas")
        else:
            print(f"❌ No se encontró: {ga_file}")
            
        if os.path.exists(lp_file):
            lp_results = pd.read_csv(lp_file)
            print(f"✓ Resultados LP cargados: {len(lp_results)} filas")
        else:
            print(f"❌ No se encontró: {lp_file}")
            
    except Exception as e:
        print(f"Error cargando archivos: {e}")
        return
    
    if ga_results is None and lp_results is None:
        print("\n❌ No se encontraron archivos de resultados.")
        print("Asegúrate de haber ejecutado:")
        print("1. python gn.py")
        print("2. python pl2.py")
        return
    
    # Crear tabla comparativa
    tabla_comparativa = []
    
    problemas = set()
    if ga_results is not None:
        problemas.update(ga_results['problema'].unique())
    if lp_results is not None:
        problemas.update(lp_results['problema'].unique())
    
    problemas = sorted(list(problemas))
    print(f"\nProblemas encontrados: {problemas}")
    
    for problema in problemas:
        print(f"\n--- Procesando {problema} ---")
        
        # Datos GA
        ga_data = ga_results[ga_results['problema'] == problema] if ga_results is not None else pd.DataFrame()
        lp_data = lp_results[lp_results['problema'] == problema] if lp_results is not None else pd.DataFrame()
        
        # Información básica del problema
        if not ga_data.empty:
            num_ciudades = ga_data.iloc[0]['num_ciudades']
            poblacion_ga = ga_data.iloc[0]['poblacion_ga']
            iteraciones_ga = ga_data.iloc[0]['iteraciones_ga']
        elif not lp_data.empty:
            num_ciudades = lp_data.iloc[0]['num_ciudades']
            poblacion_ga = "N/A"
            iteraciones_ga = "N/A"
        else:
            continue
        
        # Calcular variables y restricciones LP
        num_variables_lp = num_ciudades * (num_ciudades - 1) + num_ciudades  # x_ij + u_i
        num_restricciones_lp = 2 * num_ciudades + (num_ciudades - 1) * (num_ciudades - 2)
        
        # Agregar resultado LP
        if not lp_data.empty:
            lp_row = lp_data.iloc[0]
            tabla_comparativa.append({
                'Problema': problema,
                'Num_Ciudades': num_ciudades,
                'Poblacion_GA': "N/A",
                'Iteraciones_GA': "N/A", 
                'Num_Variables_LP': num_variables_lp,
                'Num_Restricciones_LP': num_restricciones_lp,
                'Tipo_Solucion': 'LP',
                'Tiempo_Ejecucion': lp_row['tiempo_ejecucion'],
                'Solucion_Optima_Teorica': lp_row['distancia_suboptima'],
                'Solucion_Suboptima_Encontrada': lp_row['distancia_suboptima'],
                'Porcentaje_Error': 0.0
            })
            lp_optima = lp_row['distancia_suboptima']
        else:
            lp_optima = None
        
        # Agregar mejores 3 resultados GA
        if not ga_data.empty:
            ga_sorted = ga_data.sort_values('distancia_suboptima').head(3)
            
            for idx, ga_row in ga_sorted.iterrows():
                error_pct = 0.0
                if lp_optima is not None:
                    error_pct = abs((ga_row['distancia_suboptima'] - lp_optima) / lp_optima) * 100
                
                tabla_comparativa.append({
                    'Problema': problema,
                    'Num_Ciudades': num_ciudades,
                    'Poblacion_GA': ga_row['poblacion_ga'],
                    'Iteraciones_GA': ga_row['iteraciones_ga'],
                    'Num_Variables_LP': num_variables_lp,
                    'Num_Restricciones_LP': num_restricciones_lp,
                    'Tipo_Solucion': 'GA',
                    'Tiempo_Ejecucion': ga_row['tiempo_ejecucion'],
                    'Solucion_Optima_Teorica': lp_optima if lp_optima else "N/A",
                    'Solucion_Suboptima_Encontrada': ga_row['distancia_suboptima'],
                    'Porcentaje_Error': error_pct
                })
    
    # Crear DataFrame final
    df_final = pd.DataFrame(tabla_comparativa)
    
    if df_final.empty:
        print("❌ No se pudo crear la tabla comparativa")
        return
    
    print("\n" + "="*120)
    print("TABLA COMPARATIVA FINAL - INCISO 3")
    print("="*120)
    print(df_final.to_string(index=False))
    
    # Guardar resultados
    os.makedirs(solutions_dir, exist_ok=True)
    output_file = os.path.join(solutions_dir, "tabla_comparativa_inciso3.csv")
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ Tabla guardada en: {output_file}")
    
    print("\n" + "="*70)
    print("RESUMEN POR PROBLEMA")
    print("="*70)
    
    for problema in problemas:
        print(f"\n--- {problema} ---")
        problema_data = df_final[df_final['Problema'] == problema]
        
        lp_data = problema_data[problema_data['Tipo_Solucion'] == 'LP']
        ga_data = problema_data[problema_data['Tipo_Solucion'] == 'GA']
        
        if not problema_data.empty:
            num_ciudades = problema_data.iloc[0]['Num_Ciudades']
            num_vars = problema_data.iloc[0]['Num_Variables_LP']
            num_restric = problema_data.iloc[0]['Num_Restricciones_LP']
            
            print(f"Ciudades: {num_ciudades}")
            print(f"Variables LP: {num_vars}")
            print(f"Restricciones LP: {num_restric}")
        
        if not lp_data.empty:
            lp_info = lp_data.iloc[0]
            print(f"LP: Tiempo={lp_info['Tiempo_Ejecucion']:.2f}s, Distancia={lp_info['Solucion_Suboptima_Encontrada']:.2f}")
        
        if not ga_data.empty:
            mejor_ga = ga_data.loc[ga_data['Solucion_Suboptima_Encontrada'].idxmin()]
            print(f"GA (mejor): Tiempo={mejor_ga['Tiempo_Ejecucion']:.2f}s, "
                  f"Distancia={mejor_ga['Solucion_Suboptima_Encontrada']:.2f}, "
                  f"Error={mejor_ga['Porcentaje_Error']:.2f}%")
            print(f"GA config: Población={mejor_ga['Poblacion_GA']}, Iteraciones={mejor_ga['Iteraciones_GA']}")
    
    # Crear gráfico simple
    try:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for problema in problemas:
            prob_data = df_final[df_final['Problema'] == problema]
            ga_times = prob_data[prob_data['Tipo_Solucion'] == 'GA']['Tiempo_Ejecucion']
            lp_times = prob_data[prob_data['Tipo_Solucion'] == 'LP']['Tiempo_Ejecucion']
            
            if len(ga_times) > 0:
                plt.scatter([problema] * len(ga_times), ga_times, alpha=0.7, label='GA' if problema == problemas[0] else "", color='blue')
            if len(lp_times) > 0:
                plt.scatter([problema] * len(lp_times), lp_times, alpha=0.7, label='LP' if problema == problemas[0] else "", color='red', marker='s')
        
        plt.ylabel('Tiempo (s)')
        plt.title('Tiempos de Ejecución')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        for problema in problemas:
            prob_data = df_final[df_final['Problema'] == problema]
            ga_dist = prob_data[prob_data['Tipo_Solucion'] == 'GA']['Solucion_Suboptima_Encontrada']
            lp_dist = prob_data[prob_data['Tipo_Solucion'] == 'LP']['Solucion_Suboptima_Encontrada']
            
            if len(ga_dist) > 0:
                plt.scatter([problema] * len(ga_dist), ga_dist, alpha=0.7, color='blue')
            if len(lp_dist) > 0:
                plt.scatter([problema] * len(lp_dist), lp_dist, alpha=0.7, color='red', marker='s')
        
        plt.ylabel('Distancia Total')
        plt.title('Calidad de Soluciones')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        errores_ga = df_final[df_final['Tipo_Solucion'] == 'GA']['Porcentaje_Error']
        if len(errores_ga) > 0:
            plt.hist(errores_ga, bins=min(10, len(errores_ga)), alpha=0.7, color='orange')
        plt.xlabel('Error (%)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Errores GA')
        
        plt.subplot(2, 2, 4)
        sizes = df_final.groupby('Problema')['Num_Ciudades'].first()
        plt.bar(sizes.index, sizes.values, alpha=0.7, color='green')
        plt.ylabel('Num. Ciudades')
        plt.title('Tamaño de Problemas')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        img_path = os.path.join(solutions_dir, "comparacion_ga_vs_lp.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {img_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creando gráfico: {e}")
    
    print("\n" + "="*70)
    print("ANALISIS COMPLETADO")
    print("Archivos generados:")
    print(f"- {output_file}")
    print(f"- {os.path.join(solutions_dir, 'comparacion_ga_vs_lp.png')}")
    print("="*70)

if __name__ == "__main__":
    main()