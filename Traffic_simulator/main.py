import os
import sys
import subprocess
import importlib.util


def print_header(title):
    """Imprime un encabezado formateado."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_menu():
    """Imprime el menu de opciones."""
    print("\n" + "-" * 80)
    print("MENU DE EXPERIMENTOS - SIMULACION DE TRAFICO VEHICULAR".center(80))
    print("-" * 80)
    print("\n1. Escenarios Macroscopicos        - Simulacion modelo macroscopico")
    print("2. Escenarios Microscopicos        - Simulacion modelo microscopico")
    print("3. Generar Animaciones             - Crear GIFs de simulaciones")
    print("4. Analisis Comparativo            - Comparar modelos macroscopico vs microscopico")
    print("5. Ver Animaciones                 - Visualizar GIFs generados")
    print("\n0. Salir")
    print("\n" + "-" * 80)


def get_experiment_info():
    """Obtiene informacion sobre los experimentos disponibles."""
    experiments = {
        '1': {
            'name': 'Escenarios Macroscopicos',
            'file': 'experiments/macroscopic_scenarios.py',
            'description': 'Ejecuta simulaciones del modelo macroscopico para diferentes\nescenarios iniciales'
        },
        '2': {
            'name': 'Escenarios Microscopicos',
            'file': 'experiments/microscopic_scenarios.py',
            'description': 'Ejecuta simulaciones del modelo microscopico (IDM)\npara varios escenarios de trafico'
        },
        '3': {
            'name': 'Generar Animaciones',
            'file': 'experiments/generate_animations.py',
            'description': 'Genera animaciones en formato GIF a partir de\nsimulaciones macroscopicas'
        },
        '4': {
            'name': 'Analisis Comparativo',
            'file': 'experiments/comparative_analysis.py',
            'description': 'Realiza un analisis comparativo entre los modelos\nmacroscopico y microscopico'
        },
        '5': {
            'name': 'Ver Animaciones',
            'file': 'experiments/view_animation.py',
            'description': 'Permite visualizar las animaciones generadas'
        }
    }
    return experiments


def check_file_exists(filepath):
    """Verifica si el archivo del experimento existe."""
    if not os.path.exists(filepath):
        print(f"\nError: Archivo no encontrado - {filepath}")
        return False
    return True


def run_experiment(filepath, scenario=None):
    """
    Ejecuta un experimento Python.

    Parametros:
        filepath: Ruta del archivo a ejecutar
        scenario: Numero de escenario opcional (1-7) para opciones 1 y 2
    """
    try:
        print(f"\nEjecutando: {filepath}\n")
        print("=" * 80)

        cmd = [sys.executable, filepath]

        # Agregar argumentos de escenario si aplica
        if scenario is not None:
            cmd.extend(['--scenario', str(scenario)])

        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
        )

        print("=" * 80)

        if result.returncode == 0:
            print("\nOK Experimento completado exitosamente")
        else:
            print(f"\nError: El experimento finalizo con codigo de error {result.returncode}")
            return False

        return True

    except Exception as e:
        print(f"\nError al ejecutar el experimento: {str(e)}")
        return False


def show_experiment_details(choice, experiments):
    """Muestra detalles del experimento seleccionado."""
    if choice not in experiments:
        return False

    exp = experiments[choice]
    print(f"\n{'='*80}")
    print(f"Experimento: {exp['name']}")
    print(f"{'='*80}")
    print(f"\nDescripcion:")
    print(f"{exp['description']}")
    print(f"\nArchivo: {exp['file']}")
    return True


def ask_scenario_selection():
    """Pregunta al usuario que escenario desea ejecutar."""
    print(f"\n{'-'*80}")
    print("SELECCIONAR ESCENARIO")
    print(f"{'-'*80}\n")

    scenarios = [
        ('1', 'Escenario 1: Flujo Libre'),
        ('2', 'Escenario 2: Flujo Moderado'),
        ('3', 'Escenario 3: Flujo Congestionado'),
        ('4', 'Escenario 4: Perturbacion Gaussiana'),
        ('5', 'Escenario 5: Perturbacion Sinusoidal'),
        ('6', 'Escenario 6: Dos Grupos'),
        ('7', 'Escenario 7: Gradiente Lineal'),
        ('0', 'Ejecutar TODOS los escenarios')
    ]

    for num, name in scenarios:
        print(f"  {num}. {name}")

    print(f"\n{'-'*80}")
    choice = input("Selecciona un escenario (0-7): ").strip()

    if choice not in ['0', '1', '2', '3', '4', '5', '6', '7']:
        print("\nError: Opcion no valida.")
        return None

    if choice == '0':
        return None  # None significa ejecutar todos

    return int(choice)


def main():
    """Funcion principal que ejecuta el menu interactivo."""

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    print_header("SIMULADOR DE TRAFICO VEHICULAR")
    print("\nBienvenido al simulador de trafico vehicular.")
    print("Este menu permite ejecutar diferentes experimentos de simulacion.")
    print("\nModelos implementados:")
    print("  - Macroscopico: Ecuacion de Conservacion + Greenshields")
    print("  - Microscopico: Intelligent Driver Model (IDM)")

    experiments = get_experiment_info()

    while True:
        print_menu()

        choice = input("Selecciona una opcion (0-5): ").strip()

        if choice == '0':
            print("\nGracias por usar el simulador. Hasta luego!")
            print("=" * 80 + "\n")
            break

        if choice not in experiments:
            print("\nError: Opcion no valida. Por favor, elige un numero entre 0 y 5.")
            continue

        # Mostrar detalles del experimento
        show_experiment_details(choice, experiments)

        exp_file = experiments[choice]['file']

        # Verificar que el archivo existe
        if not check_file_exists(exp_file):
            print("\nPresiona Enter para volver al menu...")
            input()
            continue

        # Para opciones 1 y 2 (escenarios), pedir seleccion de escenario
        scenario = None
        if choice in ['1', '2']:
            scenario = ask_scenario_selection()
            if scenario is None and choice in ['1', '2']:
                # None significa ejecutar todos
                scenario = None

        # Pedir confirmacion
        print(f"\nDeseas ejecutar este experimento? (s/n): ", end='')
        confirm = input().strip().lower()

        if confirm != 's':
            print("Cancelado.")
            print("\nPresiona Enter para volver al menu...")
            input()
            continue

        # Ejecutar el experimento (pasar scenario si aplica)
        success = run_experiment(exp_file, scenario=scenario)

        print("\nPresiona Enter para volver al menu...")
        input()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"\nError no esperado: {str(e)}")
        print("=" * 80 + "\n")
