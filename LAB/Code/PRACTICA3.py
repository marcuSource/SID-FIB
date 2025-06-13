import subprocess
import sys

EXPERIMENT_SCRIPTS = [
    "Experiment1.py",
    "Experiment2.py",
    "Experiment3.py",
    "Experiment4.py",
    "Experiment5.py",
    "Experiment6.py",
]

MENU = '''\nPRACTICA3 - Experimentos SID\n===========================\n1. Ejecutar TODOS los experimentos\n2. Ejecutar un experimento específico\n3. Salir\n'''

EXPERIMENT_NAMES = [
    "Experimento 1",
    "Experimento 2",
    "Experimento 3",
    "Experimento 4",
    "Experimento 5",
    "Experimento 6",
]

def run_script(script):
    print(f"\nEjecutando {script} ...\n")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode == 0:
        print(f"\n{script} finalizado correctamente.\n")
    else:
        print(f"\nError al ejecutar {script}.\n")

def run_all():
    for script in EXPERIMENT_SCRIPTS:
        run_script(script)

def run_specific():
    print("\nSeleccione el experimento a ejecutar:")
    for idx, name in enumerate(EXPERIMENT_NAMES, 1):
        print(f"{idx}. {name}")
    try:
        choice = int(input("Ingrese el número del experimento: "))
        if 1 <= choice <= len(EXPERIMENT_SCRIPTS):
            run_script(EXPERIMENT_SCRIPTS[choice-1])
        else:
            print("Opción inválida.")
    except ValueError:
        print("Entrada inválida.")

def main():
    while True:
        print(MENU)
        option = input("Seleccione una opción: ").strip()
        if option == '1':
            run_all()
        elif option == '2':
            run_specific()
        elif option == '3':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()
