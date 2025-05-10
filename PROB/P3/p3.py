import numpy as np
import nashpy as nash

def print_equilibria(game_num, A, B, row_labels, col_labels):
    """
    Calcula i mostra els equilibris de Nash per estratègies mixtes d'un joc bimatriu.
    
    Paràmetres:
    game_num (int): Número del problema/joc
    A (numpy.ndarray): Matriu de pagaments del jugador fila
    B (numpy.ndarray): Matriu de pagaments del jugador columna
    row_labels (list): Etiquetes per les estratègies del jugador fila
    col_labels (list): Etiquetes per les estratègies del jugador columna
    """
    # Formació del joc amb ambdues matrius de pagament
    game = nash.Game(A, B)
    
    print(f"Problema {game_num}: {row_labels} x {col_labels}")
    print("Matriu de pagaments Jugador 1:")
    print(A)
    print("Matriu de pagaments Jugador 2:")
    print(B)
    
    # Càlcul dels equilibris Nash
    equilibria = list(game.support_enumeration())
    
    if not equilibria:
        print("No s'han trobat equilibris de Nash per estratègies mixtes.")
        return
    
    for i, equilibrium in enumerate(equilibria):
        row_strategy, col_strategy = equilibrium
        
        # Mostrar l'estratègia mixta per a cada jugador
        print(f"\nEquilibri {i+1}:")
        
        print("Estratègia Jugador 1:", end=" ")
        for j, prob in enumerate(row_strategy):
            print(f"{row_labels[j]}: {prob:.4f}", end=" ")
        
        print("\nEstratègia Jugador 2:", end=" ")
        for j, prob in enumerate(col_strategy):
            print(f"{col_labels[j]}: {prob:.4f}", end=" ")
        
        # Càlcul de les recompenses esperades
        expected_payoff_row = row_strategy @ A @ col_strategy
        expected_payoff_col = row_strategy @ B @ col_strategy
        
        print(f"\nRecompensa esperada Jugador 1: {expected_payoff_row:.4f}")
        print(f"Recompensa esperada Jugador 2: {expected_payoff_col:.4f}")
    
    print("\n" + "-"*50 + "\n")

# Problema 2: Joc de Chicken (Hawk-Dove)
A2 = np.array([[-1, 2], [1, 0]])      # Pagaments Jugador 1
B2 = np.array([[-1, 1], [2, 0]])      # Pagaments Jugador 2
print_equilibria(2, A2, B2, ["F", "D"], ["F", "D"])

# Problema 3: Joc de Coordinació
A3 = np.array([[5, 1], [1, 3]])       # Pagaments Jugador 1
B3 = np.array([[5, 1], [1, 3]])       # Pagaments Jugador 2 (simètric)
print_equilibria(3, A3, B3, ["A", "B"], ["A", "B"])

# Problema 4: Joc de Coordinació Arriscada
A4 = np.array([[5, 2], [-1, 3]])      # Pagaments Jugador 1
B4 = np.array([[5, -1], [2, 3]])      # Pagaments Jugador 2
print_equilibria(4, A4, B4, ["N", "O"], ["N", "O"])

# Problema 5: Joc de Coordinació més Arriscada
A5 = np.array([[1, 3], [4, 0]])       # Pagaments Jugador 1
B5 = np.array([[-1, 0], [2, -1]])     # Pagaments Jugador 2
print_equilibria(5, A5, B5, ["U", "D"], ["L", "R"])

# Problema 6: Joc de Cooperació vs Competició
A6 = np.array([[1, -2, 0], [-2, 1, 0], [0, 0, 1]])     # Pagaments Jugador 1
B6 = np.array([[-2, 1, 0], [1, -2, 0], [0, 0, 1]])     # Pagaments Jugador 2
print_equilibria(6, A6, B6, ["A", "B", "C"], ["D", "E", "F"])