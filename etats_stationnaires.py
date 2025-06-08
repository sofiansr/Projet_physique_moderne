import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
# scipy.linalg est une librairie python qui permet de trouver les valeurs/vecteurs propres (et autres...)
# eigh est donc une fonction de cette librairie qui permet de faire ça à partir du hamiltonien en diagonalisant

# on utilise Schrodinger indépendante du temps

dx = 0.001
x = np.arange(0, 2, dx) # x=[0.000, 0.001, 0.002, …, 1.999]
nx = len(x) # 2000

# --- définition du potentiel ---
V = np.zeros(nx)    # V= [0, 0, ... 0] 2000 zéros     <=> potentiel nul partout
V[(x >= 0.8) & (x <= 0.9)] = -4000  # puits de potentiel avec V0=-4000

# --- construction du Hamiltonien ---
hbar = 1  
m = 1   # simplifiés comme dans Sch1d_solution_1.py du prof

# opérateur laplacien = -d²/dx²
laplacien = -2 * np.eye(nx) + np.eye(nx, k=1) + np.eye(nx, k=-1)
laplacien = laplacien / dx**2

# hamiltonien : H = -h/2m * d²/dx² + V
H = -(hbar**2 / (2 * m)) * laplacien + np.diag(V)

# --- résolution du problème aux valeurs propres ---
num_states = 5  # nombre d'états stationnaires pour le graphique
energies, psi = eigh(H, subset_by_index=[0, num_states - 1]) # diagonalisation

# --- normalisation des états ---
areas = []
transmissions = []
borne = int(1.0 / dx)  # x = 1.0
for n in range(num_states):    
    psi[:, n] = psi[:, n] / np.sqrt(np.sum(np.abs(psi[:, n])**2) * dx)
    area = np.sum(psi[:, n]**2) * dx
    trans = np.sum(psi[borne:, n]**2) * dx
    areas.append(area)
    transmissions.append(trans)

# --- génération du graphique ---
plt.figure(figsize=(10, 6))
for n in range(num_states):
    # plt.plot(x, psi[:, n]**2 + energies[n], label=f"État {n}, E={energies[n]:.2f}")
    plt.plot(x, psi[:, n]**2, label=f"État {n}, E={energies[n]:.2f}, Aire={areas[n]:.3f}, T={transmissions[n]:.2f}")
    #plt.plot(x, psi[:, n], label=f"État {n}, E={energies[n]:.2f}, Aire={areas[n]:.3f}")
# plt.plot(x, V, 'k--', label="Potentiel")
plt.plot(x, V / np.abs(np.min(V)) * np.max(psi[:, 0]**2), linestyle='--', label="Potentiel", color="orange")
plt.xlabel("x")
plt.ylabel("|ψ(x)|²") # Densité de probabilité
plt.title("États stationnaires dans un puits de potentiel")
plt.legend()
plt.grid()
plt.show()



# pour rappel :
# la diagonalisation d’une matrice H  consiste à trouver :
# les valeurs propres E (énergies des états stationnaires) (1ere valeur retournée de "eigh")
# les vecteurs propres ψ (fonctions d’onde associées à ces énergies) (2ème valeur)