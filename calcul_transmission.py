import numpy as np
import math
import matplotlib.pyplot as plt
import time

start_time = time.time()

dt = 1E-7
dx = 0.001
nx = int(2/dx)
nt = 90000
s = dt / (dx**2)

xc = 0.6
sigma = 0.05
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))

v0 = -4000
debut_puits = 0.8
largeur_puits = 0.1
fin_puits = debut_puits + largeur_puits

o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= debut_puits) & (o <= fin_puits)] = v0

#simulation pour calcul transmission
def calcul_transmission(E):
    k = math.sqrt(2 * E)
    cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
    re = np.real(cpt) # partie réelle
    im = np.imag(cpt) # partie imaginaire
    dens = np.sum(np.abs(cpt)**2) * dx

    for i in range(1, nt):
        if i % 2 != 0:
            im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2] - 2 * re[1:-1]) - 2 * re[1:-1] * V[1:-1] * dt
        else:
            re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2] - 2 * im[1:-1]) + 2 * im[1:-1] * V[1:-1] * dt

    densite_finale = re**2 + im**2
    borne_transmission = int((fin_puits + 0.1) / dx)
    transmission = np.sum(densite_finale[borne_transmission:]) * dx / dens
    return transmission

# calcul analytique
def transmission_analytique(E_vecteur, v0, a):
    T = [] # tab des transm
    V0 = abs(v0)
    for E in E_vecteur:
        if E <= 0:
            T.append(0) # état lié
            continue
        q = np.sqrt(2 * (E + V0))
        denom = 4 * E * (E + V0)
        if denom == 0:
            T.append(0)
        else:
            T.append(1 / (1 + ((V0**2) * (np.sin(q * a))**2) / denom))
    return np.array(T)

# Simulation pour 100 énergies
energies = np.linspace(100, 4500, 100)
trans_num = []

for E in energies:
    if E > 0:
        T_simule = calcul_transmission(E)
        trans_num.append(T_simule)
    else:
        trans_num.append(0) # E <= 0 (état lié)

trans_num = np.array(trans_num)
trans_theo = transmission_analytique(energies, v0, largeur_puits)

plt.figure(figsize=(12, 7))
plt.plot(energies, trans_theo, 'b-', label="Transmission Analytique")
plt.plot(energies, trans_num, 'ro', markersize=4, label="Transmission Numérique")
plt.axhline(y=1, linestyle="--", color="g", alpha=0.5, label="T=1")
plt.title("Coefficient de Transmission en fonction de l'Énergie (puits de potentiel)")
plt.xlabel("Énergie")
plt.ylabel("Transmission")
plt.grid(True)
plt.legend()
plt.show()

end_time = time.time()
print(f"\nTemps d'exécution : {end_time - start_time:.2f} secondes")