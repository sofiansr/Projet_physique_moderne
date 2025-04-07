import numpy as np
import matplotlib.pyplot as plt

# Paramètres
h = 0.1               # pas de temps
T = 10                # durée totale
N = int(T / h)        # nombre de pas
t = np.linspace(0, T, N+1)

# Initialisation des tableaux
y = np.zeros(N+1)
y[0] = 1              # y(0)
y_prime_0 = 0         # y'(0)

# Calcul de y[1] avec Taylor (ordre 2)
y[1] = y[0] + h * y_prime_0 + 0.5 * h**2 * (-y[0])  # car y'' = -y

# Algorithme de Verlet
for n in range(1, N):
    y[n+1] = 2 * y[n] - y[n-1] - h**2 * y[n]  # f(t, y) = -y

# Solution exacte pour comparaison
y_exact = np.cos(t)

# Affichage
plt.plot(t, y, label='Verlet (numérique)', linestyle='--')
plt.plot(t, y_exact, label='Solution exacte', alpha=0.7)
plt.legend()
plt.xlabel('Temps t')
plt.ylabel('y(t)')
plt.title('Méthode de Verlet : $y\'\' = -y$')
plt.grid(True)
plt.show()
