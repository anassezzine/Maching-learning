import numpy as np
import matplotlib.pyplot as plt
from pylab import rand


def generateData2(n):
    xb = (rand(n) * 2 - 1) / 2 + 0.5
    yb = (rand(n) * 2 - 1) / 2
    xr = (rand(n) * 2 - 1) / 2 + 1.5
    yr = (rand(n) * 2 - 1) / 2 - 0.5

    inputs = []
    for i in range(n):
        inputs.append([xb[i], yb[i], -1])
        inputs.append([xr[i], yr[i], 1])

    data = np.array(inputs)
    X = data[:, 0:2]
    Y = data[:, -1]

    return X, Y

# Génération et affichage des nouvelles données
X2, Y2 = generateData2(100)

plt.figure(figsize=(8, 6))
plt.scatter(X2[Y2 == -1, 0], X2[Y2 == -1, 1], color='blue', marker='o', label='Classe -1')
plt.scatter(X2[Y2 == 1, 0], X2[Y2 == 1, 1], color='red', marker='x', label='Classe 1')
plt.legend()
plt.title("Données générées (non séparables par une droite passant par l'origine)")
plt.show()

def add_bias(X):
    n_samples = X.shape[0]
    bias_column = np.ones((n_samples, 1))  # Colonne de 1
    return np.hstack((X, bias_column))  # Concaténation avec X

# Ajouter le biais aux données
X2_biased = add_bias(X2)


def perceptron_biased(X, Y, max_iter=1000):
    """
    Implémente le perceptron avec biais.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialisation des poids (y compris le biais)

    for _ in range(max_iter):
        error = False
        for i in range(n_samples):
            if Y[i] * np.dot(w, X[i]) <= 0:  # Vérifier si mal classé
                w += Y[i] * X[i]  # Mise à jour
                error = True
        if not error:
            break

    return w

# Entraînement du perceptron avec biais
w_biased = perceptron_biased(X2_biased, Y2)

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.scatter(X2[Y2 == -1, 0], X2[Y2 == -1, 1], color='blue', marker='o', label='Classe -1')
plt.scatter(X2[Y2 == 1, 0], X2[Y2 == 1, 1], color='red', marker='x', label='Classe 1')

# Tracer la frontière de séparation (droite w.x + b = 0)
x_vals = np.linspace(-1, 2, 100)
y_vals = - (w_biased[0] / w_biased[1]) * x_vals - (w_biased[2] / w_biased[1])  # Forme y = mx + b

plt.plot(x_vals, y_vals, 'k-', label="Frontière de décision (avec biais)")
plt.legend()
plt.title("Perceptron avec Biais")
plt.show()


