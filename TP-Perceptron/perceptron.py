import numpy as np
import matplotlib.pyplot as plt
from pylab import rand

def generateData(n):
    xb = (rand(n) * 2 - 1) / 2 - 0.5
    yb = (rand(n) * 2 - 1) / 2 + 0.5
    xr = (rand(n) * 2 - 1) / 2 + 0.5
    yr = (rand(n) * 2 - 1) / 2 - 0.5

    inputs = []
    for i in range(n):
        inputs.append([xb[i], yb[i], -1])
        inputs.append([xr[i], yr[i], 1])

    data = np.array(inputs)
    X = data[:, 0:2]
    Y = data[:, -1]

    return X, Y

# Génération des données
X, Y = generateData(100)

# Affichage des données
plt.figure(figsize=(8, 6))
plt.scatter(X[Y == -1, 0], X[Y == -1, 1], color='blue', marker='o', label='Classe -1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='red', marker='x', label='Classe 1')
plt.legend()
plt.title("Données générées")
plt.show()


def perceptron(X, Y, max_iter=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialisation du vecteur de poids
    
    for _ in range(max_iter):
        error = False  # Vérifier si des erreurs persistent
        for i in range(n_samples):
            if Y[i] * np.dot(w, X[i]) <= 0:  # Exemple mal classé
                w += Y[i] * X[i]  # Mise à jour du poids
                error = True
        if not error:
            break  # Si tous les exemples sont bien classés, on arrête
    
    return w

# Génération des données d'apprentissage
X, Y = generateData(100)

# Entraînement du perceptron
w = perceptron(X, Y)

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.scatter(X[Y == -1, 0], X[Y == -1, 1], color='blue', marker='o', label='Classe -1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='red', marker='x', label='Classe 1')

# Tracer la frontière de séparation (droite w.x = 0)
x_vals = np.linspace(-1, 1, 100)
y_vals = - (w[0] / w[1]) * x_vals  # Équation de la droite séparatrice

plt.plot(x_vals, y_vals, 'k-', label="Frontière de décision")
plt.legend()
plt.title("Classification avec le Perceptron")
plt.show()

def predict(X, w):
    return np.sign(np.dot(X, w))

# Génération du jeu de données test
X_test, Y_test = generateData(50)

# Prédiction sur les données de test
Y_pred = predict(X_test, w)

# Calcul de l'erreur de classification
error_rate = np.mean(Y_pred != Y_test)  # Pourcentage d'exemples mal classés

print(f"Erreur de classification : {error_rate * 100:.2f}%")

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.scatter(X_test[Y_test == -1, 0], X_test[Y_test == -1, 1], color='blue', marker='o', label='Classe -1 (Réelle)')
plt.scatter(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1], color='red', marker='x', label='Classe 1 (Réelle)')

# Afficher la frontière de décision
x_vals = np.linspace(-1, 1, 100)
y_vals = - (w[0] / w[1]) * x_vals

plt.plot(x_vals, y_vals, 'k-', label="Frontière de décision")
plt.legend()
plt.title(f"Test du Perceptron - Erreur : {error_rate * 100:.2f}%")
plt.show()


