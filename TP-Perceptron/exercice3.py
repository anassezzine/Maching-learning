import matplotlib.pyplot as plt
import numpy as np

# Chargement des données d'entraînement
with open("./TP3_data/learn.data", "r") as f:
    training_set = [eval(line) for line in f]

X = np.array([data[0] for data in training_set])
Y = np.array([data[1] for data in training_set])

# Chargement des données de test
with open("./TP3_data/test.data", "r") as f:
    test_set = [eval(line) for line in f]

X_test = np.array([data[0] for data in test_set])
Y_test = np.array([data[1] for data in test_set])


### 🌟 Définition du noyau gaussien
def kernel_gaussian(x, y, sigma=1):
    return np.exp(-np.linalg.norm(np.array(x) - np.array(y))**2 / (2 * sigma**2))


### 🌟 Implémentation du perceptron à noyau
def perceptron_kernel(X, Y, kernel, max_iter=1000):
    n_samples = len(Y)
    coefficients = np.zeros(n_samples)  # Coefficients alpha
    support_vectors = X.copy()

    for _ in range(max_iter):
        error = False
        for i in range(n_samples):
            sum_kernel = sum(coefficients[j] * Y[j] * kernel(X[j], X[i]) for j in range(n_samples))
            if Y[i] * sum_kernel <= 0:  # Mauvaise classification
                coefficients[i] += 1  # Mise à jour de alpha
                error = True
        if not error:
            break

    return coefficients, support_vectors


### 🌟 Fonction pour calculer la séparation des classes
def f_from_kernel(coeffs, support_set, kernel, x):
    return sum(coeffs[i] * Y[i] * kernel(support_set[i], x) for i in range(len(coeffs)))


### 🌟 Fonction pour calculer le score (accuracy)
def score(S, coeffs, support_set, kernel):
    correct_predictions = 0
    total_samples = len(S)

    for x, label in S:
        prediction = np.sign(sum(coeffs[i] * Y[i] * kernel(support_set[i], x) for i in range(len(coeffs))))
        if prediction == label:
            correct_predictions += 1

    return correct_predictions / total_samples


### 🚀 Expérimentation avec différentes valeurs de sigma
for sigma in [1, 2, 3, 5]:
    alphas_gauss, support_vectors_gauss = perceptron_kernel(X, Y, lambda x, y: kernel_gaussian(x, y, sigma))

    # Calcul du score (accuracy)
    accuracy = score(list(zip(X_test, Y_test)), alphas_gauss, support_vectors_gauss, lambda x, y: kernel_gaussian(x, y, sigma))
    
    # Affichage du score
    print(f"Sigma = {sigma}, Accuracy = {accuracy * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], color='blue', marker='o', label='Classe -1')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='red', marker='x', label='Classe 1')

    # Tracé de la frontière de séparation
    res = 100
    x_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), res)
    y_vals = np.linspace(min(X[:, 1]), max(X[:, 1]), res)

    for x in range(res):
        for y in range(res):
            if abs(f_from_kernel(alphas_gauss, support_vectors_gauss, lambda x, y: kernel_gaussian(x, y, sigma), [x_vals[x], y_vals[y]])) < 0.01:
                plt.plot(x_vals[x], y_vals[y], 'kx', markersize=2)  # Points proches de la frontière

    plt.legend()
    plt.title(f"Perceptron à Noyau Gaussien (σ={sigma})\nAccuracy: {accuracy * 100:.2f}%")
    plt.show()
