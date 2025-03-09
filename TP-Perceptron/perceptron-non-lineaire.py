import numpy as np
import matplotlib.pyplot as plt
from pylab import rand

def generateData3(n):
    xb = (rand(n) * 2 - 1) / 2
    yb = (rand(n) * 2 - 1) / 2  # Points dans un carré centré à l'origine

    xr = 3 * (rand(4 * n) * 2 - 1) / 2
    yr = 3 * (rand(4 * n) * 2 - 1) / 2  # Points plus éloignés

    inputs = []
    for i in range(n):
        inputs.append(((xb[i], yb[i]), -1))

    for i in range(4 * n):
        if abs(xr[i]) >= 1 or abs(yr[i]) >= 1:  # Points extérieurs à un carré de côté 2
            inputs.append(((xr[i], yr[i]), 1))

    return np.array([p[0] for p in inputs]), np.array([p[1] for p in inputs])

# Génération et affichage des données
X3, Y3 = generateData3(100)

plt.figure(figsize=(8, 6))
plt.scatter(X3[Y3 == -1, 0], X3[Y3 == -1, 1], color='blue', marker='o', label='Classe -1')
plt.scatter(X3[Y3 == 1, 0], X3[Y3 == 1, 1], color='red', marker='x', label='Classe 1')
plt.legend()
plt.title("Données non linéairement séparables")
plt.show()

def polynomial_feature_mapping(X):
    X_poly = np.c_[np.ones(X.shape[0]), X[:, 0], X[:, 1], X[:, 0]**2, X[:, 0] * X[:, 1], X[:, 1]**2]
    return X_poly

# Transformation des données
X3_poly = polynomial_feature_mapping(X3)


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

# Définition du noyau polynomial
def kernel_poly(x, y):
    return (1 + np.dot(x, y))**2  # Noyau quadratique

# Entraînement du perceptron à noyau polynomial
alphas, support_vectors = perceptron_kernel(X3_poly, Y3, kernel_poly)

def f_from_kernel(coeffs, support_set, kernel, x):
    return sum(coeffs[i] * Y3[i] * kernel(support_set[i], x) for i in range(len(coeffs)))

# Tracé de la frontière de séparation
res = 100
x_vals = np.linspace(-1.5, 1.5, res)
y_vals = np.linspace(-1.5, 1.5, res)

plt.figure(figsize=(8, 6))
plt.scatter(X3[Y3 == -1, 0], X3[Y3 == -1, 1], color='blue', marker='o', label='Classe -1')
plt.scatter(X3[Y3 == 1, 0], X3[Y3 == 1, 1], color='red', marker='x', label='Classe 1')

for x in range(res):
    for y in range(res):
        point_transformed = polynomial_feature_mapping(np.array([[x_vals[x], y_vals[y]]]))[0]
        if abs(f_from_kernel(alphas, support_vectors, kernel_poly, point_transformed)) < 0.01:
            plt.plot(x_vals[x], y_vals[y], 'kx', markersize=2)  # Points proches de la frontière

plt.legend()
plt.title("Perceptron à Noyau (Quadratique)")
plt.show()



def kernel_gaussian(x, y, sigma=1):
    return np.exp(-np.linalg.norm(np.array(x) - np.array(y))**2 / (2 * sigma**2))

# Entraînement du perceptron avec noyau Gaussien
alphas_gauss, support_vectors_gauss = perceptron_kernel(X3, Y3, kernel_gaussian)

# Expérimentation avec différents sigma
for sigma in [0.2, 0.5, 1, 2, 5]:
    alphas_gauss, support_vectors_gauss = perceptron_kernel(X3, Y3, lambda x, y: kernel_gaussian(x, y, sigma))

    plt.figure(figsize=(8, 6))
    plt.scatter(X3[Y3 == -1, 0], X3[Y3 == -1, 1], color='blue', marker='o', label='Classe -1')
    plt.scatter(X3[Y3 == 1, 0], X3[Y3 == 1, 1], color='red', marker='x', label='Classe 1')

    # Tracé de la frontière de séparation
    res = 100
    x_vals = np.linspace(-1.5, 1.5, res)
    y_vals = np.linspace(-1.5, 1.5, res)

    for x in range(res):
        for y in range(res):
            if abs(f_from_kernel(alphas_gauss, support_vectors_gauss, lambda x, y: kernel_gaussian(x, y, sigma), [x_vals[x], y_vals[y]])) < 0.01:
                plt.plot(x_vals[x], y_vals[y], 'kx', markersize=2)  # Points proches de la frontière

    plt.legend()
    plt.title(f"Perceptron à Noyau Gaussien (σ={sigma})")
    plt.show()

