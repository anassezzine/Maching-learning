import matplotlib.pyplot as plt
import numpy as np

with open("./TP3_data/learn.data", "r") as f:
    training_set = [eval(line) for line in f]


# Supposons que chaque entrée de training_set soit de la forme ([x1, x2, ...], label)
X = np.array([data[0] for data in training_set])
y = np.array([data[1] for data in training_set])

# Pour une visualisation en 2D, nous supposons que les données ont deux caractéristiques
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Visualisation des données d\'entraînement')
plt.show()


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, -1)


perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X, y)


# Chargement des données de test
with open("./TP3_data/test.data", "r") as f:
    test_set = [eval(line) for line in f]

X_test = np.array([data[0] for data in test_set])
y_test = np.array([data[1] for data in test_set])

# Prédictions
predictions = perceptron.predict(X_test)

# Calcul du score
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


def score(S, coeffs, support_set, k):
    correct_predictions = 0
    total_samples = len(S)

    for x, label in S:
        prediction = np.sign(sum(coeff * k(support, x) for coeff, support in zip(coeffs, support_set)))
        if prediction == label:
            correct_predictions += 1

    return correct_predictions / total_samples



