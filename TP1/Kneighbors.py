from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# Chargement des données iris
irisData = load_iris()
X = irisData.data  
Y = irisData.target  

# Séparation en train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random.seed())

# Création du classifieur KNN
nb_voisins = 15
clf = neighbors.KNeighborsClassifier(nb_voisins)

# Entraînement du modèle
clf.fit(X_train, Y_train)

# affiche les points qui sont differents entre le predict et le test


# Score
score_train = clf.score(X_train, Y_train)
print("Score training: ", score_train)

# Prédiction
Y_pred = clf.predict(X_test)
print(len(X_test[Y_test!=Y_pred]))

# Calcul de la précision
score_test = clf.score(X_test, Y_test)
print("Score test: ", score_test)

# Visualisation des prédictions
print(Y_pred)
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred)
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, marker="x")
plt.xlabel(irisData.feature_names[0])
plt.ylabel(irisData.feature_names[1])
plt.colorbar(label="Classe prédite")  # Ajoute une légende pour les couleurs
plt.title("Visualisation des prédictions du modèle KNN")
plt.show()

