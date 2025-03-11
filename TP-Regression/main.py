import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def regression_moindres_carres(X, y):
    """Implémente la régression linéaire par moindres carrés."""
    X_biais = np.c_[np.ones((X.shape[0], 1)), X]  # Ajout d'une colonne de biais
    w = np.linalg.inv(X_biais.T.dot(X_biais)).dot(X_biais.T).dot(y)
    return w

def prediction(X, w):
    """Prédit y en fonction de X et des poids w."""
    X_biais = np.c_[np.ones((X.shape[0], 1)), X]
    return X_biais.dot(w)

# Chargement des données
data = np.loadtxt("dataRegLin2D.txt")
X = data[:, :-1]
y = data[:, -1]

# Appliquer la régression linéaire
w = regression_moindres_carres(X, y)
y_pred = prediction(X, w)

# Appliquer la régression linéaire avec Scikit-learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_sklearn = lin_reg.predict(X)

# Appliquer la régression Ridge et Lasso
ridge = Ridge(alpha=0.001)
ridge.fit(X, y)
y_pred_ridge = ridge.predict(X)

lasso = Lasso(alpha=0.00207)
lasso.fit(X, y)
y_pred_lasso = lasso.predict(X)

# Visualisation des coefficients
print("Coefficients Moindres Carrés:", w)
print("Coefficients Scikit-learn:", lin_reg.coef_)
print("Coefficients Ridge:", ridge.coef_)
print("Coefficients Lasso:", lasso.coef_)

# Visualisation en 3D
def plot_data_3D(X, y, y_pred, title="Régression linéaire"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o', label="Données réelles")
    ax.scatter(X[:, 0], X[:, 1], y_pred, c='r', marker='x', label="Prédictions")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    ax.legend()
    plt.title(title)
    plt.show()

# Affichage des résultats
plot_data_3D(X, y, y_pred, "Régression moindres carrés")
plot_data_3D(X, y, y_pred_sklearn, "Régression avec Scikit-learn")
plot_data_3D(X, y, y_pred_ridge, "Régression Ridge")
plot_data_3D(X, y, y_pred_lasso, "Régression Lasso")

# Calcul et affichage des erreurs
erreur_moindres_carres = mean_squared_error(y, y_pred)
erreur_sklearn = mean_squared_error(y, y_pred_sklearn)
erreur_ridge = mean_squared_error(y, y_pred_ridge)
erreur_lasso = mean_squared_error(y, y_pred_lasso)

print(f"Erreur quadratique moyenne (moindres carrés) : {erreur_moindres_carres}")
print(f"Erreur quadratique moyenne (Scikit-learn) : {erreur_sklearn}")
print(f"Erreur quadratique moyenne (Ridge) : {erreur_ridge}")
print(f"Erreur quadratique moyenne (Lasso) : {erreur_lasso}")

# Visualisation de l'erreur en fonction d'alpha
alphas = np.logspace(-3, -1, 20)
ridge_errors = []
lasso_errors = []
ridge_coefs = []
lasso_coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    ridge_errors.append(mean_squared_error(y, ridge.predict(X)))
    ridge_coefs.append(ridge.coef_)
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    lasso_errors.append(mean_squared_error(y, lasso.predict(X)))
    lasso_coefs.append(lasso.coef_)

plt.figure()
plt.plot(alphas, ridge_errors, label="Ridge")
plt.plot(alphas, lasso_errors, label="Lasso")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Erreur quadratique moyenne")
plt.legend()
plt.title("Impact de la régularisation sur l'erreur")
plt.show()

# Visualisation de l'évolution des coefficients
ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)
plt.figure()
for i in range(ridge_coefs.shape[1]):
    plt.plot(alphas, ridge_coefs[:, i], label=f"Ridge coef {i}")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Valeur des coefficients")
plt.title("Évolution des coefficients Ridge")
plt.legend()
plt.show()

plt.figure()
for i in range(lasso_coefs.shape[1]):
    plt.plot(alphas, lasso_coefs[:, i], label=f"Lasso coef {i}")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Valeur des coefficients")
plt.title("Évolution des coefficients Lasso")
plt.legend()
plt.show()
