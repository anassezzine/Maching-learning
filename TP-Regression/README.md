**Rapport du TP : Régression Linéaire et Régularisation**

## **1. Introduction**

L'objectif de ce TP est d'implémenter et d'analyser différentes techniques de régression linéaire,
notamment la régression par moindres carrés, Ridge et Lasso, en utilisant `numpy` et `scikit-learn`.
Nous allons comparer leurs performances en termes d'erreur quadratique moyenne et d'impact sur les coefficients.

## **2. Régression Linéaire par Moindres Carrés**

L'algorithme implémenté suit la formule analytique :
\[
w = (X^TX)^{-1} X^T y
\]
Nous avons appliqué cette régression sur un jeu de données `dataRegLin2D.txt`, et obtenons les résultats suivants :

- **Coefficients obtenus :**
  - Moindres Carrés : `[-0.678, 0.245, 1.370]`
  - Scikit-learn : `[0.245, 1.370]`

- **Erreur quadratique moyenne (MSE) :**
  - Moindres Carrés : `0.0102`
  - Scikit-learn : `0.0102`

Les résultats sont très similaires entre l'implémentation manuelle et celle de `scikit-learn`, confirmant ainsi la validité de notre code.

## **3. Régression Ridge et Lasso**

La régression Ridge et Lasso ajoutent une pénalisation sur les coefficients pour réduire le sur-apprentissage :
- **Ridge** : Ajoute une pénalisation L2 (`\lambda \|w\|^2`)
- **Lasso** : Ajoute une pénalisation L1 (`\lambda \|w\|`), ce qui peut forcer certains coefficients à 0.

Nous avons testé ces méthodes en optimisant le paramètre `alpha` avec `GridSearchCV`.
Les meilleurs `alpha` trouvés sont :
- Ridge : `0.001`
- Lasso : `0.00207`

**Comparaison des erreurs :**
- Moindres carrés : `0.0102`
- Ridge : `0.0102`
- Lasso : `0.0103`

**Impact sur les coefficients :**
- Ridge : les coefficients sont très proches de la régression classique.
- Lasso : certains coefficients sont réduits plus fortement, ce qui favorise la sélection de variables.

## **4. Analyse Graphique**

Nous avons visualisé les résultats avec :
1. **Graphiques 3D** : Comparaison des valeurs prédites et des valeurs réelles.
2. **Impact de la régularisation sur l'erreur** : L'erreur varie en fonction de `alpha`.
3. **Évolution des coefficients** : Lasso réduit plus fortement les coefficients que Ridge.

## **5. Conclusion**

- La régression par moindres carrés fonctionne bien pour ce jeu de données.
- Ridge et Lasso offrent des avantages en termes de généralisation et de sélection de variables.
- L'optimisation du paramètre `alpha` est cruciale pour de meilleurs résultats.

Ce TP a permis de mieux comprendre les différentes techniques de régression linéaire et leur impact sur les données. 🚀

