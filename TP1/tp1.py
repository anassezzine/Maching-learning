from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt

irisData=load_iris()
X=irisData.data
Y=irisData.target
x = 0
y = 1
plt.scatter(X[:, x], X[:, y],c=Y)
plt.show()