import pylab as pl
from sklearn.datasets import load_iris

irisData=load_iris()
X=irisData.data
Y=irisData.target
colors=['red','green','blue']
for x in range(4):
    for y in range(4):
        if x!=y:
            pl.figure()
            for i in range(3):
                pl.scatter(X[Y==i][:, x],X[Y==i][:,y],color=colors[i],\
                    label=irisData.target_names[i])
            pl.legend()
            pl.xlabel(irisData.feature_names[x])
            pl.ylabel(irisData.feature_names[y])
            pl.title(u"Données Iris - dimensions %d et %d"%(x,y))
pl.show()