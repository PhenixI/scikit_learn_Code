
import numpy as np
from sklearn import datasets
iris=datasets.load_iris()
iris_X=iris.data
iris_Y=iris.target
np.unique(iris_Y)

#Split iris data in train and test data
#A random permutation ,to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[: -10]]
iris_Y_train = iris_Y[indices[: -10]]
iris_X_test  = iris_X[indices[-10 :]]
iris_Y_test  = iris_Y[indices[-10 :]]

#Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train,iris_Y_train)
print knn.predict(iris_X_test)
print iris_Y_test




