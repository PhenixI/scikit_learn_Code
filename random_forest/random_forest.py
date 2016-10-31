#for more info please visit:https://github.com/PhenixI/machine-learning/tree/master/17-RandomForest

#load dataset
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X= iris.data[:,[2,3]]
y = iris.target

#split the dataset into separate training and test datasets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

#random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',n_estimators = 10,random_state = 1,n_jobs = 2)
forest.fit(X_train,y_train)

X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))

import DecisionBoundary
DecisionBoundary.plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx = range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()