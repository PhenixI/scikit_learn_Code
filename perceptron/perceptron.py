#Perceptron using scikit-learn ,materials and source codes (implementation of perceptron(python or matlab)) refer to : https://github.com/PhenixI/machine-learning/tree/master/6-Perceptron%20and%20Neural%20Networks
#load dataset
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X= iris.data[:,[2,3]]
y = iris.target

#split the dataset into separate training and test datasets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

#scale features for optimal performance
#standardize the features using the StandardScaler class from scikit-learn's perprocessing module
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Mose alogrithms in scikit-learn support muliclass classification by default via the One-Vs-Rest method
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter = 40,eta0= 0.1,random_state = 0)
ppn.fit(X_train_std,y_train)

y_pred = ppn.predict(X_test_std)
print('Misclssified samples: %d' % (y_test != y_pred).sum())

#metric performance

from sklearn.metrics import accuracy_score
print ("Accuracy: %.2f "% accuracy_score(y_test,y_pred))  

#draw decision surface
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
import DecisionBoundary

DecisionBoundary.plot_decision_regions(X=X_combined_std,y=y_combined,classifier = ppn,test_idx = range(105,150))

import matplotlib.pyplot as plt
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()