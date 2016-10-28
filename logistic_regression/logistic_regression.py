#for more info and material (source codes )about lr please visit:   https://github.com/PhenixI/machine-learning/tree/master/2-Logistic%20Regression

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

from sklearn.linear_model import LogisticRegression
# C is directly related to the regularization parameter lamda. which is its inverse: C = 1/lamda.
lr = LogisticRegression(C=1000.0,random_state=0)
lr.fit(X_train_std,y_train)

#draw decision surface
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
import DecisionBoundary
DecisionBoundary.plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx = range(105,150))

import matplotlib.pyplot as plt
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# experiment for lamda (regularization parameter)
weights,params = [],[]
for c in np.arange(-5,5):
    lr = LogisticRegression(C=10**c,random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
plt.plot(params,weights[:,0],label='petal length')
plt.plot(params,weights[:,1],linestyle='--',label = 'petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

