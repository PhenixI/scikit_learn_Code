#for more info and material (source codes )about lr please visit:   https:https://github.com/PhenixI/machine-learning/tree/master/15-SVM

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

#train a SVM model to classify the different flowers in our Iris dataset:
from sklearn.svm import SVC
svm  = SVC(kernel='linear',C=1.0,random_state=0)
svm.fit(X_train_std,y_train)

#draw decision surface
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
import DecisionBoundary
DecisionBoundary.plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
