#nonlinear svm using kernel

#generate data
#create dataset that has the form of an XOR gate using the logical_xor function from NumPy, 
#where 100 samples will be assigned the class lable 1 and 100 samples will be assigned the 
#class lable -1,respectively:
import numpy as np
np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)

#import matplotlib.pyplot as plt
#plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],c='b', marker='x', label='1')
#plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],c='r', marker='s', label='-1')
#plt.ylim(-3.0)
#plt.legend()
#plt.show()

#replace the parameter of SVC kernel = 'linear' with kernel = 'rbf'
from sklearn.svm import SVC
svm = SVC(kernel='rbf',random_state = 0,gamma=0.10,C=10.0)
svm.fit(X_xor,y_xor)

#draw decision boundary
import DecisionBoundary
DecisionBoundary.plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.show()

#----------------------------------------------------------------
#sample 2,using Iris data
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
#the gamma parameter , can be understood as a cut-off parameter for the Gaussian sphere.
#if we increase the value of gamma, we increase the influence or reach of the training samples,
#which leads to a softer decision boundary.
#although the model fits the training dataset very well, such a classifier will likely have a 
#high generalization error on unseen data, which illustrates that the optimization of gamma also plays an important role in controlling overfitting.
svm  = SVC(kernel='rbf',C=1.0,random_state=0,gamma=0.1)
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



