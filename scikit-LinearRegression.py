#线性回归,预测
#数据导入

import numpy as np
from sklearn import datasets

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[: -20]
diabetes_X_test  = diabetes.data[-20 :]
diabetes_Y_train = diabetes.target[: -20]
diabetes_Y_test  = diabetes.target[-20 :]

#Linear Regression
from sklearn import linear_model

regr= linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_Y_train)
print regr.coef_

#the mean square error
meanSquareError=np.mean((regr.predict(diabetes_X_test)-diabetes_Y_test)**2)
print meanSquareError

#Explained variance score: 1 is perfect prediction
#and 0 means that there is no linear relationship
#between X and Y
variScore=regr.score(diabetes_X_test,diabetes_Y_test)
print variScore


#Sparse Method :only select the informative features and set
#non-informative ones.Lasso(least absolute shrinkage and selection)
#operator can set some coefficients to zero

regr = linear_model.Lasso()
alphas = np.logspace(-4,-1,6)
scores = [regr.set_params(alpha=alpha).fit(diabetes_X_train,
                                           diabetes_Y_train).
          score(diabetes_X_test,diabetes_Y_test) for alpha in alphas]

best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(diabetes_X_train,diabetes_Y_train)
print(regr.coef_)

#the mean square error
meanSquareError=np.mean((regr.predict(diabetes_X_test)-diabetes_Y_test)**2)
print meanSquareError

#Explained variance score: 1 is perfect prediction
#and 0 means that there is no linear relationship
#between X and Y
variScore=regr.score(diabetes_X_test,diabetes_Y_test)
print variScore
