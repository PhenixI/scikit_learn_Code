print (__doc__)

import numpy as np
import pylab as pl

from sklearn import linear_model
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

print("Computing regularization path using the LARs ...")
alphas,_,coefs = linear_model.lars_path(X,y,method='lasso',verbose = True)

xx = np.sum(np.abs(coefs.T),axis=1)
xx /= xx[-1]

pl.plot(xx,coefs.T)
ymin,ymax = pl.ylim()
pl.vlines(xx,ymin,ymax,linestyle='dashed')
pl.xlabel('|coef| / max|coef|')
pl.ylabel('Coefficients')
pl.title('Lasso Path')
pl.axis('tight')
pl.show()