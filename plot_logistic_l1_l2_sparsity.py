print(__doc__)

import numpy as np
import pylab as pl

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits()
X,y = digits.data,digits.target

X = StandardScaler().fit_transform(X)

#classify small against large digits
y = (y>4).astype(np.int)

#Set regularization parameter
for i,C in enumerate(10. ** np.arange(1,4)):
	clf_l1_LR = LogisticRegression(C=C,penalty='l1',tol=0.01)
	clf_l2_LR = LogisticRegression(C=C,penalty='l2',tol=0.01)
	clf_l1_LR.fit(X,y)
	clf_l2_LR.fit(X,y)

	coef_l1_LR = clf_l1_LR.coef_.ravel()
	coef_l2_LR = clf_l2_LR.coef_.ravel()

	#Coef_L1_LR contains zeros due to the L1 sparsity inducing norm
	sparsity_l1_LR = np.mean(coef_l1_LR==0) * 100
	sparsity_l2_LR = np.mean(coef_l2_LR==0) * 100

	print("C=%d" %C)
	print("Sparsity with L1 penalty : %.2f%%" % sparsity_l1_LR)
	print("score with L1 penalty: %.4f" % clf_l1_LR.score(X,y))
	print("Sparsity with L2 penalty : %.2f%%" % sparsity_l2_LR)
	print("score with L2 penalty:%.4f"  % clf_l2_LR.score(X,y))

	l1_plot = pl.subplot(3,2,2*i +1)
	l2_plot = pl.subplot(3,2,2*(i+1))

	if i==0:
		l1_plot.set_title("L1 penalty")
		l2_plot.set_title("L2 penalty")
	l1_plot.imshow(np.abs(coef_l1_LR.reshape(8,8)),interpolation='nearest',cmap='binary',vmax=1,vmin=0)
	l2_plot.imshow(np.abs(coef_l2_LR.reshape(8,8)),interpolation='nearest',cmap='binary',vmax=1,vmin=0)

	pl.text(-8,3,"C=%d" %C)
	l1_plot.set_xticks(())
	l1_plot.set_yticks(())
	l2_plot.set_xticks(())
	l2_plot.set_yticks(())

pl.show()
