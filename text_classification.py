from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time

import pylab as pl
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest,chi2
#线性回归，Ridge分类
from sklearn.linear_model import RidgeClassifier
#支持向量机
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
#感知机算法
from sklearn.linear_model import Perceptron
from sklearn.linear_mddel import PassiveAggressiveClassifier
#朴素贝叶斯
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

#Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

#parse commandLine arguments
op=OptionParser()
op.add_option("--report",action="store_true",dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",action="store",type="int",dest="select_chi2",
              help="Select some number of features using a chi-squared test")

op.add_option("--confusion_matrix",action="store_true",dest="print_cm",
              help="Print the confusion matrix.")

op.add_option("--top10",action="store_true",dest="print_top10",
              help="Print ten most discriminative terms per class for every classifier.")

op.add_option("--all_categories",action="store_true",dest="all_categories",
              help="Whether to use all categories or not.")

op.add_option("--use_hashing",action="store_ture",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",action="store",type=int,default= 2**16,
              help="n_features when using the hashing vectorizer.")

op.add_option("--filtered",action="store_true",
              help="Remove newsgroup information that is easily overfit:"
                    "headers,signatures, and quoting.")

(opts,args)=op.parse_args()


daga;





