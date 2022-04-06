#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
print 'teste'

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()
t0 = time()
clf.fit(features_train,labels_train)
print 'training time',round(time()-t0,3) ,'s'
t0 = time()
a = clf.score(features_test,labels_test)
print 'accuracy :', a
print 'predicting time', round(time()-t0,3),'s'
#########################################################
#clf.fit(X, Y)
#GaussianNB(priors=None, var_smoothing=1e-09)
#>>> print(clf.predict([[-0.8, -1]]))
