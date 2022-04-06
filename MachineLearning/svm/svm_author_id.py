#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
#dados fatiados para deixar mais leve
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(C=10000., kernel = 'rbf')
t0 = time()
clf.fit(features_train,labels_train)
print "Training time:",round(time()-t0,3),"s"

t1 = time()
pred = clf.predict(features_test)
print "Testing time:", round(time()-t1,3),"s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
print("Accuracy is equal to %0.4F %%" % (accuracy*100))
#########################################################

#########################################################
# Who wrote emails at 10,26 & 50 in the predictions

answers = [pred[10],pred[26],pred[50]]
print answers

# How many emails were predicted to be in the Chris class (pred == 1)
c_predicted = sum(pred)
print "Number of emails predicted to be from Chris:", c_predicted
