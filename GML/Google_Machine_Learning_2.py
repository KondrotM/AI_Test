# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:22:05 2017

@author: ashbeck
"""
import numpy as np
from sklearn import tree
#iris is a set of sample data for different types of flowers
from sklearn.datasets import load_iris
iris = load_iris()
#values of three different flowers from the data sets
test_idx = [0,50,100]

#training data
#deletes the testing data from the training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

#two prints compare the actual values from the data predicted values
print (test_target)

print (clf.predict(test_data))


#print (iris.feature_names)
#print (iris.target_names)
#for i in range(len(iris.target)):
#    print ("Example %d: label %s. features %s" % (i, iris.target[i], iris.data[i]))