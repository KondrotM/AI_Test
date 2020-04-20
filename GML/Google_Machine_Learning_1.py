# -*- coding: utf-8 -*-
#importing tree algorithm
from sklearn import tree

#two features that can be tested: weight, texture
features = [[140,1],[130,1],[150,0],[170,0],[140,1],[130,1],[150,0],[170,0]]
#labels for each feature: apple, orange
labels = [0,0,1,1,0,0,1,1]
#assigns the tree algorithm to clf
clf = tree.DecisionTreeClassifier()
#trains the algorithm
clf = clf.fit(features,labels)
#predicts what this combination of values will end up with
print (clf.predict([[150,1]]))