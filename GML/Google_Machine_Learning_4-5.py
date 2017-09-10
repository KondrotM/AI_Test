# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:23:25 2017

@author: officialgupta
"""
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

#euclidian distance is shortest distance between two points
def euc(a,b):
    return distance.euclidean(a,b)


#import random
#creating a learning algorithm from scratch
class ScrappyKNN():
    def fit(self, x_train, y_train):
        #x_train has features, y_train has labels
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predictions = []
        for row in x_test:
#            label = random.choice(self.y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions
    #finds closest neighbour
    def closest(self, row):
        #saves shortest distance so far
        best_dist = euc(row, self.x_train[0])
        #saves which value it is closest to
        best_index = 0
        #iterates over all the all the features
        for i in range(1, len(self.x_train)):
            dist = euc(row,self.x_train[i])
            #tests for shortest distance 
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
        

iris = datasets.load_iris()

x = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .5)

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()



my_classifier.fit(x_train,y_train)

predictions = my_classifier.predict(x_test)

print (accuracy_score(y_test, predictions))