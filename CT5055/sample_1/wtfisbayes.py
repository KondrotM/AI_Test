import numpy as np

arr1 = np.array([[0,0,0,0,0]])
arr2 = np.array([[3000,2000,4000,100,200],[3000,2000,4000,100,200]])

# no axis provided, array elements will be flattened
arr_flat = np.append(arr1, arr2, axis = 0)
arr_flat = np.delete(arr_flat, 0, 0)
print(arr_flat)  # [ 1  2  3  4 10 20 30 40]

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# val = [5 3 1 6 3]
# print(val[])
print(type(X_train[0]))
print(X_train[0])
# print(X_test)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))