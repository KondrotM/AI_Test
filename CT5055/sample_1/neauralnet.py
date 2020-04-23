import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris

import json

iris = load_iris()

# print(type(iris))
train_target = []
df = pd.read_csv('out.csv')
df = df['value']
for i in df:
	train_target.append(round(i,5))
# print(train_target)

with open('data.json') as json_file:
	data = json.load(json_file)

train_data = []
nums = 0
for i in data:
	if nums < 48:
		val = []
		j = data[i]
		for j in data[i]:
			val.append(data[i][j])
			# print (data[i][j])
		train_data.append(val)
		nums+=1
	else:
		break
	# train_data.append(data[i])

# print(train_data)

test_data = []
test_data.append(train_data.pop(10))
test_data.append(train_data.pop(20))
test_data.append(train_data.pop(30))

test_target = []
test_target.append(train_target.pop(10))
test_target.append(train_target.pop(20))
test_target.append(train_target.pop(30))

clf = tree.DecisionTreeRegressor()
clf = clf.fit(train_data, train_target)
print(test_data)
print(test_target)
print(clf.predict(test_data))