import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import json

def main():

	print('Getting Train Target...')
	train_target = np.array([])
	df = pd.read_csv('set2_rainfall_data.csv')
	df = df[['dateTime','value']]

	for i in df['value']:
		train_target = np.append(train_target, [float(round(i, 5))])

	print('Getting Train Data...')
	with open('data.json') as json_file:
		data = json.load(json_file)

	train_data = np.array([[0,0,0,0,0,0,0,0,0]])
	for i in data:
		val = []
		val = np.array([])
		for j in data[i]:
			val = np.append(val, [[int(data[i][j])]])

		# val = np.delete(val, 0 , 0)	# val.append(data[i][j])
		val = np.array([val])
		train_data = np.append(train_data, val, axis = 0)
		# train_data.append(val)

	train_data = np.delete(train_data, 0, 0)

	X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size = 0.5, random_state = 0)

	clf = tree.DecisionTreeRegressor()
	clf = clf.fit(train_data, train_target)

	gnb = LogisticRegression()
	gnb = gnb.fit(X_train, y_train)

	testData = {
	'test target' : test_target,
	'predicted' : predicted
	}

	finalDf = pd.DataFrame(testData,columns = ['test target','predicted'])
	finalDf['difference'] = finalDf['test target'] - finalDf['predicted']
	print(finalDf)
	print('Cumulative difference:')
	print(finalDf['difference'].abs().sum()/len(finalDf))

main()