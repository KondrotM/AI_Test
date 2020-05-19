import numpy as np
import pandas as pd
from sklearn import tree
import json
import matplotlib.pyplot as plt
import pdb

def findSampleDifferences(labels, samples):
	print(samples)
	# for i, j in zip(range(len(samples)), range(len(labels))):
	for i, j in zip(range(len(samples)),labels):
		sampleDt = (samples.iloc[i]['dateTime'])
		sampleDt = sampleDt[0:4]+sampleDt[5:7]+sampleDt[8:10]+'_'+sampleDt[11:13]+sampleDt[14:16]
		labelDt = (j)
		if sampleDt == labelDt:
			print('Data match')
			pass
		else:
			print ('Data mismatch found')
			print('data.json dateTime:	%s' % (labelDt))
			print('df dateTime:	%s' % (sampleDt))
			input('Press Enter to Continue') 
		# print(sampleDt, labelDt)


def main():
	print('Getting Train Target...')
	train_target = []
	test_target = []
	df = pd.read_csv('set2_rainfall_data.csv')
	df = df[['dateTime','value']]
	df['dateTime'] = pd.to_datetime(df['dateTime'])
	df.set_index('dateTime',inplace=True)

	dfTrain = df[:len(df)//2]
	dfTest = df[len(df)//2:]
	print(dfTest)

	for i in dfTrain['value']:
		train_target.append(round(i, 5))


	for i in dfTest['value']:
		test_target.append(round(i, 5))


	print('Getting Train Data...')
	with open('data.json') as json_file:
		data = json.load(json_file)

	# findSampleDifferences(data, df)
	train_data = []
	for i in data:
		val = []
		for j in data[i]:
			val.append(data[i][j])
		train_data.append(val)


	print('Getting Test Target...')
	# dfTest = pd.DataFrame(columns = [['dateTime','value']])
	# dfTest['dateTime'] = pd.to_datetime(dfTest['dateTime'])
	# dfTest.set_index('dateTime',inplace=True)
	# test_target = []
	# for i in range(len(train_target)//2):
	# 	if i % 10 == 0:
	# 		dfTest = pd.append()
	# 		test_target.append(train_target.pop(i))

	print('Getting Test Data...')
	test_data = train_data[:len(train_data)//2]
	train_data = train_data[len(train_data)//2:]

	# for i in range(len(train_data)//2):
	# 	# if i % 10 == 0:
	# 	test_data.append(train_data.pop(i))


	pdb.set_trace()
	print('Training Tree...')
	clf = tree.DecisionTreeRegressor()
	clf = clf.fit(train_data, train_target)
	# print(test_target)
	# print(test_data)
	predicted = clf.predict(test_data)
	testData = {
	'test target' : test_target,
	'predicted' : predicted
	}
	dfTest['predicted'] = predicted
	dfTest['difference'] = dfTest['value'] - dfTest['predicted']
	# finalDf = pd.DataFrame(testData,columns = ['test target','predicted'])
	# finalDf['difference'] = finalDf['test target'] - finalDf['predicted']
	# print(finalDf)
	print('Cumulative difference:')
	print(dfTest['difference'].abs().sum()/len(dfTest))


	plt.figure()

	print(df.index[0])
	# df['dateTime'] = df['dateTime'].dt.strftime('%Y %m %d - %H:%M')
	dfTest[['value','predicted']].plot()
	plt.show()
	# plt.scatter(df.dateTime,df['value'])
	# plt.scatter(len(train_data), len(train_target))
	# plt.plot(test_data, predicted, label = 'prediction')
	# plt.xlabel('data')
	# plt.ylabel('target')
	# plt.title('Decision Tree Regression')
	# plt.legend()
	# plt.show()

main()