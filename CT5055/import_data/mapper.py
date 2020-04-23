import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def init():
	df = pd.read_csv('readings-2019-05-01.amended.csv')
	df = df[['dateTime','value']]
	df.astype({'value': 'float64'})
	# print(df.dtypes)
	datas = df.groupby('dateTime')['value'].mean()
	# datas.plot()
	flat = datas.reset_index()
	# print(flat.head())
	# print(flat)
	# print(flat.iloc[20:40])
	colsToDrop = []
	for i in range(len(flat)):
		if (flat.iloc[i]['dateTime'][-2]) != '0' or flat.iloc[i]['dateTime'][-5] != '0':
			colsToDrop.append(i)
			# flat = flat.drop([0,i])
			# print(flat.head())

	for i in colsToDrop:
		# try:
		flat = flat.drop([i])
		# flat = flat.reset_index()
		# except:
			# pass
	flat = flat.reset_index()[['dateTime','value']]
	flat.to_csv('out.csv')
	# print(flat)
	# print(type(flat))
	# flat.plot()
	# plt.show()
	# flat.plot()
	# plt.show()
	# print(datas.head())
	# datas = datas.to_frame()

	# for i in datas:
	# 	print (i)
	# # print(df.head())

def clean():
	rd = []
	with open('readings-2019-05-01.csv') as csvFile:
		reader = csv.reader(csvFile, delimiter=',')
		# rd = reader
		i = 0
		cats = []
		for row in reader:
			# rd.append(row)
			if '|' in row[2]:
				nums = []
				avg = 0
				while '|' in row[2]:
					ind = (row[2].index('|'))
					num = row[2][:ind]
					nums.append(float(num))
					row[2] = row[2][ind+1:]
				# print(row[2][ind+1:])
				# row[2] = row[2][:ind]
				# print (row[2])
				avg = np.sum(nums)
				# print(nums)
				row[2] = avg
				# print(avg)

			if 'tipping_bucket' in row[1]:
				rd.append(row)
			# val = row[1][-29:]
			# if val not in cats:
			# 	cats.append(val)

		# for i in cats:
		# 	print(i)

	with open('readings-2019-05-01.amended.csv','w',newline='') as f:
		writer = csv.writer(f)
		# for row in rd:
		writer.writerows(rd)


		# i = 0
		# for row in reader:
		# 	if i < 10:
		# 		if '|' in row[2]:
		# 			# ind = (row[2].index('|'))
		# 			# row[2] = row[2][:ind]
		# 			print (row[2])
		# 			i+=1
		# 	else:
		# 		break

heads = ['dateTime','values']
values = []

init()
# clean()
