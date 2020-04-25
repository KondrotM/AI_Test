import numpy as np
import pandas as pd
import time
import urllib.request
import codecs
import pdb
import csv

months = ['05','06','07']
data = {
	'dateTime' : [4,2,55,1],
	'value' : [2,4,2,1]
}
dfs = pd.DataFrame(columns = ['dateTime','value'])
# dfs2 = pd.DataFrame(data, columns = ['dateTime','value'])
# dfs = pd.concat([dfs, dfs2], ignore_index = True)
# dfs = pd.concat([dfs, dfs2], ignore_index = True)
# dfs = pd.concat([dfs, dfs2], ignore_index = True)
# print(dfs)

def clean(url):
	rd = [['dateTime','measure','value']]
	data = {
		'dateTime' : [],
		'measure' : [],
		'value' : []
	}
	response = urllib.request.urlopen(url)
	csvfile = csv.reader(codecs.iterdecode(response, 'utf-8'))
	# response = requests.get(url, stream=True)
	# print(response[])
	# text = response.iter_lines()
	# print(text)
	# reader = csv.reader(text, delimiter=',')

	for row in csvfile:
		if 'tipping_bucket' in row[1]:
			if row[0][-2] == '0' and row[0][-5] == '0' and row[0][-3] == '0':
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
					avg = np.sum(nums) // len(nums)
					# print(nums)
					row[2] = avg

				data['dateTime'].append(row[0])
				data['measure'].append(row[1])
				data['value'].append(float(row[2]))

	return data

def main ():
	for month in months:
		if month != '06':
			for day in range(1,32):
				# pdb.set_trace()
				if day < 10:
					pathDay = '0%s' % (day)
				else:
					pathDay = day
				data = clean('http://environment.data.gov.uk/flood-monitoring/archive/readings-2019-%s-%s.csv' % (month, pathDay))
				df = pd.DataFrame(data, columns = ['dateTime','measure','value'])
				# pdb.set_trace()
				df = df.groupby('dateTime')['value'].mean()
				df = df.reset_index()
				# print(df)
				global dfs
				dfs = pd.concat([dfs,df], ignore_index = True)
				print('Downloaded %s %s' % (month, pathDay))
				# pdb.set_trace()
				time.sleep(1)
		if month == '06':
			for day in range(1,31):
				# pdb.set_trace()
				if day < 10:
					pathDay = '0%s' % (day)
				else:
					pathDay = day
				data = clean('http://environment.data.gov.uk/flood-monitoring/archive/readings-2019-%s-%s.csv' % (month, pathDay))
				df = pd.DataFrame(data, columns = ['dateTime','measure','value'])
				# pdb.set_trace()
				df = df.groupby('dateTime')['value'].mean()
				df = df.reset_index()
				dfs = pd.concat([dfs,df], ignore_index = True)
				print('Downloaded %s %s' % (month, pathDay))
				time.sleep(1)
				# currpercent = 0
				# for i in range(len(df)):
				# 	# pdb.set_trace()
				# 	if currpercent//100 < i//len(df):
				# 		currpercent += 1
				# 		print('%s percent complete' % (currpercent))
				# 	doDrop = False
				# 	# pdb.set_trace()
				# 	if 'tipping_bucket' in df['measure'][i]:
				# 		if df['dateTime'][i][-2] == '0' and df['dateTime'][i][-5] == '0':
				# 			pass
				# 		else:
				# 			doDrop = True
				# 	else:
				# 		doDrop = True
				# 	if doDrop:
				# 		# pdb.set_trace()
				# 		df.drop([i],inplace=True)
				# 		# pdb.set_trace()
				# datas = df.groupby('dateTime')['value'].mean()
				# df = datas.reset_index()



main()
dfs.to_csv('set2_rainfall_data.csv')