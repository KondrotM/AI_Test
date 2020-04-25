import numpy as np
import pandas as pd
import time
import pdb

months = ['05','06','07']


for month in months:
	if month == '06':
		for day in range(1,32):
			# pdb.set_trace()
			if day < 10:
				pathDay = '0%s' % (day)
			else:
				pathDay = day
			df = pd.read_csv('http://environment.data.gov.uk/flood-monitoring/archive/readings-2020-04-18.csv')
			# pdb.set_trace()
			currpercent = 0
			for i in range(len(df)):
				# pdb.set_trace()
				if currpercent//100 < i//len(df):
					currpercent += 1
					print('%s percent complete' % (currpercent))
				doDrop = False
				# pdb.set_trace()
				if 'tipping_bucket' in df['measure'][i]:
					if df['dateTime'][i][-2] == '0' and df['dateTime'][i][-5] == '0':
						pass
					else:
						doDrop = True
				else:
					doDrop = True
				if doDrop:
					# pdb.set_trace()
					df.drop([i],inplace=True)
					# pdb.set_trace()
			datas = df.groupby('dateTime')['value'].mean()
			df = datas.reset_index()
			print(df)
			# time.sleep(1)
			break


