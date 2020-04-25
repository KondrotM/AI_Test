from PIL import Image
import colorsys
import json
import numpy as np
# import pandas as pd
import pdb

months  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

hours = ['0000','0030','0100','0130','0200','0230','0300','0330','0400','0430','0500','0530','0600','0630','0700','0730','0800','0930','1000','1030','1100','1130','1200','1230','1300','1330','1400','1430','1500','1530','1600','1630','1700','1730','1800','1830','1900','1930','2000','2030','2100','2130','2200','2230','2300','2330']
hours = ['0830','0900']
colours = [(50,50,255,255), # blue
(100,150,255,255), # light_blue
(150,150,50,255), # olive
(247,247,150,255), # yellow
(255,200,50,255), # orange
(255,50,50,255), # red
(255,50,255,255), # purple
(255,255,255,255) # white
]

dataCSV = {
}


def getImage(path, filename):
	img = Image.open(path+filename)
	img = img.convert('RGBA')
	# datas = img.getdata()
	pixdata = img.load()

	width, height = img.size

	dataCSV[filename[:-4]] = {}
	dataCSV[filename[:-4]]['pix'] = 0

	for i in colours:
		dataCSV[filename[:-4]][str(i)] = 0
	for y in range(height):
		for x in range(width):

			check1 = False
			check2 = False
			check3 = False
			if pixdata[x, y][0] - 50 <= pixdata[x, y][1] <= pixdata[x, y][0] + 50:
				check1 = True
			if pixdata[x, y][1] - 50 <= pixdata[x, y][2] <= pixdata[x, y][1] + 50:
				check2 = True
			if pixdata[x, y][2] - 50 <= pixdata[x, y][0] <= pixdata[x, y][2] + 50:
				check3 = True

			if check1 and check2 and check3:
				pixdata[x, y] = (255,255,255,0)
			else:
				# print(pixdata[x,y])
				pixdata[x,y] = closest(colours, pixdata[x,y])
				if pixdata[x,y] == (255,255,255,0):
					pass
				else:
					dataCSV[filename[:-4]][str(pixdata[x,y])] += 1
					dataCSV[filename[:-4]]['pix'] += 1
					# pdb.set_trace()
				# print('closest ' + str(pixdata[x,y]))

	path = r'./edited/'+filename[:-4]+'.png'
	img.save(path, 'PNG')
	dataJson = json.dumps(dataCSV)
	f = open('data.json','w')
	f.write(dataJson)
	f.close()
		# json.dump(dataCSV, outfile)
	print('done')

	# pd.read_json(json.dumps(dataCSV)).to_csv('pandas.csv')
	# dataCSV.to_csv('dict.csv', index = None)

# code from https://stackoverflow.com/a/54244301/13095638
def closest(colors,color):
    colors = np.array(colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))
    smallest_distance = colors[index_of_smallest]
    tpl = tuple(smallest_distance.reshape(1, -1)[0])
    if len(tpl) > 4:
    	tpl = tpl[:3]
    if tpl == (255,255,255,255):
    	tpl = (255,255,255,0)
    return tpl
    # return tuple(map(tuple, smallest_distance))

def main():
	filepath = r'C:/Users/ashbeck/Documents/Code/scripts/set1/'
	for month in range(1, 12):
		if month in [1,3,5,7,8,10,12]:
			if month < 10:
				pathMonth = '0%s' % (month)
			else:
				pathMonth = month

			for day in range(1, 32):
				if day < 10:
					pathDay = '0%s' % (day)
				else:
					pathDay = day
				for hour in hours:
					print('Opening Image...')
					filename = '2019' + str(pathMonth) + str(pathDay) + '_' + str(hour) + '.jpg'
					try:
						getImage(filepath, filename)
					except:
						print("Error Occurred")
		elif month in [4,6,9,11]:
			if month < 10:
				pathMonth = '0%s' % (month)
			else:
				pathMonth = month

			for day in range(1, 31):
				if day < 10:
					pathDay = '0%s' % (day)
				else:
					pathDay = day
				for hour in hours:
					print('Opening Image...')
					filename = '2019' + str(pathMonth) + str(pathDay) + '_' + str(hour) + '.jpg'
					try:
						getImage(filepath, filename)
					except:
						print("Error Occurred")
		elif month in [2]:
			if month < 10:
				pathMonth = '0%s' % (month)
			else:
				pathMonth = month

			for day in range(1, 29):
				if day < 10:
					pathDay = '0%s' % (day)
				else:
					pathDay = day
				for hour in hours:
					print('Opening Image...')
					filename = '2019' + str(pathMonth) + str(pathDay) + '_' + str(hour) + '.jpg'
					try:
						getImage(filepath, filename)
					except:
						print("Error Occurred")
	print('Processing Finished')
# print(dataCSV['filename']['bluePix'])

# filename = 'wong.png'
# print(filename[:-4])

main()

# getImage(r'./unedited/','20190501_0000.jpg')