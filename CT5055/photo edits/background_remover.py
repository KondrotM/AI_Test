from PIL import Image
import colorsys
import json

months  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

hours = ['0000','0030','0100','0130','0200','0230','0300','0330','0400','0430','0500','0530','0600','0630','0700','0730','0800','0930','1000','1030','1100','1130','1200','1230','1300','1330','1400','1430','1500','1530','1600','1630','1700','1730','1800','1830','1900','1930','2000','2030','2100','2130','2200','2230','2300','2330']

pixCount = {}
def getImage(path, filename):
	im = Image.open(path+filename)
	img = im.convert('RGBA')
	datas = img.getdata()

	pixNo = 0


	newData = []
	for item in datas:
		check1 = False
		check2 = False
		check3 = False
		if item[0] - 50 <= item[1] <= item[0] + 50:
			check1 = True
		if item[1] - 50 <= item[2] <= item[1] + 50:
			check2 = True
		if item[2] - 50 <= item[0] <= item[2] + 50:
			check3 = True

		if check1 and check2 and check3:
			newData.append((item[0],item[1],item[2],0))
		else:
			newData.append(item)
			pixNo += 1

	img.putdata(newData)
	img.save(filename, "png")

	pixCount[filename] = pixNo
	pixJson = json.dumps(pixCount)
	f = open('dict.json','w')
	f.write(pixJson)
	f.close()

	print('Image %s Saved' % (filename))

def main():
	pixCount = {}
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

main()
# getImage('t')