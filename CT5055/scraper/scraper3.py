import os
import time
import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup
import pdb

months  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

hours = ['0000','0030','0100','0130','0200','0230','0300','0330','0400','0430','0500','0530','0600','0630','0700','0730','0800','0930','1000','1030','1100','1130','1200','1230','1300','1330','1400','1430','1500','1530','1600','1630','1700','1730','1800','1830','1900','1930','2000','2030','2100','2130','2200','2230','2300','2330']
def hoursTest():
	for hour in hours:
		url = "https://www.metcheck.com/DATA/ARCHIVE/PRECIPITYPE/UK/ARCHIVE/2019/2/10/2019210_%s.jpg" % (hour)
		urllib.request.urlretrieve(url, os.path.basename(url))
		time.sleep(5)

# hoursTest()

def main():
	for month in range(1, 12):
		if month in [1,3,5,7,8,10,12]:
			print('Month Downloaded')

		elif month in [4,6,9,11]:
			if month < 10:
				dlMonth = '0%s' % (month)
			else:
				dlMonth = month
			for day in range (1, 32):
				if day < 10:
					dlDay = '0%s' % (day)
				else:
					dlDay = day
				for hour in hours:
					url = "https://www.metcheck.com/DATA/ARCHIVE/PRECIPITYPE/UK/ARCHIVE/2019/%s/%s/2019%s%s_%s.jpg" % (month, day, month, day, hour)
					dlUrl = "https://www.metcheck.com/DATA/ARCHIVE/PRECIPITYPE/UK/ARCHIVE/2019/1/10/2019%s%s_%s.jpg" % (dlMonth, dlDay, hour)
					urllib.request.urlretrieve(url, os.path.basename(dlUrl))
					print(url)
					print(dlUrl)
					print ('Downloaded %s 	%s 	%s' % (month, day, hour))
					time.sleep(3)
		else:
			if month < 10:
				dlMonth = '0%s' % (month)
			else:
				dlMonth = month
			for day in range (1, 32):
				if day < 10:
					dlDay = '0%s' % (day)
				else:
					dlDay = day
				for hour in hours:
					url = "https://www.metcheck.com/DATA/ARCHIVE/PRECIPITYPE/UK/ARCHIVE/2019/%s/%s/2019%s%s_%s.jpg" % (month, day, month, day, hour)
					dlUrl = "https://www.metcheck.com/DATA/ARCHIVE/PRECIPITYPE/UK/ARCHIVE/2019/1/10/2019%s%s_%s.jpg" % (dlMonth, dlDay, hour)
					urllib.request.urlretrieve(url, os.path.basename(dlUrl))
					print(url)
					print(dlUrl)
					print ('Downloaded %s 	%s 	%s' % (month, day, hour))
					time.sleep(3)

main()

# urllib.request.urlretrieve(url, os.path.basename(url))

# html = urllib.request.urlopen(url)
# soup = BeautifulSoup(html, features="html.parser")

# print(soup)

# imgs = soup.findAll("div", {"class":"thumb-pic"})
# print(imgs)
# for img in imgs:
#         imgUrl = img.a['href'].split("imgurl=")[1]
#         urllib.request.urlretrieve(imgUrl, os.path.basename(imgUrl))