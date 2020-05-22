from PIL import Image
import colorsys
import json
import numpy as np
import glob, os

# import pandas as pd
import pdb

colours = [(50,50,255,255), # blue
(100,150,255,255), # light_blue
(150,150,50,255), # olive
(247,247,150,255), # yellow
(255,200,50,255), # orange
(255,50,50,255), # red
(255,50,255,255), # purple
(255,255,255,255) # white
]

def getImage(img):
	# img = Image.open(path+filename)
	img = img.convert('RGBA')
	# datas = img.getdata()
	pixdata = img.load()

	width, height = img.size

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
				pixdata[x,y] = unicolour(pixdata[x,y])
					# pdb.set_trace()
				# print('closest ' + str(pixdata[x,y]))

	return img


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

def unicolour(colour):
	if colour[3] == 255:
		tpl = (0,0,0,255)

	else:
		tpl = (255,255,255,0)

	return tpl






def main():
	for infile in glob.glob("./thumbnails/edited/*.png"):
		file, ext = os.path.splitext(infile)
		img = Image.open(infile)

		img = getImage(img)
		img.save('./thumbnails/bw/'+file[-23:]+'.png', 'PNG')
	print('Images Saved')

		# pdb.set_trace()

main()

