from PIL import Image
import colorsys
import pandas as pd


im = Image.open(r'E:/scripts/set1/20190101_0000.jpg', 'r')
img = im.convert("RGBA")
datas = img.getdata()


newData = []
for item in datas:
	check1 = False
	check2 = False
	check3 = False
	if item[0] - 10 <= item[1] <= item[0] + 10:
		check1 = True
	if item[1] - 10 <= item[2] <= item[1] + 10:
		check2 = True
	if item[2] - 10 <= item[0] <= item[2] + 10:
		check3 = True

	if check1 and check2 and check3:
		newData.append((item[0],item[1],item[2],0))
	else:
		newData.append(item)

img.putdata(newData)
img.save("img2.png", "PNG")


# pix_val = list(im.getdata())

# imgVals = []
# for i in pix_val:
# 	if i in imgVals:
# 		pass
# 	else:
# 		imgVals.append(i)



# print(imgVals)

# def HSVColour(img):
# 	if isinstance(img, Image.Image):
# 		r, g, b = img.split()
# 		Hdat = []
# 		Sdat = []
# 		Vdat = []
# 		for rd, gn, bl in zip(r.getdata(),g.getdata(),b.getdata()):
# 			h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)

# 			Hdat.append(int(h*255.))
# 			Sdat.append(int(s*255.))
# 			Vdat.append(int(v*255.))
# 		r.putdata(Hdat)
# 		g.putdata(Sdat)
# 		b.putdata(Vdat)
# 		return Image.merge('RGB',(r,g,b))
# 	else:
# 		return None

# b = HSVColour(im)
# b.save('b.jpg')
