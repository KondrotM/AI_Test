import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from PIL import Image
import colorsys


im = Image.open(r'E:/scripts/set1/20190101_0000.jpg', 'r')
pix_val = list(im.getdata())

imgVals = []
for i in pix_val:
	if i in imgVals:
		pass
	else:
		imgVals.append(i)

# print(imgVals)


palette = np.array([[255, 0, 0], # index 0: red
	[0,255,0], # index 1: green	
	[0,0,255], # index 2: blue
	[255,255,255], # index 3: white
	[0,0,0], # index 4 black
	[255,255,0]], # index 5: yellow
	dtype=np.uint8)

m, n = 4, 6


# indices = np.random.randint(0, len(palette), size=(4, 6))
# # indices = [[2,1,4,2,1],[5,2,1,3,5]]
# print(indices)
# print(len(palette))
# print(list(i for i in range(len(palette))))
# for i in range(len(palette)):
# 	print(palette[i])

# for i in range(np.random.randint(0,6)):
# 	print(palette[i])
print(imgVals[40:60])
plt.imshow(imgVals[40:60])
plt.colorbar()
plt.show()