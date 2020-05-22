from PIL import Image
import glob, os

# taken from https://pillow.readthedocs.io/en/3.1.x/reference/Image.html

size = 256, 384

for infile in glob.glob("../sample_1/edited/*.png"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.thumbnail(size)
    im.save("../sample_1/thumbails/" + file[-13:] + ".thumbnail.png", "PNG")
    # print(file[-13:])
    # break;

print('Files Thumbnailed')