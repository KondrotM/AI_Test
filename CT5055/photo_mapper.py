from PIL import Image
import colorsys
import json
import numpy as np
# import pandas as pd
import pdb
import pathlib
import pandas as pd
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
# session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
colours = {
	(255,255,255,0) : 0, # white
	(0,0,0,255) : 1, # blue
}

pixels = {
	0: (255,255,255,0),
	1: (0,0,0,255),
}

imgs_list = []

def getImage(path):
	img = Image.open(path)
	img = img.convert('RGBA')

	pixdata = img.load()

	width, height = img.size

	# pdb.set_trace()
	img = []

	for y in range(height):
		width_list = []
		for x in range(width):
			img.append(colours[pixdata[x, y]])
		# img.append(width_list)


	return np.array(img)

def parseImage(img, limiter):
	im = Image.new('RGBA', (256, 384))
	pix = im.load()
	x = 0
	y = -1
	for xy in range(256 * 384):
		if xy % 256 == 0:
			y += 1
			x = 0
		if img[xy] > limiter:
			val = 1
		else:
			val = 0
		pix[x, y] = pixels[val]
		x += 1

	im.save('test.png', 'PNG')

	# # pdb.set_trace()
	# for y in range(1152):
	# 	for x in range(768):

	# 		# pdb.set_trace()
	# 		pix[x, y] = pixels[img[y + x]]
	# im.save("test.png", "PNG")


def getImages(filepath):
	
	data_dir = pathlib.Path(filepath)
	images = list(data_dir.glob('*'))

	imgs = []
	dateTime = []

	for i in images:
		print("Processing image %s" % i)
		img = getImage(i)
		# parseImage(img)
		# return
		imgs.append(img)
		# pdb.set_trace()
		dateTime.append(str(i)[72:85])
		# pdb.set_trace()

	return np.array(imgs), np.array(dateTime)


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i, step)
		data.append(dataset[indices])

		if single_step:
			labels.append(target[i+target_size])
		else:
			labels.append(target[i:i+target_size])

	return np.array(data), np.array(labels)

# img = getImage(r'C:/Users/ashbeck/Documents/GitHub/AI_Test/CT5055/sample_1/edited/','20190501_0000.png')
def main():

	BATCH_SIZE = 12
	TRAIN_SPLIT = 100
	BUFFER_SIZE = 100
	past_history = 20
	future_target = 24
	STEP = 1

	EPOCHS = 10
	EVALUATION_INTERVAL = 200
	# x_imgs, y_imgs = getImages(r'C:/Users/ashbeck/Documents/GitHub/AI_Test/CT5055/sample_1/edited/')
	# x_train = x_imgs[:100]
	# y_train = y_imgs[:100]

	# x_val = x_imgs[100:]
	# y_val = y_imgs[100:]
	# pdb.set_trace()

	imgs, datetime = getImages(r'C:/Users/ashbeck/Documents/GitHub/AI_Test/CT5055/sample_1/thumbnails/bw/')

	df = pd.DataFrame(imgs)
	df.index = datetime
	df.index = pd.to_datetime(df.index, format = '%Y%m%d_%H%S')
	
	dataset = df.values


	x_train_single, y_train_single = multivariate_data(dataset, dataset, 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step = True)

	x_val_single, y_val_single = multivariate_data(dataset, dataset, TRAIN_SPLIT, None, past_history, future_target, STEP, single_step = True)

	print ('Single window of past history : {}'.format(x_train_single[0].shape))
	# pdb.set_trace()

	train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
	train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	# pdb.set_trace()
	single_step_model = tf.keras.models.Sequential()
	single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
	# pdb.set_trace()
	single_step_model.add(tf.keras.layers.Dense(y_train_single.shape[1]))

	single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

	single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_single, validation_steps=3)

	for x, y in val_data_single.take(1):
		pdb.set_trace()
		print(single_step_model.predict(x).shape)

	for x, y in val_data_single.take(3):
		pdb.set_trace()
		plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(), single_step_model.predict(x)[0]], 12, 'Single Step Prediction')
		plot.show()

	# df = pd.DataFrame({'dateTime' : datetime, 'imgs' : imgs})
	# print(df.head())

main()
# pdb.set_trace()
# parseImage(img)
