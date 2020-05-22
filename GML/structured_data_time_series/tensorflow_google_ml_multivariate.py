import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import pdb

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# zip_path = tf.keras.utils.get_file(
#		 origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#		 fname='jena_climate_2009_2016.csv.zip',
#		 extract=True)
# csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv('jena_climate_2009_2016.csv')

def univariate_data(dataset, start_index, end_index, history_size, target_size):
	# data is a list of array points which the computer will analyse to associate with a label, they look kinda like this:
	# [-1.2,-3,-0.4],[-3,-0.4,-1],[-0.4,-1,0.2],...
	# there number of points in each array is given as the 'history_size' (20)
	# so the numbers in each array are always a sample of size 20 where each number is a point in history
	data = []
	# labels are the correct values to conclude from the data being predicted
	# they look kinda like this : [0.42,0.60,0.73,...]
	labels = []


	# if history size is 20, you can't take the 20 previous samples if you start at 1
	start_index = start_index + history_size

	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i)
		# Reshape data from (history_size,) to (history_size, 1)
		data.append(np.reshape(dataset[indices], (history_size, 1)))
		labels.append(dataset[i+target_size])
	# pdb.set_trace()
	return np.array(data), np.array(labels)

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


tf.random.set_seed(13)

BATCH_SIZE = 256
TRAIN_SPLIT = 300000
BUFFER_SIZE = 10000
past_history = 720
future_target = 72
EPOCHS = 10
EVALUATION_INTERVAL = 200
STEP = 6


features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']

features = df[features_considered]
features.index = df['Date Time']

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

x_train_single, y_train_single = multivariate_data(dataset, dataset, 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset, TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=True)
print ('Single window of past history : {}'.format(x_train_single[0].shape))
pdb.set_trace()

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(3))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

for x, y in val_data_single.take(1):
	print(single_step_model.predict(x).shape)
	# pdb.set_trace()

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_single, validation_steps=50)


def plot_train_history(history, title):
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(loss))

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title(title)
	plt.legend()

	plt.show()

plot_train_history(single_step_history, 'Single Step Training and validation loss')
def create_time_steps(length):
	return list(range(-length, 0))

def show_plot(plot_data, delta, title):
	labels = ['History', 'True Future', 'Model Prediction']
	marker = ['.-', 'rx', 'go']
	time_steps = create_time_steps(plot_data[0].shape[0])
	if delta:
		future = delta
	else:
		future = 0

	plt.title(title)
	for i, x in enumerate(plot_data):
		if i:
			plt.plot(future, plot_data[i], marker[i], markersize=10,
							 label=labels[i])
		else:
			plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future+5)*2])
	plt.xlabel('Time-Step')
	return plt

for x, y in val_data_single.take(3):
	pdb.set_trace()
	plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(), single_step_model.predict(x)[0]], 12, 'Single Step Prediction')
	plot.show()


# single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_single, validation_steps=50)