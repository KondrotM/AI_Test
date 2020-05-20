# Tutorial followed from https://www.tensorflow.org/tutorials/structured_data/time_series

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
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
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


tf.random.set_seed(13)

TRAIN_SPLIT = 300000

# just picking what data we want to analyse
uni_data = df['T (degC)']
uni_data.index = df['Date Time']
print(uni_data.head())

# uni_data.plot(subplots=True)


# normalising data points
uni_data = uni_data.values

uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

uni_data = (uni_data-uni_train_mean)/uni_train_std


# picking history of 20
univariate_past_history = 20

# you want to predict the immediately next value, not x units into the future
univariate_future_target = 0

# the data used to train the algorithm
x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,univariate_past_history, univariate_future_target)

# the data used to predict the data
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
univariate_past_history, univariate_future_target)

print ('Single window of past history')
print (x_train_uni[0])

print ('\n Target temperature to predict')
print (y_train_uni[0])

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
			plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
		else:
			plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future+5)*2])
	plt.xlabel('Time-Step')
	plt.show()
	return plt

def baseline(history):
	return np.mean(history)

# show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
#            'Baseline Prediction Example')

### RECURRENT NEURAL NETWORK
# Well-suited for time-series data
# RNNs process the time-series data step-by-step, maintaining an internal state which summarises the information they've seen
# RNN tutorial: https://www.tensorflow.org/tutorials/text/text_generation
# The following is a specialised RNN layer called Long Short Term Memory (LSTM)

# Data must be shuffled and packed into batches before it is trained
# Batch size is the number of examples
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BATCH_SIZE = 256
BUFFER_SIZE = 10000


train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


# > x_train_uni.shape
# > [29980, 20, 1]
simple_lstm_model = tf.keras.models.Sequential([ 
	tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
	tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

for x, y in val_univariate.take(1):
	print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_univariate, validation_steps=50)

for x, y in val_univariate.take(3):
	pdb.set_trace()
	plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
	plot.show()


# show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
