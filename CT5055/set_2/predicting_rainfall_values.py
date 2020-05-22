import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import pdb

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv('set2_rainfall_data.csv', index_col = 0)
df['dateTime'] = pd.to_datetime(df['dateTime'])
df.set_index('dateTime', inplace=True)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
	data = []
	labels = []

	start_index = start_index + history_size

	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i)

		data.append(np.reshape(dataset[indices], (history_size, 1)))
		labels.append(dataset[i+target_size])

	return np.array(data), np.array(labels)


tf.random.set_seed(13)

print(df.head())
df.plot(subplots=True)
plt.show()
# pdb.set_trace()
# print('hey')

TRAIN_SPLIT = 2000

uni_data_values = df.values

univariate_past_history = 20

univariate_future_target = 0
pdb.set_trace()

x_train_uni, y_train_uni = univariate_data(uni_data_values, 0, TRAIN_SPLIT, univariate_past_history, univariate_future_target)

x_test_uni, y_test_uni = univariate_data(uni_data_values, TRAIN_SPLIT, None, univariate_past_history, univariate_future_target)
pdb.set_trace()
print ('Single window of past history')
print (x_train_uni[0])

print ('\n Target rainfall to predict')
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

# show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,'Baseline prediction example')
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test_univariate = tf.data.Dataset.from_tensor_slices((x_test_uni, y_test_uni))
test_univariate = test_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([ 
	tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
	tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

for x, y in test_univariate.take(1):
	print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=test_univariate, validation_steps=3)

for x, y in test_univariate.take(3):
	pdb.set_trace()
	plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
	plot.show()
