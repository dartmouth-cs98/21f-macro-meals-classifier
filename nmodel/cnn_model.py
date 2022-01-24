#import tflearn
import tensorflow as tf
#from tflearn.python.layers.conv import conv_2d, max_pool_2d
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, InputLayer, Dropout, Dense
from tensorflow.python.keras.optimizers import Adam
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression

def get_model(IMG_SIZE,no_of_fruits,LR):
	try:
		tf.reset_default_graph()
	except:
		print("tensorflow")
	#convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
	#convnet = conv_2d(convnet, 32, 5, activation='relu')
	#convnet = max_pool_2d(convnet, 5)
	#convnet = conv_2d(convnet, 64, 5, activation='relu')
	#convnet = max_pool_2d(convnet, 5)
	#convnet = conv_2d(convnet, 128, 5, activation='relu')
	#convnet = max_pool_2d(convnet, 5)
	#convnet = conv_2d(convnet, 64, 5, activation='relu')
	#convnet = max_pool_2d(convnet, 5)
	#convnet = conv_2d(convnet, 32, 5, activation='relu')
	#convnet = max_pool_2d(convnet, 5)
	#convnet = fully_connected(convnet, 1024, activation='relu')
	#convnet = dropout(convnet, 0.8)
	#convnet = fully_connected(convnet, no_of_fruits, activation='softmax')


	#convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	#model = tflearn.DNN(convnet, tensorboard_dir='log')

	model = tf.keras.Sequential()
	model.add(InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3), name='input'))
	model.add(Conv2D(filters = 32,  kernel_size = 5, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
	model.add(MaxPool2D(5))
	model.add(Conv2D(64, 5, activation='relu'))
	model.add(MaxPool2D(5))
	model.add(Conv2D(128, 5, activation='relu'))
	model.add(MaxPool2D(5))
	#model.add(Conv2D(64, 5, activation='relu'))
	#model.add(MaxPool2D(5))
	#model.add(Conv2D(32, 5, activation='relu'))
	#model.add(MaxPool2D(5))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.8))
	model.add(Dense(no_of_fruits, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer=Adam(), name='targets')

	return model