"""
dnn model definition
"""

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# model definition
def dnn_model(radius_points, hidden_layers, layer_size):
	"""
	dnn model definition
	"""
	model = Sequential()
	model.add(Input(shape=(radius_points, )))
	for _ in range(hidden_layers):
		model.add(Dense(layer_size, activation='relu'))
	model.add(Dense(1))
	return model

