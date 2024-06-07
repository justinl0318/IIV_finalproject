import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

ACCOUNTED_LENGTH = 10

dataset = np.load("dataset.npy")

angles = []
for i in range(1, len(dataset)):
    dx = dataset[i][0] - dataset[i-1][0]
    dy = dataset[i][1] - dataset[i-1][1]
    angle = math.atan2(dy, dx)
    angles.append(angle)


x, y = [], []
for i in range(len(angles) - ACCOUNTED_LENGTH):
    x.append(angles[i: i+ACCOUNTED_LENGTH])
    y.append(angles[i+ACCOUNTED_LENGTH])

X = np.array(x)
y = np.array(y)

split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Build the model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))  # Output layer with 1 unit for predicting the next angle
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))

model.save("trajectory_model.h5")