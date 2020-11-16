import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import layers
from tensorflow.keras import activations

DATA_DIR = '/home/mrrumpf/build/vscode-template-linux/rainfall.csv'
dataset = pd.read_csv(DATA_DIR)
rainfall_df = dataset[['rainfall']]

rainfall_df

train_split= 0.9
split_idx = int(len(rainfall_df) * 0.9)
training_set = rainfall_df[:split_idx].values
test_set = rainfall_df[split_idx:].values

# 5-day prediction using 30 days data
x_train = []
y_train = []
n_future = 5 #Next 5 days rainfall forecast
n_past = 30 #Past 30 days
for i in range(0, len(training_set) - n_past - n_future + 1):
    x_train.append(training_set[i : i + n_past, 0])
    y_train.append(training_set[i + n_past : i + n_past + n_future, 0])

x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1))

EPOCHS = 500
BATCH_SIZE = 32

regressor = Sequential()

regressor.add(Bidirectional(LSTM(units=30, return_sequences=True, input_shape = (x_train.shape[1], 1))))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = n_future, activation=activations.relu))

regressor.compile(optimizer="adam", loss="mean_squared_error" ,metrics=["acc"])

regressor.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

x_test = test_set[: n_past, 0]
y_test = test_set[n_past : n_past + n_future, 0]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (1, x_test.shape[0], 1))predicted_temperature = regressor.predict(x_test)

print('Predicted temperature {}'.format(predicted_temperature))
print('Real temperature {}'.format(y_test))