import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import layers
from tensorflow.keras import activations

DATA_DIR = '/home/mrrumpf/build/vscode-template-linux/planb.csv'
dataset = pd.read_csv(DATA_DIR)
temp_df = dataset[['DailyAverageTemp']]
temp_df['DailyAverageTemp'] = temp_df['DailyAverageTemp'].astype(np.float32)

print(temp_df.shape)
print(temp_df.columns)
print(temp_df.info())
print(temp_df.isnull().sum())

train_split= 0.9
split_idx = int(len(temp_df) * 0.9)
training_set = temp_df[:split_idx].values
test_set = temp_df[split_idx:].values

# 5-day prediction using 30 days data
x_train = []
y_train = []
n_future = 5 #Next 5 days forecast
n_past = 30 #Past 30 days
for i in range(0, len(training_set) - n_past - n_future + 1):
    x_train.append(training_set[i : i + n_past, 0])
    y_train.append(training_set[i + n_past : i + n_past + n_future, 0])

x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1))

EPOCHS = 50
BATCH_SIZE = 10

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
regressor.save("planb.tfmodel")

x_test = test_set[: n_past, 0]
y_test = test_set[n_past : n_past + n_future, 0]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
print("xtest: ", x_test)
predicted_temperature = regressor.predict(x_test)

print('Predicted temperature {}'.format(predicted_temperature))
print('Real temperature {}'.format(y_test))

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_saved_model("/home/mrrumpf/build/planb/vscode-template-linux/planb.tfmodel")
model_no_quant_tflite = converter.convert()

# Save the model to disk
open("/home/mrrumpf/build/planb/vscode-template-linux/planb.tflite", "wb").write(model_no_quant_tflite)

# Convert the model to the TensorFlow Lite format with quantization
#def representative_dataset():
#  for i in range(500):
#    yield([x_train[i].reshape(1, 1)])
# Set the optimization flag.
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce integer only quantization
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8
##converter.inference_output_type = tf.int8
# Provide a representative dataset to ensure we quantize correctly.
#converter.representative_dataset = representative_dataset
#model_tflite = converter.convert()

# Save the model to disk
#open("/home/mrrumpf/build/planb/vscode-template-linux/planb.tflite", "wb").write(model_tflite)


# Install xxd if it is not available
#!apt-get update && apt-get -qq install xxd
# Convert to a C source file, i.e, a TensorFlow Lite for Microcontrollers model
#!xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}
# Update variable names
#REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
#!sed -i 's/'{REPLACE_TEXT}'/g_model/g' {MODEL_TFLITE_MICRO}
#xxd -i planb.tflite > planb.cc