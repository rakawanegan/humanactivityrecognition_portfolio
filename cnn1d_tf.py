import datetime
import os

import joblib
import keras
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lib.preprocess import load_data

MODEL_NAME = "cnn1d_tf"
print(MODEL_NAME)
start_date = datetime.datetime.now()
print("Start time: ", start_date)
diridx = 0
dirname = f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}_{diridx}"
while os.path.exists(f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}_{diridx}"):
    dirname = f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}_{diridx}"
    diridx += 1
# Same labels will be reused throughout the program
LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40
# x, y, z acceleration as features
N_FEATURES = 3
# Define column name of the label vector
LABEL = "ActivityEncoded"
# set random seed
SEED = 314

x_train, x_test, y_train, y_test = load_data(
    LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED
)

model = Sequential()
model.add(
    Conv1D(160, 12, input_shape=(x_train.shape[1], x_train.shape[2]), activation="relu")
)
model.add(Conv1D(128, 10, activation="relu"))
model.add(Conv1D(96, 8, activation="relu"))
model.add(Conv1D(64, 6, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))

print(model.summary())

model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

epochs = 150
batch_size = 1024
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
)

joblib.dump(model, f"{dirname}/raw/model.pt")

y_pred = model.predict(x_test).argmax(axis=-1)
y_test = y_test.argmax(axis=-1)

predict = pd.DataFrame([y_pred, y_test]).T
predict.columns = ["predict", "true"]
predict.to_csv(f"{dirname}/raw/predict.csv")

sample = x_train[3199].reshape(1, TIME_PERIODS, N_FEATURES)
outputs = [model.layers[i].output for i in range(7)]
model_view = Model(inputs=model.inputs, outputs=outputs)
model_view.summary()

end_date = datetime.datetime.now()
print("End time: ", end_date)
print("Total time: ", end_date - start_date)

param = dict()
param["MODEL_NAME"] = MODEL_NAME
param["start_date"] = start_date
param["end_date"] = end_date
param["LABELS"] = LABELS
param["TIME_PERIODS"] = TIME_PERIODS
param["STEP_DISTANCE"] = STEP_DISTANCE
param["N_FEATURES"] = N_FEATURES
param["LABEL"] = LABEL
param["SEED"] = SEED

joblib.dump(param, f"{dirname}/raw/param.pkl")
