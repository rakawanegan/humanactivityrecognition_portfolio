import copy
import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from lib.preprocess import load_data
from lib.local_utils import is_worse, SeqDataset

import datetime
import os

import joblib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from lib.preprocess import load_data
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


MODEL_NAME = "error_bar"
print("MODEL_NAME: ", MODEL_NAME)
start_date = datetime.datetime.now()
print("Start time: ", start_date)
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

dirname = f"result/{MODEL_NAME}/cnn1d"

x_train, x_test, y_train, y_test = load_data(
    LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED
)

# Hyperparameters
MAX_EPOCH = 200
BATCH_SIZE = 128
REF_SIZE = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
print("Device: ", device)
print("Max Epochs: ", MAX_EPOCH)
print("Early Stopping Reference Size: ", REF_SIZE)

model = Sequential()
model.add(
    Conv1D(160, 12, input_shape=(x_train.shape[1], x_train.shape[2]), activation="relu")
)
model.add(Conv1D(128, 10, activation="relu"))
model.add(Conv1D(96, 8, activation="relu"))
model.add(Conv1D(64, 6, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
# activation function is identity
model.add(Dense(6, activation="linear"))
# model.add(Dense(6, activation="softmax"))

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

y_pred = model.predict(x_test)
y_test = y_test.argmax(axis=-1)

y_pred = pd.DataFrame(y_pred)
y_test = y_test.argmax(axis=-1)
y_pred["true"] = y_test

y_pred.to_csv(f"{dirname}/predict.csv")
