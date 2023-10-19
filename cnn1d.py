import datetime
import os

import joblib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from lib.model import ConvolutionalNetwork1D as CNN1D

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

input_size = x_train.shape[2]
num_classes = 6
model = CNN1D(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters())

epochs = 150
batch_size = 1024

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(np.argmax(y_train, axis=1))

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(np.argmax(y_test, axis=1))

for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        inputs = x_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    outputs = model(x_test)
    predicted = torch.argmax(outputs, dim=1)
    correct = (predicted == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total
    print(f'Accuracy on test data: {100 * accuracy:.2f}%')

joblib.dump(model, f"{dirname}/raw/model.pt")

y_pred = model.predict(x_test).argmax(axis=-1)
y_test = y_test.argmax(axis=-1)

predict = pd.DataFrame([y_pred, y_test]).T
predict.columns = ["predict", "true"]
predict.to_csv(f"{dirname}/raw/predict.csv")

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
