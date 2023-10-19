import datetime
import os

import joblib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.model import ConvolutionalNetwork as CNN1D
from lib.local_utils import SeqDataset
from lib.preprocess import load_data

MODEL_NAME = "cnn1d"
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
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

input_size = x_train.shape[2]
num_classes = 6

epochs = 150
BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN1D(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters())

train = SeqDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
test = SeqDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()
)


for epoch in range(epochs):
    model.train()
    for i, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

model.eval()
y_pred = list()
for batch in test_loader:
    x, _ = batch
    x = x.to(device)
    optimizer.zero_grad()
    output = model(x)
    y_pred.append(output.detach().cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0).argmax(axis=-1)
y_test = y_test.argmax(axis=-1)

torch.save(
    model.to('cpu').state_dict(),
    f"{dirname}/raw/model.pt"
)
torch.save(y_test, f"{dirname}/raw/y_test.tsr")
torch.save(x_test, f"{dirname}/raw/x_test.tsr")

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
