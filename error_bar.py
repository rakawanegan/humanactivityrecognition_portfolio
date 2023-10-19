import copy
import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from lib.model import ConvolutionalNetwork as CNN1D
from lib.preprocess import load_data
from lib.local_utils import SeqDataset

import datetime
import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from lib.preprocess import load_data
import datetime
import os

import joblib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

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

input_size = x_train.shape[2]
num_classes = 6

epochs = 150
BATCH_SIZE = 1024
REF_SIZE = 3
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

losslist = list()
model.train()
best_model = copy.deepcopy(model)
for epoch in range(epochs):
    losses = list()
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    ls = np.mean(losses)
    if epoch > 10:
        if min(losslist) > ls:
            best_model = copy.deepcopy(model)
        # if is_worse(losslist, REF_SIZE, "minimize"):
        #     break
    if epoch % 10 == 0:
        print("Epoch: ", epoch, "Loss: ", loss.item())
    losslist.append(ls)

model = best_model
model.eval()
y_pred = list()
for batch in test_loader:
    x, _ = batch
    x = x.to(device)
    optimizer.zero_grad()
    output = model(x)
    y_pred.append(output.detach().cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0)
y_test = y_test.argmax(axis=-1)
y_pred = pd.DataFrame(y_pred)
y_pred["true"] = y_test

y_pred.to_csv(f"{dirname}/predict.csv")
