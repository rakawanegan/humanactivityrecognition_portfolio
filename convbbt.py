import copy
import datetime
import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from lib.model import PreConvTransformer
from lib.preprocess import get_data

MODEL_NAME = "convbbt"
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

x_train, x_test, y_train, y_test = get_data(
    LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES
)

# Hyperparameters
MAX_EPOCH = 500
BATCH_SIZE = 128
REF_SIZE = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
print("Device: ", device)
print("Max Epochs: ", MAX_EPOCH)
print("Early Stopping Reference Size: ", REF_SIZE)

pct_params = {
    "hidden_ch": 15,
    "num_classes": len(LABELS),
    "input_dim": TIME_PERIODS,
    "channels": N_FEATURES,
    "hidden_dim": 128,
    "depth": 5,
    "heads": 8,
    "mlp_dim": 1024,
    "dropout": 0.01,
    "emb_dropout": 0.01,
}
model = PreConvTransformer(**pct_params).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


class SeqDataset(TensorDataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train = SeqDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
test = SeqDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()
)


def is_worse(losslist, REF_SIZE, axis="minimize"):
    if axis == "minimize":
        return all(
            x > y for x, y in zip(losslist[-REF_SIZE:], losslist[-REF_SIZE - 1 : -1])
        )
    elif axis == "maximize":
        return all(
            x < y for x, y in zip(losslist[-REF_SIZE:], losslist[-REF_SIZE - 1 : -1])
        )
    else:
        raise ValueError("Invalid axis value: " + axis)


losslist = list()

for ep in range(1, MAX_EPOCH + 1):
    losses = list()
    for batch in train_loader:
        x, t = batch
        x = x.to(device)
        t = t.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    ls = np.mean(losses)
    if ep > REF_SIZE:
        if min(losslist) > ls:
            best_model = copy.deepcopy(model)
        if is_worse(losslist, REF_SIZE, "minimize"):
            print(f"early stopping at epoch {ep} with loss {ls:.5f}")
            break
    print(f"Epoch {ep + 0:03}: | Loss: {ls:.5f}")
    losslist.append(ls)
model = best_model
plt.plot(losslist)
plt.title("Loss curve")
plt.xlabel("Epoch")
plt.ylabel("Loss mean")
plt.savefig(f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}/processed/loss.png")

model.eval()
joblib.dump(model, f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}/raw/model.pkl")
y_pred = list()
for batch in test_loader:
    x, _ = batch
    x = x.to(device)
    optimizer.zero_grad()
    output = model(x)
    y_pred.append(output.detach().cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0).argmax(axis=-1)
y_test = y_test.argmax(axis=-1)

predict = pd.DataFrame([y_pred, y_test]).T
predict.columns = ["predict", "true"]
predict.to_csv(f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}/raw/predict.csv")

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

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

joblib.dump(param, f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}/raw/param.pkl")
