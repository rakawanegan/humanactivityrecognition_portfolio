import copy
import datetime
import os

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from lib.model import PreConvTransformer
from lib.preprocess import load_data
from lib.local_utils import is_worse, SeqDataset


MODEL_NAME = "optuna_convbbt"
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

x_train, x_test, y_train, y_test = load_data(
    LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED
)

dirname = f"result/0727_{MODEL_NAME}_0"

# Hyperparameters
MAX_EPOCH = 200
BATCH_SIZE = 128
REF_SIZE = 5
TIMEOUT_HOURS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
print("Device: ", device)
print("Max Epochs: ", MAX_EPOCH)
print("Early Stopping Reference Size: ", REF_SIZE)


train = SeqDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
test = SeqDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())

adam_searchspace = {
    "lr": [1e-6, 1e-5, 1e-4, 1e-3],
    "beta1": [0.9, 0.95, 0.99, 0.999],
    "beta2": [0.9, 0.95, 0.99, 0.999],
    "eps": [1e-9, 1e-8, 1e-7, 1e-6],
}

calr_searchspace = {
    "T_max": [50, 100, 150, 200],
    "eta_min": [0, 1e-8, 1e-7, 1e-6, 1e-5],
}

convbbt_searchspace = {
    "hidden_ch": [3, 5, 7, 8, 10, 15],
    "depth": [3, 5, 6, 8],
    "heads": [3, 5, 6, 8, 10],
    "hidden_dim": [64, 128, 256, 512, 1024],
    "mlp_dim": [256, 512, 1024, 2048],
    "dropout": [0.01, 0.1, 0.25, 0.5, 0.8],
    "emb_dropout": [0.01, 0.1, 0.25, 0.5, 0.8],
}

search_space = adam_searchspace | calr_searchspace | convbbt_searchspace
print("Search Space: ", search_space)

study = joblib.load(f"{dirname}/study.pkl")

all_params = dict(study.best_params)
adam_params = {k: all_params[k] for k in adam_searchspace.keys()}
adam_params["betas"] = (adam_params.pop("beta1"), adam_params.pop("beta2"))
calr_params = {k: all_params[k] for k in calr_searchspace.keys()}
convbbt_params = {k: all_params[k] for k in convbbt_searchspace.keys()}
convbbt_params["input_dim"] = TIME_PERIODS
convbbt_params["num_classes"] = len(LABELS)
convbbt_params["channels"] = N_FEATURES

loss_function = nn.CrossEntropyLoss()
model = PreConvTransformer(**convbbt_params).to(device)
optimizer = torch.optim.Adam(model.parameters(), **adam_params)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **calr_params)


train = SeqDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
test = SeqDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()
)
loss_list = list()

best_model = copy.deepcopy(model)
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
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    ls = np.mean(losses)
    if ep > calr_params["T_max"]:
        if min(loss_list) > ls:
            best_model = copy.deepcopy(model)
        if is_worse(loss_list, REF_SIZE, "minimize"):
            print(f"early stopping at epoch {ep} with loss {ls:.5f}")
            break
    print(f"Epoch {ep + 0:03}: | Loss: {ls:.5f}")
    loss_list.append(ls)
model = best_model

plt.plot(loss_list)
plt.title("Loss curve")
plt.xlabel("Epoch")
plt.ylabel("Loss mean")
plt.savefig(f"{dirname}/processed/assets/loss.png")


model.eval()
joblib.dump(
    model, f"{dirname}/raw/model.pkl"
)
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
predict.to_csv(f"{dirname}/raw/predict.csv")

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
param["search_space"] = search_space
param["TIMEOUT_HOURS"] = TIMEOUT_HOURS

joblib.dump(param, f"{dirname}/raw/param.pkl")
