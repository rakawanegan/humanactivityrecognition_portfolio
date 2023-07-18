import copy
import datetime
import os

import joblib
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from lib.model import PreConvTransformer
from lib.preprocess import get_data
from lib.local_utils import is_worse


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

x_train, x_test, y_train, y_test = get_data(
    LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES
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


search_space = {
    "hidden_ch": [3, 5, 7, 8, 10, 15],
    "depth": [3, 5, 6, 8],
    "heads": [3, 5, 6, 8, 10],
    "hidden_dim": [64, 128, 256, 512, 1024],
    "mlp_dim": [256, 512, 1024, 2048],
    "dropout": [0.01, 0.1, 0.25, 0.5, 0.8],
    "emb_dropout": [0.01, 0.1, 0.25, 0.5, 0.8],
    "batch_size": [16, 32, 64, 128, 256, 512],
}
print("Search Space: ", search_space)


def obj(trial):
    params = {
        "num_classes": len(LABELS),
        "input_dim": TIME_PERIODS,
        "channels": N_FEATURES,
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim", search_space["hidden_dim"]
        ),
        "hidden_ch": trial.suggest_categorical("hidden_ch", search_space["hidden_ch"]),
        "depth": trial.suggest_categorical("depth", search_space["depth"]),
        "heads": trial.suggest_categorical("heads", search_space["heads"]),
        "mlp_dim": trial.suggest_categorical("mlp_dim", search_space["mlp_dim"]),
        "dropout": trial.suggest_categorical("dropout", search_space["dropout"]),
        "emb_dropout": trial.suggest_categorical(
            "emb_dropout", search_space["emb_dropout"]
        ),
    }
    train_loader = DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()
    )
    test_loader = DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()
    )
    model = PreConvTransformer(**params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losslist = list()
    for epoch in range(MAX_EPOCH):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losslist.append(loss.item())
        ls = np.mean(losses)
        losslist.append(ls)
        p_models.append(copy.deepcopy(model))
        if ep > REF_SIZE and is_worse(losslist, REF_SIZE, "minimize"):
            print(f"early stopping at epoch {ep} with loss {ls:.5f}")
            model = p_models[-REF_SIZE]
            break
        if ep > REF_SIZE:
            del p_models[0]  # del oldest model
    model.eval()
    accuracies = list()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            accuracies.append(
                accuracy_score(
                    outputs.detach().cpu().numpy().argmax(axis=-1),
                    labels.cpu().numpy().argmax(axis=-1),
                )
            )
    accuracy = np.mean(accuracies)
    print(f"Accuracy: {accuracy}")
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(obj, n_trials=1000)
print(study.best_trial)
joblib.dump(study, f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}/raw/study.pkl")

best_params = dict(study.best_params)
best_params["input_dim"] = TIME_PERIODS
best_params["num_classes"] = len(LABELS)
best_params["channels"] = N_FEATURES


model = PreConvTransformer(**best_params).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


train = SeqDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
test = SeqDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()
)
loss_list = list()
p_models = list()

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
    loss_list.append(ls)
    if ep > REF_SIZE and is_worse(loss_list, REF_SIZE, "minimize"):
        print(f"early stopping at epoch {ep} with loss {ls:.5f}")
        model = p_models[-REF_SIZE]
        break
    if ep > REF_SIZE:
        del p_models[0]  # del oldest model
    loss_list.append(ls)
    print(f"Epoch {ep + 0:03}: | Loss: {ls:.5f}")


model.eval()
joblib.dump(
    model, f"result/{start_date.strftime('%m%d')}_optuna{MODEL_NAME}/raw/model.pkl"
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
param["search_space"] = search_space

joblib.dump(param, f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}/raw/param.pkl")
