import os

import datetime
import joblib
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lib.model import ViT
from lib.preprocess import get_data


MODEL_NAME = "optuna_vit1d"
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


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

search_space = {
    "patch_size": [1, 2, 5, 8, 10, 16, 40],
    "dim": [32, 64, 128, 256, 512],
    "depth": [3, 5, 6, 8],
    "heads": [3, 5, 6, 8, 10],
    "mlp_dim": [256, 512, 1024, 2048],
    "dropout": [0.01, 0.1, 0.25, 0.5, 0.8],
    "emb_dropout": [0.01, 0.1, 0.25, 0.5, 0.8],
    "batch_size": [32, 64, 128, 256, 512],
}

def obj(trial):
    params = {
        "seq_len": TIME_PERIODS,
        "num_classes": len(LABELS),
        "channels": N_FEATURES,
        "patch_size": trial.suggest_categorical("patch_size", search_space["patch_size"]),
        "dim": trial.suggest_categorical("dim", search_space["dim"]),
        "depth": trial.suggest_categorical("depth", search_space["depth"]),
        "heads": trial.suggest_categorical("heads", search_space["heads"]),
        "mlp_dim": trial.suggest_categorical("mlp_dim", search_space["mlp_dim"]),
        "dropout": trial.suggest_categorical("dropout", search_space["dropout"]),
        "emb_dropout": trial.suggest_categorical(
            "emb_dropout", search_space["emb_dropout"]
        ),
    }
    batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()
    )
    model = ViT(**params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    accuracies = list()
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
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
best_params["seq_len"] = TIME_PERIODS
best_params["num_classes"] = len(LABELS)
best_params["channels"] = N_FEATURES

epochs = 100
batch_size = best_params.pop("batch_size")
model = ViT(**best_params).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
train_loader = DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()
)

loss_list = [np.inf]
for ep in range(1, epochs):
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
    if loss_list[-1] < ls:
        print(f"early stopping at epoch {ep} with loss {ls:.5f}")
        break
    print(f"Epoch {ep + 0:03}: | Loss: {ls:.5f}")
    loss_list.append(ls)


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

predict = pd.DataFrame([y_pred,y_test], columns=["predict", "true"])
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