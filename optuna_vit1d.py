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

from lib.model import ViT
from lib.preprocess import get_data
from lib.local_utils import is_worse, SeqDataset


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

diridx = 0
while os.path.exists(f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}_{diridx}"):
    diridx += 1
diridx -= 1
dirname = f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}_{diridx}"

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


train = SeqDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
test = SeqDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())

adam_searchspace = {
    "lr": [],
    "beta1": [],
    "beta2": [],
    "eps": [],
}

calr_searchspace = {
    "T_max": [],
    "eta_min": [],
    "last_epoch": [],
}

vit_searchspace = {
    "patch_size": [],
    "dim": [],
    "depth": [],
    "heads": [],
    "mlp_dim": [],
    "dropout": [],
    "emb_dropout": [],
}

search_space = adam_searchspace | calr_searchspace | vit_searchspace


def obj(trial):
    adm_params = {
        "lr": trial.suggest_categorical("lr", adam_searchspace["lr"]),
        "beta1": trial.suggest_categorical("beta1", adam_searchspace["beta1"]),
        "beta2": trial.suggest_categorical("beta2", adam_searchspace["beta2"]),
        "eps": trial.suggest_categorical("eps", adam_searchspace["eps"]),
    }
    adm_params["betas"] = (adm_params["beta1"], adm_params["beta2"])

    calr_params = {
        "T_max": trial.suggest_categorical("T_max", calr_searchspace["T_max"]),
        "eta_min": trial.suggest_categorical("eta_min", calr_searchspace["eta_min"]),
        "last_epoch": trial.suggest_categorical("last_epoch", calr_searchspace["last_epoch"]),
    }
    vit_params = {
        "seq_len": TIME_PERIODS,
        "num_classes": len(LABELS),
        "channels": N_FEATURES,
        "patch_size": trial.suggest_categorical(
            "patch_size", search_space["patch_size"]
        ),
        "dim": trial.suggest_categorical("dim", vit_searchspace["dim"]),
        "depth": trial.suggest_categorical("depth", vit_searchspace["depth"]),
        "heads": trial.suggest_categorical("heads", vit_searchspace["heads"]),
        "mlp_dim": trial.suggest_categorical("mlp_dim", vit_searchspace["mlp_dim"]),
        "dropout": trial.suggest_categorical("dropout", vit_searchspace["dropout"]),
        "emb_dropout": trial.suggest_categorical(
            "emb_dropout", vit_searchspace["emb_dropout"]
        ),
    }

    train_loader = DataLoader(train, batch_size=BATCH_SIZE)
    test_loader = DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **adm_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **calr_params)
    model = ViT(**vit_params).to(device)

    losslist = list()
    opbest_model = copy.deepcopy(model)
    for ep in range(MAX_EPOCH):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        ls = np.mean(losses)
        scheduler.step()
        if ep > calr_params["T_max"]:
            if min(losslist) > ls:
                opbest_model = copy.deepcopy(model)
            if is_worse(losslist, REF_SIZE, "minimize"):
                print(f"early stopping at epoch {ep} with loss {ls:.5f}")
                break
        print(f"Epoch {ep + 0:03}: | Loss: {ls:.5f}")
        losslist.append(ls)
    model = opbest_model

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

all_params = dict(study.best_params)
adam_params = {k: all_params[k] for k in adam_searchspace.keys()}
calr_params = {k: all_params[k] for k in calr_searchspace.keys()}
vit_params = {k: all_params[k] for k in vit_searchspace.keys()}
vit_params["seq_len"] = TIME_PERIODS
vit_params["num_classes"] = len(LABELS)
vit_params["channels"] = N_FEATURES


model = ViT(**vit_params).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

loss_function = nn.CrossEntropyLoss()
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
param["search_space"] = search_space
param["MAX_EPOCH"] = MAX_EPOCH
param["BATCH_SIZE"] = BATCH_SIZE

joblib.dump(param, f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}/raw/param.pkl")
