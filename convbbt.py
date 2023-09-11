import copy
import datetime
import os

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from lib.model import PreConvTransformer
from lib.preprocess import load_data
from lib.local_utils import send_email, is_worse, SeqDataset


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

diridx = 0
while os.path.exists(f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}_{diridx}"):
    diridx += 1
diridx -= 1
dirname = f"result/{start_date.strftime('%m%d')}_{MODEL_NAME}_{diridx}"

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

adm_params = {
    "lr": 0.0001,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0,
    "amsgrad": False,
}

calr_params = {
    "T_max": 150,
    "eta_min": 1e-05,
    "last_epoch": -1,
    "verbose": False,
}

pct_params = {
    "num_classes": len(LABELS),
    "input_dim": TIME_PERIODS,
    "channels": N_FEATURES,
    "hidden_ch": 15,
    "hidden_dim": 128,
    "depth": 5,
    "heads": 8,
    "mlp_dim": 1024,
    "dropout": 0.01,
    "emb_dropout": 0.01,
}
model = PreConvTransformer(**pct_params).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), **adm_params)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **calr_params)


train = SeqDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
test = SeqDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()
)

losslist = list()

ep_start = datetime.datetime.now()
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
        if min(losslist) > ls:
            best_model = copy.deepcopy(model)
        if is_worse(losslist, REF_SIZE, "minimize"):
            print(f"early stopping at epoch {ep} with loss {ls:.5f}")
            break
    print(f"Epoch {ep + 0:03}: | Loss: {ls:.5f}")
    losslist.append(ls)
    if ep == 1:
        ep_delta = datetime.datetime.now() - ep_start
        print(f"Estimated time: {ep_delta * MAX_EPOCH}")
        print(f"Estimated finish: {datetime.datetime.now() + ep_delta * MAX_EPOCH}")
        send_email(
            "Training started",
            f"Training started at {ep_start.strftime('%Y-%m-%d %H:%M:%S')}\nEstimated time: {ep_delta * MAX_EPOCH}\nEstimated finish: {datetime.datetime.now() + ep_delta * MAX_EPOCH}"
        )
model = best_model

plt.plot(losslist)
plt.title("Loss curve")
plt.xlabel("Epoch")
plt.ylabel("Loss mean")
plt.savefig(f"{dirname}/processed/assets/loss.png")

model.eval()
torch.save(
    model.to('cpu').state_dict(),
    f"{dirname}/raw/model.pt"
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

torch.save(y_test, f"{dirname}/raw/y_test.tsr")
torch.save(x_test, f"{dirname}/raw/x_test.tsr")

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
param["MODEL NAME"] = MODEL_NAME
param["start date"] = start_date
param["end date"] = end_date
param["LABELS"] = LABELS
param["TIME PERIODS"] = TIME_PERIODS
param["STEP DISTANCE"] = STEP_DISTANCE
param["N FEATURES"] = N_FEATURES
param["LABEL"] = LABEL
param["SEED"] = SEED
param["MAX EPOCH"] = MAX_EPOCH
param["BATCH SIZE"] = BATCH_SIZE
param["REF SIZE"] = REF_SIZE
param["Adam params"] = adm_params
param["CosineAnnealingLRScheduler params"] = calr_params
param["Model params"] = pct_params

joblib.dump(param, f"{dirname}/raw/param.pkl")
