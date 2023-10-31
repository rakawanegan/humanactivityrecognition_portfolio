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


MODEL_NAME = "convbbt-nonlearned"
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

dirname = f"result/error_bar/{MODEL_NAME}/"

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
    "hidden_ch": 25,
    "hidden_dim": 1024,
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

model.eval()
y_pred = list()
for batch in test_loader:
    x, _ = batch
    x = x.to(device)
    optimizer.zero_grad()
    output = model(x)
    y_pred.append(output.detach().cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0)
y_pred = pd.DataFrame(y_pred)
y_test = y_test.argmax(axis=-1)
y_pred["true"] = y_test

y_pred.to_csv(f"{dirname}/predict.csv")
