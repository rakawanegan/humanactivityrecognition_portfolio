import copy
import datetime
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from torch.utils.data import DataLoader

from lib.model import PreConvTransformer
from lib.preprocess import load_data
from lib.local_utils import send_email, is_worse, SeqDataset


dirname = "result/input_diff"
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

# Hyperparameters
MAX_EPOCH = 200
BATCH_SIZE = 128
REF_SIZE = 5
SAMPLING_RATE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
print("Device: ", device)
print("Max Epochs: ", MAX_EPOCH)
print("Early Stopping Reference Size: ", REF_SIZE)

def vanilla(data):
    return data

def absoulte(data):
    inputs = copy.deepcopy(data)
    outputs = copy.deepcopy(data)
    for i in range(len(inputs)):
        outputs[i] = np.sqrt(np.sum(np.square(inputs[i])))
    return outputs[:,:,0]

def gaussian_filter(data, sigma=1.5, k=5):
    inputs = copy.deepcopy(data)
    outputs = copy.deepcopy(data)
    w = np.exp(-np.square(np.arange(-k,k+1))/(2*sigma*sigma))
    for i in range(len(inputs)):
        for j in range(3):
            outputs[i,:,j] = np.convolve(np.pad(inputs[i,:,j], (k,k), 'edge'), w, mode='valid')
    return outputs

def median_filter(data, k=5):
    inputs = copy.deepcopy(data)
    outputs = copy.deepcopy(data)
    n, m, c = inputs.shape
    for i in range(n):
        for j in range(c):
            for kernel in range(k, m-k):
                outputs[i,kernel,j] = np.median(inputs[i,kernel-k:kernel+k+1,j])
    return outputs

def difference(data, differential=1): # differential = 1 or SAMPLING_RATE
    inputs = copy.deepcopy(data)
    outputs = copy.deepcopy(data)
    outputs[:,0,:] = 0
    for ip in inputs:
        for i in range(1, len(ip)):
            for j in range(len(ip[i])):
                outputs[i,j] = ip[i,j] - ip[i-1,j]
                outputs[i,j] *= differential
    return outputs

def integral(data):
    inputs = copy.deepcopy(data)
    outputs = copy.deepcopy(data)
    outputs[0] = 0
    for ip in inputs:
        for i in range(1, len(ip)):
            for j in range(len(ip[i])):
                outputs[i,j] = outputs[i-1,j] + ip[i,j]
                outputs[i,j] /= SAMPLING_RATE
    return outputs

def normalize(train, test):
    ch_num = train.shape[2]
    for i in range(ch_num):
        mean = np.mean(train[:,:,i])
        std = np.std(train[:,:,i])
        train[:,:,i] = (train[:,:,i] - mean) / std
        test[:,:,i] = (test[:,:,i] - mean) / std
    return train, test


def run(preprocessor, name):
    print("------------------")
    x_train, x_test, y_train, y_test = load_data(
        LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED
    )
    # preprocess data
    x_train = preprocessor.func(x_train)
    x_test = preprocessor.func(x_test)

    # axis-wise input
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
    print(type(x_train))


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
        # scheduler.step()
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
            # send_email(
            #     "Training started",
            #     f"Training started at {ep_start.strftime('%Y-%m-%d %H:%M:%S')}\nEstimated time: {ep_delta * MAX_EPOCH}\nEstimated finish: {datetime.datetime.now() + ep_delta * MAX_EPOCH}"
            # )
    model = best_model

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

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        linewidths=0.5,
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 14},
        square=True,
    )
    plt.title("Kernel \nAccuracy:{0:.3f}".format(accuracy_score(y_test, y_pred)))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{dirname}/{name}-cross-tab.png")
    report = classification_report(y_test, y_pred, target_names=LABELS)
    print(report)
    print("------------------")


# we can change this code to preprocess data

def vannilapreprocessor(data):
        data = vanilla(data)
        return data

def gaussianpreprocessor(data):
    data = gaussian_filter(data)
    return data

def medianpreprocessor(data):
    data = median_filter(data)
    return data

def differencepreprocessor(data):
    data = difference(data)
    return data

def differencedifferencepreprocessor(data):
    data = difference(data, 1)
    data = difference(data, 1)
    return data

def integralpreprocessor(data):
    data = integral(data)
    return data

def integralintegralpreprocessor(data):
    data = integral(data)
    data = integral(data)
    return data

def gaussiandifferencepreprocessor(data):
    data = gaussian_filter(data)
    data = difference(data)
    return data

def gaussiandifferencedifferencepreprocessor(data):
    data = gaussian_filter(data)
    data = difference(data)
    data = difference(data)
    return data

def mediandifferencepreprocessor(data):
    data = median_filter(data)
    data = difference(data)
    return data

def mediandifferencedifferencepreprocessor(data):
    data = median_filter(data)
    data = difference(data)
    data = difference(data)
    return data

vanilla = vannilapreprocessor()
gaussian = gaussianpreprocessor()
median = medianpreprocessor()
difference = differencepreprocessor()
differencedifference = differencedifferencepreprocessor()
integral = integralpreprocessor()
integralintegral = integralintegralpreprocessor()
gaussiandifference = gaussiandifferencepreprocessor()
gaussiandifferencedifference = gaussiandifferencedifferencepreprocessor()
mediandifference = mediandifferencepreprocessor()
mediandifferencedifference = mediandifferencedifferencepreprocessor()

preprocessors = {
    "vanilla": vanilla,
    "gaussian": gaussian,
    "median": median,
    "difference": difference,
    "differencedifference": differencedifference,
    "integral": integral,
    "integralintegral": integralintegral,
    "gaussiandifference": gaussiandifference,
    "gaussiandifferencedifference": gaussiandifferencedifference,
    "mediandifference": mediandifference,
    "mediandifferencedifference": mediandifferencedifference,
}

for name, preprocessor in preprocessors.items():
    run(preprocessor, name)
