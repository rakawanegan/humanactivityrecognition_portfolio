import copy
import os

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


batch_size = 64
epochs = 150
ref_size = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
pct_params = {
    "hidden_ch": 15,
    "num_classes": len(LABELS),
    "input_dim": TIME_PERIODS,
    "channels": N_FEATURES,
    "dim": 128,
    "depth": 3,
    "heads": 3,
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
    train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()
)


def is_worse(losslist, ref_size, axis="minimize"):
    if axis == "minimize":
        return all(
            x > y for x, y in zip(losslist[-ref_size:], losslist[-ref_size - 1 : -1])
        )
    elif axis == "maximize":
        return all(
            x < y for x, y in zip(losslist[-ref_size:], losslist[-ref_size - 1 : -1])
        )
    else:
        raise ValueError("Invalid axis value: " + axis)


losslist = list()
p_models = list()

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
    p_models.append(copy.deepcopy(model))
    if ep > ref_size and is_worse(losslist, ref_size, "minimize"):
        print(f"early stopping at epoch {ep} with loss {ls:.5f}")
        model = p_models[0]
        break
    if ep > ref_size:
        del p_models[0]  # del oldest model
    print(f"Epoch {ep + 0:03}: | Loss: {ls:.5f}")
    losslist.append(ls)


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

# Creates a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Transform to df for easier plotting
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
plt.savefig("results/convbackbone_transformer_predict.png")

print(classification_report(y_test, y_pred, target_names=LABELS))

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])
