import joblib
import os
import argparse
import datetime
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from make_labnotebook import generate_experiment_memo

def result_process(name):
    name = f"{datetime.datetime.now().strftime('%m%d')}_{name}"
    path = os.path.join("result", name, "raw/")
    study = joblib.load(os.path.join(path, "study.pkl"))
    log = pd.read_csv(os.path.join(path, "experiment.log"), index_col=0)
    y = pd.read_csv(os.path.join(path, "predict.csv"), index_col=0)
    # Creates a confusion matrix
    y_pred = y["predict"]
    y_test = y["true"]
    cm = confusion_matrix(y_test, y_pred)

    # Transform to df for easier plotting
    LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
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
    plt.savefig(f"results/{name}/processed/cross-tab.png")

    print(classification_report(y_test, y_pred, target_names=LABELS))