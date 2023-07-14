import joblib
import os
import argparse
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from make_labnotebook import generate_experiment_memo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="0711_SAMPLE")
    args = parser.parse_args()
    return args

def result_process():
    args = parse_args()
    path = os.path.join("../result", args.path, "raw/")
    study = joblib.load(os.path.join(path, "study.pkl"))
    log = 0
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
    plt.savefig(f"results/processed/{args.path}.png")

    print(classification_report(y_test, y_pred, target_names=LABELS))