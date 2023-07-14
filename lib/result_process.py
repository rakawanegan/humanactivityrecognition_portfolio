import joblib
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def create_experiment_memo(dir, content):
    file_name = os.path.join(dir, "lab_notebook.md")

    with open(file_name, "w") as f:
        f.write(content)

def generate_experiment_memo(dir:str, date, experiment_info:dict):
    memo_content = f"# 実験メモ\n\n## 日付\n{date}\n"

    for key, value in experiment_info.items():
        memo_content += f"\n## {key}\n{value}\n"

    create_experiment_memo(dir, memo_content)

def result_process(name):
    path = os.path.join("result", name, "raw/")
    model = joblib.load(os.path.join(path, "model.pkl"))
    param = joblib.load(os.path.join(path, "param.pkl"))
    if os.path.exists(os.path.join(path, "study.pkl")):
        study = joblib.load(os.path.join(path, "study.pkl"))
        best_trial = study.best_trial
        search_space = param["search_space"]
    else:
        study = None
        best_trial = None
        search_space = None
    # log = pd.read_csv(os.path.join(path, "experiment.log"), index_col=0)
    y = pd.read_csv(os.path.join(path, "predict.csv"), index_col=0)

    # Creates a confusion matrix
    y_pred = y["predict"]
    y_test = y["true"]
    cm = confusion_matrix(y_test, y_pred)

    # Transform to df for easier plotting
    LABELS = param["LABELS"]
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
    plt.savefig(f"result/{name}/processed/cross-tab.png")
    report = classification_report(y_test, y_pred, target_names=LABELS)

    content = dict()
    content["MODEL NAME"] = param.pop("MODEL_NAME")
    content["start_date"] = param.pop("start_date")
    content["end_date"] = param.pop("end_date")
    content["report"] = report
    if study is not None:
        content["optuna's param"] = best_trial
        content["optuna search space"] = param.pop("search_space")
    content["feature param"] = param
    content["model size"] = f"{model.__sizeof__()//1e+9} GB"
    content["confusion_matrix"] = "![alt](./cross-tab.png)"

    generate_experiment_memo(f"result/{name}/processed/", content["start_date"], content)