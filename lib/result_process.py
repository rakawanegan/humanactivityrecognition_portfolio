import joblib
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import subprocess


def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error: {e.stderr}")
        return None

def convert_to_markdown_table(input_string):
    lines = input_string.strip().split('\n')
    headers = lines[0].split()
    data = [line.split() for line in lines[1:]]

    # Generate the table header
    table = '| | ' + ' | '.join(headers) + ' |\n'
    table += '| ' + ' | '.join(['---'] * (len(headers) + 1)) + ' |\n'

    # Generate the table rows
    for row in data:
        row = list(map(lambda x: x.replace("accuracy", " accuracy ||"), row))
        table += '| ' + ' | '.join(row) + ' |\n'

    return table


def create_experiment_memo(dir, content):
    file_name = os.path.join(dir, "lab_notebook.md")

    with open(file_name, "w") as f:
        f.write(content)

def generate_experiment_memo(dir:str, date, experiment_info:dict):
    memo_content = f"# Lab Notebook\n\n"

    for key, value in experiment_info.items():
        memo_content += f"\n## {key}\n{value}\n"

    create_experiment_memo(dir, memo_content)

def result_process(name):
    path = os.path.join("result", name, "raw/")
    # model = joblib.load(os.path.join(path, "model.pkl"))
    param = joblib.load(os.path.join(path, "param.pkl"))
    study = joblib.load(os.path.join(path, "study.pkl")) if os.path.exists(os.path.join(path, "study.pkl")) else None
    # best_trial = study.best_trial if study is not None else None
    search_space = param.pop("search_space") if study is not None else None
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
    time_diff = (param["end_date"] - param["start_date"]).total_seconds()
    execution_time = f"{int(time_diff // 3600)} hours {int((time_diff % 3600) // 60)} minutes {int(time_diff % 60)} seconds"
    content = {
        "Model name": param.pop("MODEL_NAME"),
        "Start date": param.pop("start_date"),
        "End date": param.pop("end_date"),
        "Execution time": execution_time,
        "Report": convert_to_markdown_table(report),
        "Optuna search space": '\n'.join([f'- {key}: {"".join(str(value)) if isinstance(value, list) else str(value)}' for key, value in search_space.items()]) if study is not None else None,
        "Feature param": '\n'.join([f'- {key}: {", ".join(value) if isinstance(value, list) else str(value)}' for key, value in param.items()]),
        "Model size": run_command(f'stat {os.path.join(path, "model.pkl")} | grep Size').split('\t')[0] + " B",
        "Confusion_matrix": "![alt](./cross-tab.png)"
    }

    generate_experiment_memo(f"result/{name}/processed/", content["Start date"], content)
