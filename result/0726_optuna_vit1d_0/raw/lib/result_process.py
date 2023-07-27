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

def create_study(dirname: str) -> list:
    if os.path.exists(os.path.join(dirname, "raw", "study.pkl")):
        return None
    study = joblib.load(os.path.join(dirname, "raw", "study.pkl"))
    best_trial = study.best_trial
    param_keys = list(best_trial.params.keys())
    param_values = {key: list() for key in param_keys}
    scores = list()

    for trial in study.trials:
        for key in param_keys:
            param_values[key].append(trial.params[key])
        scores.append(trial.value)

    # show scatter
    for key in param_keys:
        plt.figure()
        plt.scatter(param_values[key], scores, s=8)
        plt.xlabel(key)
        plt.ylabel("score")
        plt.savefig(os.path.join(dirname, "processed/assets", f"{key}.png"))
        plt.close()
    return param_keys

def create_experiment_memo(dir, content):
    file_name = os.path.join(dir, "lab_notebook.md")

    with open(file_name, "w") as f:
        f.write(content)

def generate_experiment_memo(dir:str, date, experiment_info:dict):
    memo_content = f"# Lab Notebook\n\n"

    for key, value in experiment_info.items():
        memo_content += f"\n## {key}\n{value}\n"

    create_experiment_memo(dir, memo_content)

def result_process(dirname):
    path = os.path.join(dirname, "raw/")
    param = joblib.load(os.path.join(path, "param.pkl"))
    hparam_names = create_study(dirname)
    search_space = param.pop("search_space") if hparam_names is not None else None
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
    plt.savefig(f"{dirname}/processed/assets/cross-tab.png")
    report = classification_report(y_test, y_pred, target_names=LABELS)
    time_diff = (param["end date"] - param["start date"]).total_seconds()
    execution_time = f"{int(time_diff // 3600)} hours {int((time_diff % 3600) // 60)} minutes {int(time_diff % 60)} seconds"
    hparam_picdirs = {name: f"./assets/{name}.png" for name in hparam_names} if hparam_names is not None else None
    content = {
        "Model name": param.pop("MODEL NAME"),
        "Start date": param.pop("start date"),
        "End date": param.pop("end date"),
        "Execution time": execution_time,
        "Report": convert_to_markdown_table(report),
        "Optuna search space": '\n'.join([f'- {key}: {"".join(str(value)) if isinstance(value, list) else str(value)}' for key, value in search_space.items()]) if search_space is not None else None,
        "Adam param": '\n'.join([f'- {key}: {", ".join(value) if isinstance(value, list) else str(value)}' for key, value in param.pop("Adam params").items()]),
        "CosineAnnealingLRScheduler param": '\n'.join([f'- {key}: {", ".join(value) if isinstance(value, list) else str(value)}' for key, value in param.pop("CosineAnnealingLRScheduler params").items()]),
        "Model param": '\n'.join([f'- {key}: {", ".join(value) if isinstance(value, list) else str(value)}' for key, value in param.pop("Model params").items()]),
        "Feature param": '\n'.join([f'- {key}: {", ".join(value) if isinstance(value, list) else str(value)}' for key, value in param.items()]),
        "Model size": run_command(f'stat {os.path.join(path, "model.pkl")} | grep Size').split('\t')[0] + " B",
        "Confusion_matrix": "![alt](./assets/cross-tab.png)",
        "Loss curve": "![alt](./assets/loss.png)",
        "Hyper parameter plots": '\n'.join([f'### {key}\n: {", ".join(value) if isinstance(value, list) else str(value)}' for key, value in hparam_picdirs.items()]) if hparam_picdirs is not None else None,
    }

    generate_experiment_memo(f"{dirname}/processed/", content["Start date"], content)
