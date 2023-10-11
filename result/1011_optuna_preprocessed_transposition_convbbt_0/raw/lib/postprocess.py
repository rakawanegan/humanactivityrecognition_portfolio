import joblib
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import subprocess
import optuna


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
    if "optuna" in dirname:
        study = optuna.load_study(study_name=dirname, storage=f"sqlite:///{dirname}/raw/optuna.db")

        importtance_fig = optuna.visualization.matplotlib.plot_param_importances(study)
        importtance_fig.figure.savefig(os.path.join(dirname, "processed/assets",'optimization_importance.png'), bbox_inches='tight')

        histry_fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        histry_fig.figure.savefig(os.path.join(dirname, "processed/assets",'optimization_history.png'), bbox_inches='tight')

        best_trial = study.best_trial
        param_keys = list(best_trial.params.keys())
        return param_keys
    else:
        return None

def create_experiment_memo(dir, content):
    file_name = os.path.join(dir, "lab_notebook.md")

    with open(file_name, "w") as f:
        f.write(content)

def generate_experiment_memo(dir:str, date, experiment_info:dict):
    memo_content = f"# Lab Notebook\n\n"

    for key, value in experiment_info.items():
        memo_content += f"\n## {key}\n{value}\n"

    create_experiment_memo(dir, memo_content)

def post_process(dirname):
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
    time_diff = (param["end_date"] - param["start_date"]).total_seconds()
    execution_time = f"{int(time_diff // 3600)} hours {int((time_diff % 3600) // 60)} minutes {int(time_diff % 60)} seconds"
    hparam_picdirs = [f"![]({os.path.join('./assets',f'optimization_{name}.png')})" for name in ["history", "importance"]] if hparam_names is not None else None
    content = {
        "Model name": param.pop("MODEL_NAME"),
        "Start date": param.pop("start_date"),
        "End date": param.pop("end_date"),
        "Execution time": execution_time,
        "Report": convert_to_markdown_table(report),
        "Optuna search space": '\n'.join([f'- {key}: {"".join(str(value)) if isinstance(value, list) else str(value)}' for key, value in search_space.items()]) if search_space is not None else None,
        "Feature param": '\n'.join([f'- {key}: {", ".join(value) if isinstance(value, list) else str(value)}' for key, value in param.items()]),
        "Model size": run_command(f'stat {os.path.join(path, "model.pt")} | grep Size').split('\t')[0] + " B",
        "Confusion_matrix": "![alt](./assets/cross-tab.png)",
        "Loss curve": "![alt](./assets/loss.png)",
        "optuna search plots": '\n'.join(hparam_picdirs) if hparam_picdirs is not None else None,
    }

    generate_experiment_memo(f"{dirname}/processed/", content["Start date"], content)
