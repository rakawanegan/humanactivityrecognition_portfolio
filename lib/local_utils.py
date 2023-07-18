import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
import configparser
import subprocess
import joblib
import matplotlib.pyplot as plt
import os


def run_command(command):
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error: {e.stderr}")
        return None


def read_config(path,name):
    config = configparser.ConfigParser()
    config.read(path)
    config_dict = dict(config[name])
    type_dict = {"int":int,"float":float,"str":str}
    for key,value in config_dict.items():
        type_, value = value.split(" ")
        config_dict[key] = type_dict[type_](value)
    return config_dict


def send_email(subject:str, body:str) -> bool:
    dic = read_config("./config.ini","gmail")
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.starttls()
    smtpobj.login(dic["adress"], dic["password"])

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = dic["adress"]
    msg['To'] = dic["to"]
    msg['Date'] = formatdate()

    smtpobj.send_message(msg)
    smtpobj.close()
    return True


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


def is_worse(losslist, REF_SIZE, axis="minimize"):
    if axis == "minimize":
        return all(
            x > y for x, y in zip(losslist[-REF_SIZE:], losslist[-REF_SIZE - 1 : -1])
        )
    elif axis == "maximize":
        return all(
            x < y for x, y in zip(losslist[-REF_SIZE:], losslist[-REF_SIZE - 1 : -1])
        )
    else:
        raise ValueError("Invalid axis value: " + axis)