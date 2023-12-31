import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
import configparser
import subprocess
import joblib
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset

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


class SeqDataset(TensorDataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]