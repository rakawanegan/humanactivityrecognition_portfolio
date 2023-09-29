import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
import configparser
import subprocess
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import string

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



def plot_activity(activity, data, savedir="processed/assets/miss_activity_plots"):

    def _plot_axis(ax, x, y, title):
        ax.plot(x, y, 'r')
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
        ax.set_xlim([min(x), max(x)])
        ax.grid(True)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
        figsize=(15, 10),
        sharex=True)
    time_steps = np.linspace(0, 4, num=data.shape[0])
    _plot_axis(ax0, time_steps, data[:,0], 'X-Axis')
    _plot_axis(ax1, time_steps, data[:,1], 'Y-Axis')
    _plot_axis(ax2, time_steps, data[:,2], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    # os.makedirs(savedir, exist_ok=True)
    # plt.savefig(f"{savedir}/{activity}-{''.join([random.choice(string.ascii_letters + string.digits) for i in range(7)])}.png")
    # plt.show()

def n_plot_activity(activity, data, savedir="processed/assets/miss_activity_plots"):

    def _plot_axis(ax, x, y, title):
        ax.plot(x, y, 'r')
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
        ax.set_xlim([min(x), max(x)])
        ax.grid(True)

    if len(data.shape) == 1:
        ch = 1
        fig, axs = plt.subplots(
            nrows=ch,
            figsize=(15, 5),
            sharex=True)
        time_steps = np.linspace(0, 4, num=data.shape[0])
        _plot_axis(axs, time_steps, data, '1-Axis')
        plt.subplots_adjust(hspace=0.2)
        fig.suptitle(activity)
        plt.subplots_adjust(top=0.90)
        # os.makedirs(savedir, exist_ok=True)
        # plt.savefig(f"{savedir}/{activity}-{''.join([random.choice(string.ascii_letters + string.digits) for i in range(7)])}.png")
        # plt.show()
    else:
        ch = data.shape[1]
        fig, axs = plt.subplots(
            nrows=ch,
            sharex=True)
        time_steps = np.linspace(0, 4, num=data.shape[0])
        for i in range(ch):
            _plot_axis(axs[i], time_steps, data[:,i], f'{i+1}-Axis')
        plt.subplots_adjust(hspace=0.2)
        fig.suptitle(activity)
        plt.subplots_adjust(top=0.90)
        # os.makedirs(savedir, exist_ok=True)
        # plt.savefig(f"{savedir}/{activity}-{''.join([random.choice(string.ascii_letters + string.digits) for i in range(7)])}.png")
        # plt.show()