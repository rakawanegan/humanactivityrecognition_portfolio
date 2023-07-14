import argparse
import subprocess
import datetime
import os

from lib.result_process import result_process as rp

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error: {e.stderr}")
        return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="vit1d")
    args = parser.parse_args()
    return args


def main():
    os.mkdir(f"result/{datetime.datetime.now().strftime('%m%d')}_{args.path}")
    args = parse_args()
    main = run_command(f"python3 {args.path}.py result/{datetime.datetime.now().strftime('%m%d')}_{args.path}/raw/experiment.log")
    rp()