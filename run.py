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
    args = parse_args()
    date = datetime.datetime.now().strftime('%m%d')
    os.makedirs(f"result/{date}_{args.path}")
    os.makedirs(f"result/{date}_{args.path}/raw", exist_ok=True)
    os.makedirs(f"result/{date}_{args.path}/processed", exist_ok=True)
    cp = run_command(f"cp {args.path}.py result/{date}_{args.path}/raw/")
    main = run_command(f"python3 {args.path}.py > result/{datetime.datetime.now().strftime('%m%d')}_{args.path}/raw/experiment.log")
    rp(f"{date}_{args.path}")


if __name__ == "__main__":
    main()