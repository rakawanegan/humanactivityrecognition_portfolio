import argparse
import datetime
import os

from lib.result_process import result_process as rp
from lib.local_utils import send_email, run_command


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="vit1d")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    date = datetime.datetime.now().strftime("%m%d")
    idx = 0
    while os.path.exists(f"result/{date}_{args.path}_{idx}"):
        idx += 1
    dirname = f"result/{date}_{args.path}_{idx}"
    print(f"{date}_{args.path}_{idx} running...")
    os.makedirs(dirname)
    os.makedirs(f"{dirname}/raw", exist_ok=True)
    os.makedirs(f"{dirname}/processed", exist_ok=True)
    cp_lib = run_command(f"cp -r lib/ {dirname}/raw/")
    cp_main = run_command(f"cp {args.path}.py {dirname}/raw/")
    print("setup done")
    main = run_command(
        f"python3 {args.path}.py > {dirname}/raw/experiment.log"
    )
    print("main done")
    rp(dirname)
    print("result done")
    print(f"{date}_{args.path}_{idx} done")
    send_email(f"{date}_{args.path}_{idx} done", f"{date}_{args.path}_{idx} is done")


if __name__ == "__main__":
    main()
