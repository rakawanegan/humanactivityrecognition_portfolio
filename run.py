import argparse
import datetime
import os

from lib.postprocess import post_process as pp
from lib.local_utils import send_email, run_command


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="vit1d")
    parser.add_argument("--git", type=str, default="push")
    args = parser.parse_args()
    return args


def main():
    fetch = run_command("python3 setup.py")
    print(fetch)
    # pip = run_command("pip3 install -r requirements.txt")
    args = parse_args()
    date = datetime.datetime.now().strftime("%m%d")
    idx = 0
    while os.path.exists(f"result/{date}_{args.path}_{idx}"):
        idx += 1
    dirname = f"result/{date}_{args.path}_{idx}"
    print(f"{date}_{args.path}_{idx} running...")
    os.makedirs(dirname)
    os.makedirs(f"{dirname}/raw")
    os.makedirs(f"{dirname}/processed")
    os.makedirs(f"{dirname}/processed/assets")
    cp_lib = run_command(f"cp -r lib/ {dirname}/raw/")
    cp_main = run_command(f"cp {args.path}.py {dirname}/raw/")
    print("setup done")
    main = run_command(
        f"python3 main/{args.path}.py > {dirname}/raw/experiment.log"
    )
    print("main done")
    pp(dirname)
    print("result done")
    print(f"{date}_{args.path}_{idx} done")
    # send_email(f"{date}_{args.path}_{idx} done", f"{date}_{args.path}_{idx} is done")
    if args.git == "push":
        gitpull = run_command(f"git pull")
        gita = run_command(f"git add .")
        print("git add")
        message = run_command(f"tail -5 {dirname}/raw/experiment.log")
        # gitc = run_command(f"git commit -m {message}")
        gitc = run_command(f'git commit -m "add result"')
        print("git commit")
        print(message)
        gitpush = run_command(f"git push")
        print("git push")
        print("git done")
    print("done")


if __name__ == "__main__":
    main()
