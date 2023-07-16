import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, default="formatted")
    parser.add_argument("--to", type=str, default="master")
    args = parser.parse_args()
    return args


def run_command(command):
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error: {e.stderr}")
        return None


def main():
    args = parse_args()
    pip = run_command("pip install isort black")
    print(pip)
    isort = run_command("isort ../*.py")
    print(isort)
    black = run_command("black ../*.py")
    print(black)
    git = run_command(f"git add ../.;git commit -m {args.m};git push origin {args.to}")
    print(git)


if __name__ == "__main__":
    main()
