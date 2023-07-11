import joblib
import os
import argparse

from ../lib.make_labnotebook import generate_experiment_memo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="0711_SAMPLE")
    args = parser.parse_args()
    return args

args = parse_args()
path = os.path.join("../result", args.path, "raw/")
study = joblib.load(os.path.join(path, "study.pkl"))
log = 0