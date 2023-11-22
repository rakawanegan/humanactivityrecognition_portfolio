import datetime

import numpy as np
import pandas as pd

from lib.preprocess import load_data
import datetime
import os



MODEL_NAME = "random_simuration"
print("MODEL_NAME: ", MODEL_NAME)
start_date = datetime.datetime.now()
print("Start time: ", start_date)
# Same labels will be reused throughout the program
LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40
# x, y, z acceleration as features
N_FEATURES = 3
# Define column name of the label vector
LABEL = "ActivityEncoded"
# set random seed
SEED = 314

dirname = f"result/error_bar/{MODEL_NAME}"

x_train, x_test, y_train, y_test = load_data(
    LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED
)

np.random.seed(SEED)

y_test = y_test.argmax(axis=-1)

y_pred = list()
for test in y_test:
    y_pred.append(np.random.randint(0, 6))
random_df = pd.DataFrame(y_pred, columns=["random_pred"])

y_pred = list()
for test in y_test:
    if test == 2 or test == 3:
        y_pred.append(np.random.choice([2, 3]))
    else:
        y_pred.append(np.random.choice([0, 1, 4, 5]))
semi_random_df = pd.DataFrame(y_pred, columns=["semi_random_pred"])


y_df = pd.DataFrame()
y_df["random"] = random_df["random_pred"]
y_df["semi_random"] = semi_random_df["semi_random_pred"]
y_df["true"] = y_test

y_df.to_csv(f"{dirname}/predict.csv")
