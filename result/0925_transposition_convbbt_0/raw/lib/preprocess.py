import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_data(LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED, n_rows:int=False):
    def read_data(file_path):
        column_names = [
            "user-id",
            "activity",
            "timestamp",
            "x-axis",
            "y-axis",
            "z-axis",
        ]
        df = pd.read_csv(file_path, header=None, names=column_names)
        df.dropna(axis=0, how="any", inplace=True)
        return df

    # Load data set containing all the data from csv
    df = read_data("./data/WISDM_ar_v1.1.csv")

    # Transform the labels from String to Integer via LabelEncoder
    le = LabelEncoder()
    # Add a new column to the existing DataFrame with the encoded values
    df[LABEL] = le.fit_transform(df["activity"].values.ravel())

    def create_segments_and_labels(df, time_steps, step, label_name):
        # Number of steps to advance in each iteration (for me, it should always
        # be equal to the time_steps in order to have no overlap between segments)
        # step = time_steps
        segments = []
        labels = []
        for i in range(0, len(df) - time_steps, step):
            xs = df["x-axis"].values[i : i + time_steps]
            ys = df["y-axis"].values[i : i + time_steps]
            zs = df["z-axis"].values[i : i + time_steps]
            # Retrieve the most often used label in this segment
            label = stats.mode(df[label_name][i : i + time_steps], keepdims=True)[0][0]
            segments.append([xs, ys, zs])
            labels.append(label)

        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(
            -1, time_steps, N_FEATURES
        )
        labels = np.asarray(labels)

        return reshaped_segments, labels

    x, y = create_segments_and_labels(df, TIME_PERIODS, STEP_DISTANCE, LABEL)
    if n_rows:
        x, y = x[:n_rows], y[:n_rows]

    # print(f"{LABELS} -> {le.transform(LABELS)}")
    ohe = OneHotEncoder()
    Y_one_hot = ohe.fit_transform(y.reshape(-1, 1)).toarray()

    return train_test_split(x, Y_one_hot, test_size=0.3, random_state=SEED) # x_train, x_test, y_train, y_test
