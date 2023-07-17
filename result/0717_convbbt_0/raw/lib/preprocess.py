import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_data(LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES):
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

    def show_basic_dataframe_info(dataframe):
        # Shape and how many rows and columns
        print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
        print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))

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
        users = []
        for i in range(0, len(df) - time_steps, step):
            xs = df["x-axis"].values[i : i + time_steps]
            ys = df["y-axis"].values[i : i + time_steps]
            zs = df["z-axis"].values[i : i + time_steps]
            # Retrieve the most often used label in this segment
            label = stats.mode(df[label_name][i : i + time_steps])[0][0]
            segments.append([xs, ys, zs])
            labels.append(label)
            user = stats.mode(df["user-id"][i : i + time_steps])[0][0]
            users.append(user - 1)

        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(
            -1, time_steps, N_FEATURES
        )
        labels = np.asarray(labels)
        users = np.asarray(users)

        return reshaped_segments, labels, users

    x, y, y_users = create_segments_and_labels(df, TIME_PERIODS, STEP_DISTANCE, LABEL)

    # print(f"{LABELS} -> {le.transform(LABELS)}")

    Y_one_hot = to_categorical(y)
    Y_one_hot_users = to_categorical(y_users)

    return train_test_split(x, Y_one_hot, test_size=0.33, random_state=42)
