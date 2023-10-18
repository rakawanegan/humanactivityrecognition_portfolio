import numpy as np
import pandas as pd
from scipy import stats
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED, n_rows:int=False):
    print("Loading data...")
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
    ohe = OneHotEncoder()
    Y_one_hot = ohe.fit_transform(y.reshape(-1, 1)).toarray()
    # print(x.shape)
    print("Data loaded.")
    return train_test_split(x, Y_one_hot, test_size=0.3, random_state=SEED) # x_train, x_test, y_train, y_test

def load_preprocessed_data(LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED, normalize=True):
    SAMPLING_RATE = 20
    def _absoulte(data):
        inputs = copy.deepcopy(data)
        outputs = copy.deepcopy(data)
        for i in range(len(inputs)):
            outputs[i] = np.sqrt(np.sum(np.square(inputs[i])))
        return outputs[:,:,0]

    def _gaussian_filter(data, sigma=1.5, k=5):
        inputs = copy.deepcopy(data)
        outputs = copy.deepcopy(data)
        w = np.exp(-np.square(np.arange(-k,k+1))/(2*sigma*sigma))
        for i in range(len(inputs)):
            for j in range(3):
                outputs[i,:,j] = np.convolve(np.pad(inputs[i,:,j], (k,k), 'edge'), w, mode='valid')
        return outputs

    def _median_filter(data, k=5):
        inputs = copy.deepcopy(data)
        outputs = copy.deepcopy(data)
        n, m, c = inputs.shape
        for i in range(n):
            for j in range(c):
                for kernel in range(k, m-k):
                    outputs[i,kernel,j] = np.median(inputs[i,kernel-k:kernel+k+1,j])
        return outputs

    def _difference(data, differential=1): # differential = 1 or SAMPLING_RATE
        inputs = copy.deepcopy(data)
        outputs = copy.deepcopy(data)
        outputs[:,0,:] = 0
        for ip in inputs:
            for i in range(1, len(ip)):
                for j in range(len(ip[i])):
                    outputs[i,j] = ip[i,j] - ip[i-1,j]
                    outputs[i,j] *= differential
        return outputs

    def _integral(data):
        inputs = copy.deepcopy(data)
        outputs = copy.deepcopy(data)
        outputs[0] = 0
        for ip in inputs:
            for i in range(1, len(ip)):
                for j in range(len(ip[i])):
                    outputs[i,j] = outputs[i-1,j] + ip[i,j]
                    outputs[i,j] /= SAMPLING_RATE
        return outputs

    def normalize(train, test):
        ch_num = train.shape[2]
        for i in range(ch_num):
            mean = np.mean(train[:,:,i])
            std = np.std(train[:,:,i])
            train[:,:,i] = (train[:,:,i] - mean) / std
            test[:,:,i] = (test[:,:,i] - mean) / std
        return train, test

    def _preprocess(data, normalize=True):
        axislist = list()
        axislist.append(data)
        axislist.append(_difference(data))
        axislist.append(_difference(_difference(data)))
        axislist.append(_gaussian_filter(data))
        axislist.append(_difference(_gaussian_filter(data)))
        axislist.append(_difference(_difference(_gaussian_filter(data))))
        axislist.append(_median_filter(data))
        axislist.append(_difference(_median_filter(data)))
        axislist.append(_difference(_difference(_median_filter(data))))
        axislist.append(_integral(data))
        axislist.append(_integral(_integral(data)))
        axislist = np.array(axislist)
        axislist = axislist.reshape(axislist.shape[1], axislist.shape[2], axislist.shape[0] * axislist.shape[3])
        axislist = np.append(axislist, _absoulte(data).reshape(axislist.shape[0], axislist.shape[1], 1), axis=2)
        return axislist

    x_train, x_test, y_train, y_test = load_data(LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED)
    print("Preprocessing data...")
    x_train = _preprocess(x_train)
    x_test = _preprocess(x_test)
    if normalize:
        print("Normalizing data...")
        x_train, x_test = normalize(x_train, x_test)
    print("Data loaded and preprocessed.")
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
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

    # load_data(LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED)
    load_preprocessed_data( LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, N_FEATURES, SEED)