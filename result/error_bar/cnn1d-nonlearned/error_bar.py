from lib.preprocess import load_data
import datetime
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.models import Sequential


MODEL_NAME = "cnn1d-nonlearned"
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

model = Sequential()
model.add(
    Conv1D(160, 12, input_shape=(x_train.shape[1], x_train.shape[2]), activation="relu")
)
model.add(Conv1D(128, 10, activation="relu"))
model.add(Conv1D(96, 8, activation="relu"))
model.add(Conv1D(64, 6, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))

print(model.summary())

model.layers[-1].activation = None

model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

y_pred = model.predict(x_test)
y_test = y_test.argmax(axis=-1)
y_pred = pd.DataFrame(y_pred)
y_pred["true"] = y_test

y_pred.to_csv(f"{dirname}/predict.csv")
