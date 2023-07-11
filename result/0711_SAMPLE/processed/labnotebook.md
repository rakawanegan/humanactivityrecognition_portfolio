# cnn1d_tf

## start date
07/11 21:00

## execution time
06:58:08

## model name
1D-CNN

## hyper parameter
[optuna's param or else]

## params state
total params：346,566
| Layer (type)              | Output Shape | Param #  |
|---------------------------|--------------|----------|
| conv1d_1 (Conv1D)         | (None, 69, 160)  | 5,920  |
| conv1d_2 (Conv1D)         | (None, 60, 128)  | 204,928  |
| conv1d_3 (Conv1D)         | (None, 53, 96)   | 98,400  |
| conv1d_4 (Conv1D)         | (None, 48, 64)   | 36,928  |
| global_max_pooling1d_1 (GlobalMaxPooling1D) | (None, 64) | 0  |
| dropout_1 (Dropout)       | (None, 64)  | 0  |
| dense_1 (Dense)           | (None, 6)   | 390  |

## result
|   tf-1DCNN  | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Downstairs  |   0.86    |  0.83  |   0.85   |   824   |
| Jogging     |   0.97    |  0.98  |   0.98   |  2832   |
| Sitting     |   0.99    |  0.97  |   0.98   |   501   |
| Standing    |   0.97    |  0.98  |   0.97   |   410   |
| Upstairs    |   0.86    |  0.86  |   0.86   |  1025   |
| Walking     |   0.98    |  0.98  |   0.98   |  3468   |
|             |           |        |          |         |
| Accuracy    |           |        |   0.95   |  9060   |
| Macro Avg   |   0.94    |  0.93  |   0.94   |  9060   |
| Weighted Avg|   0.95    |  0.95  |   0.95   |  9060   |

![alt](./cross-tab.png)

## code
[code escape #]