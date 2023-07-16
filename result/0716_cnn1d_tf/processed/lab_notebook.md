# Lab Notebook


## Model name
cnn1d_tf

## Start date
2023-07-16 19:54:16.401622

## End date
2023-07-16 19:57:51.617205

## Execution time
0 hours 3 minutes 35 seconds

## Report
| | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
|  |
| Downstairs | 0.85 | 0.85 | 0.85 | 824 |
| Jogging | 0.97 | 0.98 | 0.98 | 2832 |
| Sitting | 0.99 | 0.97 | 0.98 | 501 |
| Standing | 0.98 | 0.97 | 0.97 | 410 |
| Upstairs | 0.89 | 0.82 | 0.85 | 1025 |
| Walking | 0.97 | 0.99 | 0.98 | 3468 |
|  |
|  accuracy || | 0.95 | 9060 |
| macro | avg | 0.94 | 0.93 | 0.93 | 9060 |
| weighted | avg | 0.95 | 0.95 | 0.95 | 9060 |


## Optuna search space
None

## Feature param
- LABELS: Downstairs, Jogging, Sitting, Standing, Upstairs, Walking
- TIME_PERIODS: 80
- STEP_DISTANCE: 40
- N_FEATURES: 3
- LABEL: ActivityEncoded
- SEED: 314

## Model size
Size: 2807850    B

## Confusion_matrix
![alt](./cross-tab.png)
