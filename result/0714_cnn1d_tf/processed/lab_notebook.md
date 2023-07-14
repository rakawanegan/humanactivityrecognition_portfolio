# Lab Notebook


## Model name
cnn1d_tf

## Start date
2023-07-14 14:37:30.115661

## End date
2023-07-14 14:41:05.525955

## Report
| precision | recall | f1-score | support |
| --- | --- | --- | --- |
|  |
| Downstairs | 0.82 | 0.89 | 0.86 | 824 |
| Jogging | 0.98 | 0.98 | 0.98 | 2832 |
| Sitting | 1.00 | 0.95 | 0.98 | 501 |
| Standing | 0.92 | 0.98 | 0.95 | 410 |
| Upstairs | 0.90 | 0.82 | 0.86 | 1025 |
| Walking | 0.98 | 0.99 | 0.98 | 3468 |
|  |
| accuracy | 0.95 | 9060 |
| macro | avg | 0.93 | 0.94 | 0.93 | 9060 |
| weighted | avg | 0.95 | 0.95 | 0.95 | 9060 |


## Feature param
- LABELS: Downstairs, Jogging, Sitting, Standing, Upstairs, Walking
- TIME_PERIODS: 80
- STEP_DISTANCE: 40
- N_FEATURES: 3
- LABEL: ActivityEncoded
- SEED: 314


## Model size
3.2e-05 MB

## Confusion_matrix
![alt](./cross-tab.png)
