# Lab Notebook

## date
2023-07-14 14:17:09.648923

## Model name
cnn1d_tf

## Start date
2023-07-14 14:17:09.648923

## End date
2023-07-14 14:20:44.853639

## Report
| precision | recall | f1-score | support |
| --- | --- | --- | --- |
|  |
| Downstairs | 0.86 | 0.87 | 0.87 | 824 |
| Jogging | 0.97 | 0.98 | 0.98 | 2832 |
| Sitting | 1.00 | 0.96 | 0.98 | 501 |
| Standing | 0.97 | 0.97 | 0.97 | 410 |
| Upstairs | 0.88 | 0.86 | 0.87 | 1025 |
| Walking | 0.98 | 0.98 | 0.98 | 3468 |
|  |
| accuracy | 0.96 | 9060 |
| macro | avg | 0.94 | 0.94 | 0.94 | 9060 |
| weighted | avg | 0.96 | 0.96 | 0.96 | 9060 |


## Feature param
{'LABELS': ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'], 'TIME_PERIODS': 80, 'STEP_DISTANCE': 40, 'N_FEATURES': 3, 'LABEL': 'ActivityEncoded', 'SEED': 314}

## Model size
3.2e-05 MB

## Confusion_matrix
![alt](./cross-tab.png)
