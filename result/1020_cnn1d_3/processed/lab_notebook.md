# Lab Notebook


## Model name
cnn1d

## Start date
2023-10-20 11:09:00.141292

## End date
2023-10-20 11:09:56.642169

## Execution time
0 hours 0 minutes 56 seconds

## Report
| | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
|  |
| Downstairs | 0.00 | 0.00 | 0.00 | 755 |
| Jogging | 0.05 | 0.03 | 0.04 | 2586 |
| Sitting | 0.00 | 0.00 | 0.00 | 461 |
| Standing | 0.00 | 0.00 | 0.00 | 352 |
| Upstairs | 0.00 | 0.00 | 0.00 | 917 |
| Walking | 0.39 | 0.81 | 0.53 | 3166 |
|  |
|  accuracy || | 0.32 | 8237 |
| macro | avg | 0.07 | 0.14 | 0.09 | 8237 |
| weighted | avg | 0.17 | 0.32 | 0.22 | 8237 |


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
Size: 1388975    B

## Confusion_matrix
![alt](./assets/cross-tab.png)

## Loss curve
![alt](./assets/loss.png)

## optuna search plots
None
