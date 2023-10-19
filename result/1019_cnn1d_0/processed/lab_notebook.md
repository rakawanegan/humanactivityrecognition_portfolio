# Lab Notebook


## Model name
cnn1d

## Start date
2023-10-19 12:51:29.085435

## End date
2023-10-19 12:53:27.357455

## Execution time
0 hours 1 minutes 58 seconds

## Report
| | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
|  |
| Downstairs | 0.70 | 0.01 | 0.02 | 755 |
| Jogging | 0.91 | 0.96 | 0.93 | 2586 |
| Sitting | 0.71 | 0.88 | 0.78 | 461 |
| Standing | 0.96 | 0.06 | 0.12 | 352 |
| Upstairs | 0.21 | 0.00 | 0.01 | 917 |
| Walking | 0.64 | 0.98 | 0.77 | 3166 |
|  |
|  accuracy || | 0.73 | 8237 |
| macro | avg | 0.69 | 0.48 | 0.44 | 8237 |
| weighted | avg | 0.70 | 0.73 | 0.64 | 8237 |


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
