# Lab Notebook


## Model name
convbbt

## Start date
2023-07-17 13:42:01.733368

## End date
2023-07-17 13:54:04.078364

## Execution time
0 hours 12 minutes 2 seconds

## Report
| | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
|  |
| Downstairs | 0.89 | 0.82 | 0.86 | 824 |
| Jogging | 0.99 | 0.98 | 0.98 | 2832 |
| Sitting | 0.98 | 0.98 | 0.98 | 501 |
| Standing | 0.98 | 0.98 | 0.98 | 410 |
| Upstairs | 0.87 | 0.84 | 0.85 | 1025 |
| Walking | 0.95 | 0.98 | 0.97 | 3468 |
|  |
|  accuracy || | 0.95 | 9060 |
| macro | avg | 0.94 | 0.93 | 0.94 | 9060 |
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
Size: 10612929   B

## Confusion_matrix
![alt](./cross-tab.png)

## Loss curve
![alt](./loss.png)
