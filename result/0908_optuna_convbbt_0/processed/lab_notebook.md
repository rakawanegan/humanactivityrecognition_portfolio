# Lab Notebook


## Model name
optuna_convbbt

## Start date
2023-09-08 14:47:06.430636

## End date
2023-09-09 01:12:07.552612

## Execution time
10 hours 25 minutes 1 seconds

## Report
| | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
|  |
| Downstairs | 0.00 | 0.00 | 0.00 | 755 |
| Jogging | 0.31 | 0.32 | 0.31 | 2586 |
| Sitting | 0.06 | 0.11 | 0.08 | 461 |
| Standing | 0.00 | 0.00 | 0.00 | 352 |
| Upstairs | 0.00 | 0.00 | 0.00 | 917 |
| Walking | 0.38 | 0.56 | 0.45 | 3166 |
|  |
|  accuracy || | 0.32 | 8237 |
| macro | avg | 0.12 | 0.16 | 0.14 | 8237 |
| weighted | avg | 0.24 | 0.32 | 0.27 | 8237 |


## Optuna search space
- lr: [1e-06, 1e-05, 0.0001, 0.001]
- beta1: [0.9, 0.95, 0.99, 0.999]
- beta2: [0.9, 0.95, 0.99, 0.999]
- eps: [1e-09, 1e-08, 1e-07, 1e-06]
- T_max: [50, 100, 150, 200]
- eta_min: [0, 1e-08, 1e-07, 1e-06, 1e-05]
- hidden_ch: [3, 5, 7, 8, 10, 15]
- depth: [3, 5, 6, 8]
- heads: [3, 5, 6, 8, 10]
- hidden_dim: [64, 128, 256, 512, 1024]
- mlp_dim: [256, 512, 1024, 2048]
- dropout: [0.01, 0.1, 0.25, 0.5, 0.8]
- emb_dropout: [0.01, 0.1, 0.25, 0.5, 0.8]

## Feature param
- LABELS: Downstairs, Jogging, Sitting, Standing, Upstairs, Walking
- TIME_PERIODS: 80
- STEP_DISTANCE: 40
- N_FEATURES: 3
- LABEL: ActivityEncoded
- SEED: 314
- TIMEOUT_HOURS: 10

## Model size
Size: 69717937   B

## Confusion_matrix
![alt](./assets/cross-tab.png)

## Loss curve
![alt](./assets/loss.png)

## optuna search plots
![](./assets/optimization_history.png)
![](./assets/optimization_importance.png)
