# Lab Notebook


## Model name
transposition_convbbt

## Start date
2023-09-25 16:11:35.747954

## End date
2023-09-25 16:23:26.506410

## Execution time
0 hours 11 minutes 50 seconds

## Report
| | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
|  |
| Downstairs | 0.43 | 0.51 | 0.47 | 755 |
| Jogging | 0.96 | 0.95 | 0.96 | 2586 |
| Sitting | 0.97 | 0.98 | 0.97 | 461 |
| Standing | 0.98 | 0.95 | 0.97 | 352 |
| Upstairs | 0.58 | 0.51 | 0.54 | 917 |
| Walking | 0.92 | 0.92 | 0.92 | 3166 |
|  |
|  accuracy || | 0.85 | 8237 |
| macro | avg | 0.81 | 0.80 | 0.80 | 8237 |
| weighted | avg | 0.86 | 0.85 | 0.85 | 8237 |


## Optuna search space
None

## Feature param
- LABELS: Downstairs, Jogging, Sitting, Standing, Upstairs, Walking
- TIME_PERIODS: 80
- STEP_DISTANCE: 40
- N_FEATURES: 3
- LABEL: ActivityEncoded
- SEED: 314
- MAX_EPOCH: 200
- BATCH_SIZE: 128
- REF_SIZE: 5
- Adam_params: {'lr': 0.0001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False}
- CosineAnnealingLRScheduler_params: {'T_max': 150, 'eta_min': 1e-05, 'last_epoch': -1, 'verbose': False}
- Model_params: {'num_classes': 6, 'input_dim': 3, 'channels': 80, 'hidden_ch': 50, 'hidden_dim': 10, 'depth': 5, 'heads': 8, 'mlp_dim': 30, 'dropout': 0.01, 'emb_dropout': 0.01}

## Model size
Size: 496515     B

## Confusion_matrix
![alt](./assets/cross-tab.png)

## Loss curve
![alt](./assets/loss.png)

## optuna search plots
None
