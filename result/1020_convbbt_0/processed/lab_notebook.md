# Lab Notebook


## Model name
convbbt

## Start date
2023-10-20 09:46:27.138696

## End date
2023-10-20 10:08:59.087692

## Execution time
0 hours 22 minutes 31 seconds

## Report
| | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
|  |
| Downstairs | 0.58 | 0.31 | 0.40 | 755 |
| Jogging | 0.95 | 0.97 | 0.96 | 2586 |
| Sitting | 0.99 | 0.96 | 0.98 | 461 |
| Standing | 0.94 | 0.97 | 0.96 | 352 |
| Upstairs | 0.57 | 0.33 | 0.42 | 917 |
| Walking | 0.78 | 0.95 | 0.86 | 3166 |
|  |
|  accuracy || | 0.83 | 8237 |
| macro | avg | 0.80 | 0.75 | 0.76 | 8237 |
| weighted | avg | 0.81 | 0.83 | 0.81 | 8237 |


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
- Model_params: {'num_classes': 6, 'input_dim': 80, 'channels': 3, 'hidden_ch': 25, 'hidden_dim': 1024, 'depth': 5, 'heads': 8, 'mlp_dim': 1024, 'dropout': 0.01, 'emb_dropout': 0.01}

## Model size
Size: 84531587   B

## Confusion_matrix
![alt](./assets/cross-tab.png)

## Loss curve
![alt](./assets/loss.png)

## optuna search plots
None
