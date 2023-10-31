# Lab Notebook


## Model name
nonlearned_convbbt

## Start date
2023-10-31 19:44:52.231416

## End date
2023-10-31 19:45:58.455660

## Execution time
0 hours 1 minutes 6 seconds

## Report
| | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
|  |
| Downstairs | 0.00 | 0.00 | 0.00 | 755 |
| Jogging | 0.31 | 1.00 | 0.48 | 2586 |
| Sitting | 0.00 | 0.00 | 0.00 | 461 |
| Standing | 0.00 | 0.00 | 0.00 | 352 |
| Upstairs | 0.00 | 0.00 | 0.00 | 917 |
| Walking | 0.00 | 0.00 | 0.00 | 3166 |
|  |
|  accuracy || | 0.31 | 8237 |
| macro | avg | 0.05 | 0.17 | 0.08 | 8237 |
| weighted | avg | 0.10 | 0.31 | 0.15 | 8237 |


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
