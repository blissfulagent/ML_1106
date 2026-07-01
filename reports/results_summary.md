# Results Summary — Credit Card Fraud Detection

## Dataset Imbalance

- 284,807 transactions total, 492 labeled fraud (~0.17% positive class).
- `Time` dropped (near-zero correlation with target); duplicate rows dropped.
- Stratified train/validation/test split preserves this ~0.17% ratio in every split so
  validation/test metrics reflect real-world class balance.

## ANN Approach

- StandardScaler fit on training data only, applied to validation/test.
- Custom feed-forward network, hyperparameters (hidden layers, neurons per layer,
  learning rate, dropout) tuned via Optuna with `MedianPruner`, maximizing validation
  F2-score.
- Trained with Focal Loss (`alpha`/`gamma`-weighted BCE) plus a balanced batch sampler
  to counter class imbalance during training.
- Final model retrained with best Optuna params, using early stopping (patience-based)
  against validation loss.
- Decision threshold selected on the validation precision-recall curve by maximizing
  F2, then applied once to the test set.

**Test results:** F2 = 0.7315, Accuracy = 0.9988, Recall = 0.7684

## XGBoost Approach

- Same stratified split methodology, `Time` dropped, duplicates removed.
- `scale_pos_weight` used to rebalance the weighted logistic loss for the minority
  (fraud) class.
- Hyperparameters (including a `scale_pos_weight` scaling factor and tree parameters)
  tuned via Optuna with `MedianPruner`, maximizing validation F2-score.
- Final model trained with early stopping rounds; decision threshold selected on the
  validation precision-recall curve by maximizing F2, then applied once to the test
  set.

**Test results:** F2 = 0.7768, Accuracy = 0.9995, Recall = 0.7474

## Final Comparison

| Model    | Test F2 | Test Accuracy | Test Recall |
|----------|--------:|---------------:|------------:|
| ANN      | 0.7315  | 0.9988          | 0.7684      |
| XGBoost  | 0.7768  | 0.9995          | 0.7474      |

## Recommendation

XGBoost performed better on F2 (and accuracy), making it the stronger overall choice
for this dataset. The ANN gave slightly higher recall, so it may still be worth
considering in scenarios where catching marginally more fraud cases is prioritized
over overall precision/F2 — for example as part of an ensemble or a secondary review
signal alongside XGBoost's predictions.
