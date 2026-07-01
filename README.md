# Credit Card Fraud Detection using Optimized ANN and XGBoost

## Overview

This project tackles credit card fraud detection, a classic highly-imbalanced binary
classification problem. Two modeling approaches are built, tuned, and compared on the
same stratified train/validation/test splits:

1. A PyTorch Artificial Neural Network (ANN) trained with Focal Loss and tuned with Optuna.
2. An XGBoost classifier trained with `scale_pos_weight` and also tuned with Optuna.

Both models are evaluated with a decision threshold selected on the validation set by
maximizing the F2-score, then evaluated once on the held-out test set.

## Dataset

The dataset is the well-known [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
(European cardholder transactions, September 2013). It contains 284,807 transactions,
of which only 492 (~0.17%) are fraudulent. Features `V1`-`V28` are PCA-transformed
components (for anonymization), with `Amount` and `Time` provided in original form. The
`Time` column is dropped during preprocessing as it is almost uncorrelated with the
target.

The raw CSV is not included in this repository (see `.gitignore`) — download it from
Kaggle and place it alongside the notebook if you want to rerun it.

## Problem Statement

Fraud detection is a highly imbalanced classification problem: fraudulent transactions
make up a tiny fraction of all transactions. A naive classifier can achieve >99.8%
accuracy by predicting "not fraud" every time, while catching zero fraud. The real goal
is to catch as many fraudulent transactions as possible (recall) while keeping false
positives manageable (precision) — which is why this project optimizes and reports
F2-score rather than accuracy alone.

## Methods Used

- **Stratified train/validation/test split** — preserves the fraud/non-fraud ratio
  across all three splits.
- **StandardScaler fit only on training data** — avoids leaking validation/test
  distribution information into feature scaling.
- **PyTorch ANN** — a configurable feed-forward network (number of hidden layers,
  neurons per layer, dropout, learning rate all tunable).
- **Focal Loss** — down-weights easy (majority-class) examples so the network focuses
  learning signal on the rare fraud cases.
- **Balanced batch sampling** — a custom sampler is used during training so that each
  mini-batch has a more balanced mix of fraud/non-fraud examples.
- **Optuna hyperparameter tuning** — a `MedianPruner`-based study searches network
  architecture and optimizer hyperparameters (ANN) and `scale_pos_weight` / tree
  hyperparameters (XGBoost), maximizing validation F2-score.
- **Early stopping** — training stops once validation loss stops improving, and the
  best checkpoint is restored (patience-based, ANN "marathon" training phase).
- **Precision-Recall threshold tuning** — instead of the default 0.5 cutoff, the
  decision threshold is chosen on the validation set by maximizing F2-score on the
  precision-recall curve, then applied once to the test set.
- **XGBoost with `scale_pos_weight`** — rebalances the loss for the positive (fraud)
  class instead of resampling the data, tuned via Optuna alongside other tree
  hyperparameters.

## Results

| Model    | Test F2 | Test Accuracy | Test Recall |
|----------|--------:|---------------:|------------:|
| ANN      | 0.7315  | 0.9988          | 0.7684      |
| XGBoost  | 0.7768  | 0.9995          | 0.7474      |

XGBoost achieved the higher F2-score and accuracy overall, while the ANN achieved
slightly higher recall (catching a marginally larger share of actual fraud cases at
its chosen threshold).

## Why F2-Score Instead of Only Accuracy

With fraud at ~0.17% of transactions, accuracy is a misleading metric — a model that
always predicts "not fraud" still scores >99.8% accuracy while being useless. F2-score
(a weighted harmonic mean of precision and recall that weights recall twice as heavily
as precision) is used instead because, in fraud detection, missing an actual fraud
case (a false negative) is typically far more costly than investigating a legitimate
transaction flagged as suspicious (a false positive). Optimizing for F2 directly targets
that business priority rather than being distorted by the class imbalance.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (splitting, scaling, metrics)
- PyTorch (ANN)
- XGBoost
- Optuna (hyperparameter tuning)
- TensorBoard (training curve logging)
- matplotlib, seaborn (visualization)

## How to Run the Notebook

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   and place `creditcard.csv` in the same directory referenced by the notebook.
3. Open and run `notebooks/credit-card-fraud-detection-ann-xgboost.ipynb` top to bottom.
   - The ANN section runs an Optuna study, trains the final model with early stopping,
     and evaluates it on validation and test sets.
   - The XGBoost section runs its own Optuna study, trains the final model, and
     evaluates it the same way.

## Key Learnings

- Accuracy is not a meaningful metric on highly imbalanced data; threshold-tuned F2
  score gives a much more honest picture of fraud-catching performance.
- Fitting the scaler only on training data (not train+validation or the full dataset)
  is essential to avoid subtle data leakage.
- Focal Loss and balanced batch sampling both help a neural network learn from a
  small minority class, but a well-tuned gradient-boosted tree model with
  `scale_pos_weight` remains a very strong, more sample-efficient baseline for
  tabular fraud data.
- Choosing the decision threshold via the precision-recall curve on validation data
  (rather than a fixed 0.5 cutoff) is necessary to get useful precision/recall
  trade-offs in a highly imbalanced setting.

## Limitations and Possible Improvements

- The dataset covers only two days of European transactions in 2013; results may not
  generalize to other time periods, regions, or fraud patterns.
- PCA-anonymized features (`V1`-`V28`) limit interpretability of what drives each
  model's decisions.
- Both models are evaluated on a single train/val/test split; k-fold cross-validation
  or repeated splits would give a better estimate of result variance.
- No model explainability (e.g. SHAP) is currently included, which would help justify
  flagged transactions in a real fraud-review workflow.
- Further gains could come from ensembling the ANN and XGBoost predictions, or from
  cost-sensitive threshold selection based on actual business costs of false
  positives vs. false negatives, rather than F2 alone.
