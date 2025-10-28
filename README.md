# Credit Card Fraud Detection (Stacked ML + Deep Learning)

This project builds a **high-performance fraud detection system** using a **stacked ensemble of XGBoost, Feed-Forward Neural Network (FFN), and LSTM** models optimized for GPU acceleration on large-scale transactional data.

---

## Project Overview
- **Dataset:** 6.3M transactions (Kaggle: `PS_20174392719_1491204439457_log.csv`)
- **Goal:** Detect fraudulent financial transactions in real time.
- **Tech Stack:** `Python`, `TensorFlow`, `XGBoost`, `Scikit-learn`, `imbalanced-learn`, `Kaggle GPU`

---

## Model Architecture
1. **Feature Engineering**
   - Derived 16 tabular features including transaction deltas, ratios, and log-scaled amounts.
   - Encoded categorical features and scaled numeric variables.

2. **Base Models**
   - **XGBoost (GPU)** — gradient boosting with imbalance-aware loss.
   - **Feed-Forward Neural Network (FFN)** — trained with oversampling (SMOTE + RandomOverSampler).
   - **LSTM (optional)** — models sequential transaction patterns for temporal fraud detection.

3. **Stacking Ensemble**
   - Meta-learner: Logistic Regression combining XGBoost and FFN predictions.
   - Achieved **ROC-AUC ≈ 0.9999**, **Recall ≈ 99.5%**, **Precision ≈ 32%**.

---

## Key Results
| Metric | Score |
|:--|:--:|
| ROC-AUC | **0.9999** |
| Recall | **0.995** |
| Precision | **0.324** |
| F1 Score | **0.489** |

- Top features: `amount`, `step`, `oldbalanceOrg`, `newbalanceDest`, `deltaDest`
- Trained on GPU in Kaggle environment with deterministic reproducibility.

---

