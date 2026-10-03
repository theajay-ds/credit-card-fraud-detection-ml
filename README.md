# Credit Card Fraud Detection using Machine Learning

A Machine Learning project that detects potentially fraudulent credit card transactions using feature engineering, preprocessing, and a Random Forest classification model.

## Project Overview

Credit card fraud is a major problem in digital payments. The objective of this project is to build a machine learning system that analyzes transaction-related information and predicts whether a transaction is likely to be fraudulent.

The project includes:

- Data inspection and analysis
- Risk analysis
- Feature engineering
- Data preprocessing
- Random Forest classification
- Model evaluation
- Transaction prediction
- Streamlit-based user interface

## Dataset

The project uses a credit card transaction dataset containing **10,000 transactions** and **10 columns**.

### Main Features

| Feature | Description |
|---|---|
| transaction_id | Unique transaction identifier |
| amount | Transaction amount |
| transaction_hour | Hour when the transaction occurred |
| merchant_category | Category of the merchant |
| foreign_transaction | Indicates whether the transaction is foreign |
| location_mismatch | Indicates a location mismatch |
| device_trust_score | Device trust/risk score |
| velocity_last_24h | Number of recent transactions |
| cardholder_age | Age of the cardholder |
| is_fraud | Target variable |

The dataset contains:

- **9,849 non-fraud transactions**
- **151 fraud transactions**
- Fraud rate: **1.51%**

## Feature Engineering

Additional features were created to improve the transaction risk representation:

- `is_night`
- `is_early_morning`
- `high_velocity`
- `low_device_trust`
- `foreign_location_risk`
- `amount_log`
- `amount_per_velocity`

These features represent transaction timing, transaction velocity, device trust, location risk, and transaction amount patterns.

## Machine Learning Model

The project uses a **Random Forest Classifier**.

### Model configuration

- Number of trees: 200
- Class weighting: Balanced
- Random state: 42
- Train-test split: 80:20
- Stratified train-test split

### Preprocessing

Numerical features are processed using:

- StandardScaler

Categorical features are processed using:

- OneHotEncoder

The preprocessing steps are combined using a ColumnTransformer.

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

### Test Results

| Metric | Score |
|---|---:|
| Accuracy | 1.00 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 Score | 1.00 |
| ROC-AUC | 1.00 |
| PR-AUC | 1.00 |

### Confusion Matrix

```text
                 Predicted
              Non-Fraud  Fraud

Actual
Non-Fraud       1970       0
Fraud              0      30