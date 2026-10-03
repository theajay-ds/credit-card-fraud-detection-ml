import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from feature_engineering import add_features


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"C:\Users\Sanjay Gupta\Downloads\credit.xls"

MODEL_DIR = "../models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "rf_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "feature_config.pkl"
)


# ============================================================
# 1. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_excel(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\nApplying feature engineering...")

df, device_threshold = add_features(df)

print("Feature engineering completed.")

print("\nDevice trust threshold:", device_threshold)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["is_fraud", "transaction_id"]
)

y = df["is_fraud"]


print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# 7. FEATURE TYPES
# ============================================================

categorical_features = [
    "merchant_category"
]

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)


# ============================================================
# 8. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


print("\nFitting preprocessing...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_processed,
    y_train
)

print("\nRandom Forest training completed.")


# ============================================================
# 10. PREDICTIONS
# ============================================================

rf_pred = rf_model.predict(
    X_test_processed
)

rf_probability = rf_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred
    )
)

roc_auc = roc_auc_score(
    y_test,
    rf_probability
)

pr_auc = average_precision_score(
    y_test,
    rf_probability
)

print("\nROC-AUC:", round(roc_auc, 6))
print("PR-AUC :", round(pr_auc, 6))


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

feature_names = (
    preprocessor.get_feature_names_out()
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    rf_model,
    MODEL_PATH
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


# ============================================================
# 14. SAVE FEATURE CONFIGURATION
# ============================================================

feature_config = {
    "device_threshold": float(device_threshold),
    "high_velocity_threshold": 5,
    "fraud_threshold": 0.50,
    "high_amount_threshold": float(
        df["amount"].quantile(0.75)
    )
}

joblib.dump(
    feature_config,
    CONFIG_PATH
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nPreprocessor:")
print(PREPROCESSOR_PATH)

print("\nFeature configuration:")
print(CONFIG_PATH)

print("\nTraining completed successfully.")