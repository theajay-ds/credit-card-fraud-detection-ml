import os
import joblib
import pandas as pd

from feature_engineering import add_features


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Trained model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rf_model.pkl"
)

# Preprocessor
PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)

# Feature configuration
CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_config.pkl"
)

# New transaction Excel file
DEFAULT_INPUT = r"C:\Users\Sanjay Gupta\Downloads\new_transactions_template_huggingface_100k.xlsx"

# Prediction output
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "prediction_results.xlsx"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "transaction_id",
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    if probability < 0.30:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# RISK FACTORS
# ============================================================

def get_risk_factors(row, config):

    factors = []

    device_threshold = config[
        "device_threshold"
    ]

    high_velocity_threshold = config[
        "high_velocity_threshold"
    ]

    high_amount_threshold = config[
        "high_amount_threshold"
    ]

    # Foreign transaction
    if row["foreign_transaction"] == 1:
        factors.append(
            "Foreign transaction"
        )

    # Location mismatch
    if row["location_mismatch"] == 1:
        factors.append(
            "Location mismatch"
        )

    # Low device trust
    if row["device_trust_score"] <= device_threshold:
        factors.append(
            "Low device trust"
        )

    # High transaction velocity
    if row["velocity_last_24h"] >= high_velocity_threshold:
        factors.append(
            "High transaction velocity"
        )

    # Night transaction
    if row["is_night"] == 1:
        factors.append(
            "Night-time transaction"
        )

    # Early morning
    if row["is_early_morning"] == 1:
        factors.append(
            "Early-morning transaction"
        )

    # Foreign + location mismatch
    if row["foreign_location_risk"] == 1:
        factors.append(
            "Foreign transaction with location mismatch"
        )

    # High amount
    if row["amount"] >= high_amount_threshold:
        factors.append(
            "High transaction amount"
        )

    if len(factors) == 0:
        factors.append(
            "No major risk indicators"
        )

    return ", ".join(factors)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD DETECTION")
    print("CONSOLE PREDICTION SYSTEM")
    print("=" * 70)


    # ========================================================
    # 1. CHECK MODEL FILES
    # ========================================================

    print("\nLoading trained model...")

    if not os.path.exists(MODEL_PATH):

        print("\nERROR: rf_model.pkl not found.")
        print("Expected location:")
        print(MODEL_PATH)

        return


    if not os.path.exists(PREPROCESSOR_PATH):

        print("\nERROR: preprocessor.pkl not found.")
        print("Expected location:")
        print(PREPROCESSOR_PATH)

        return


    if not os.path.exists(CONFIG_PATH):

        print("\nERROR: feature_config.pkl not found.")
        print("Expected location:")
        print(CONFIG_PATH)

        return


    # ========================================================
    # 2. LOAD MODEL
    # ========================================================

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    config = joblib.load(
        CONFIG_PATH
    )

    print("Model loaded successfully.")


    # ========================================================
    # 3. CHECK INPUT EXCEL FILE
    # ========================================================

    print("\n" + "-" * 70)
    print("NEW TRANSACTION DATA")
    print("-" * 70)

    print("\nUsing input file:")
    print(DEFAULT_INPUT)


    if not os.path.exists(DEFAULT_INPUT):

        print("\nERROR: Excel file not found.")
        print("Expected location:")
        print(DEFAULT_INPUT)

        return


    # ========================================================
    # 4. LOAD EXCEL DATA
    # ========================================================

    print("\nLoading new transaction data...")

    try:

        df = pd.read_excel(
            DEFAULT_INPUT
        )

    except Exception as e:

        print("\nERROR while reading Excel file:")
        print(e)

        return


    print(
        "Rows loaded:",
        len(df)
    )


    # ========================================================
    # 5. CHECK REQUIRED COLUMNS
    # ========================================================

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]


    if missing_columns:

        print(
            "\nERROR: Missing required columns:"
        )

        for column in missing_columns:

            print(
                "-",
                column
            )

        return


    print("Required columns verified.")


    # ========================================================
    # 6. FEATURE ENGINEERING
    # ========================================================

    print(
        "\nApplying feature engineering..."
    )

    try:

        df, _ = add_features(
            df,
            device_threshold=config[
                "device_threshold"
            ]
        )

    except Exception as e:

        print(
            "\nERROR during feature engineering:"
        )

        print(e)

        return


    print(
        "Feature engineering completed."
    )


    # ========================================================
    # 7. PREPARE FEATURES
    # ========================================================

    X_new = df.drop(
        columns=["transaction_id"],
        errors="ignore"
    )


    # Remove target column if present
    X_new = X_new.drop(
        columns=["is_fraud"],
        errors="ignore"
    )


    # ========================================================
    # 8. PREPROCESS DATA
    # ========================================================

    print(
        "\nPreprocessing transaction data..."
    )

    try:

        X_processed = preprocessor.transform(
            X_new
        )

    except Exception as e:

        print(
            "\nERROR during preprocessing:"
        )

        print(e)

        return


    print(
        "Preprocessing completed."
    )


    # ========================================================
    # 9. MODEL PREDICTION
    # ========================================================

    print(
        "\nRunning Random Forest prediction..."
    )

    try:

        probability = model.predict_proba(
            X_processed
        )[:, 1]

    except Exception as e:

        print(
            "\nERROR during prediction:"
        )

        print(e)

        return


    fraud_threshold = config[
        "fraud_threshold"
    ]


    prediction = (
        probability >= fraud_threshold
    ).astype(int)


    # ========================================================
    # 10. CREATE RESULTS
    # ========================================================

    results = pd.DataFrame()


    results[
        "transaction_id"
    ] = df[
        "transaction_id"
    ]


    results[
        "fraud_probability"
    ] = probability


    results[
        "risk_score"
    ] = probability * 100


    results[
        "prediction"
    ] = prediction


    results[
        "prediction_label"
    ] = results[
        "prediction"
    ].map({
        0: "NOT FRAUD",
        1: "FRAUD"
    })


    results[
        "risk_level"
    ] = [
        get_risk_level(p)
        for p in probability
    ]


    results[
        "risk_factors"
    ] = [
        get_risk_factors(
            row,
            config
        )
        for _, row in df.iterrows()
    ]


    # ========================================================
    # 11. DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)


    for _, row in results.iterrows():

        print(
            f"\nTransaction ID : "
            f"{row['transaction_id']}"
        )

        print(
            f"Fraud Probability : "
            f"{row['fraud_probability'] * 100:.2f}%"
        )

        print(
            f"Risk Score : "
            f"{row['risk_score']:.2f}/100"
        )

        print(
            f"Prediction : "
            f"{row['prediction_label']}"
        )

        print(
            f"Risk Level : "
            f"{row['risk_level']}"
        )

        print(
            f"Risk Factors : "
            f"{row['risk_factors']}"
        )

        print("-" * 70)


    # ========================================================
    # 12. SAVE RESULTS
    # ========================================================

    output_directory = os.path.dirname(
        OUTPUT_PATH
    )


    os.makedirs(
        output_directory,
        exist_ok=True
    )


    try:

        results.to_excel(
            OUTPUT_PATH,
            index=False
        )

    except Exception as e:

        print(
            "\nERROR while saving results:"
        )

        print(e)

        return


    # ========================================================
    # 13. SUMMARY
    # ========================================================

    print(
        "\nResults saved successfully!"
    )

    print(
        "Output file:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nTotal transactions:",
        len(results)
    )

    print(
        "Fraud predictions:",
        int(
            (prediction == 1).sum()
        )
    )

    print(
        "Non-fraud predictions:",
        int(
            (prediction == 0).sum()
        )
    )

    print(
        "\nPrediction completed."
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()