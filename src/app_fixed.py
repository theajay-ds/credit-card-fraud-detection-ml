import os
import io
import joblib
import pandas as pd
import streamlit as st

from feature_engineering import add_features


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rf_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_config.pkl"
)

# =========================================================
# FIXED INPUT FILE
# =========================================================

INPUT_FILE = r"C:\Users\Sanjay Gupta\Downloads\new_transactions_template_huggingface_100k.xlsx"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("💳 Credit Card Fraud Detection")

st.caption(
    "Machine Learning based transaction risk analysis"
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
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


# =========================================================
# LOAD MODEL FILES
# =========================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    config = joblib.load(
        CONFIG_PATH
    )

    return model, preprocessor, config


# =========================================================
# CHECK INPUT FILE
# =========================================================

if not os.path.exists(INPUT_FILE):

    st.error(
        "Input Excel file was not found."
    )

    st.code(INPUT_FILE)

    st.stop()


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.subheader("🔍 Transaction Analysis")

st.write(
    "Click the button below to analyze the predefined transaction file."
)

analyze_button = st.button(
    "🔍 Analyze Transactions",
    type="primary",
    width="stretch"
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    try:

        # -------------------------------------------------
        # LOAD INPUT DATA
        # -------------------------------------------------

        df = pd.read_excel(
            INPUT_FILE
        )


        # -------------------------------------------------
        # CHECK REQUIRED COLUMNS
        # -------------------------------------------------

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            st.error(
                "Required columns are missing from the input file."
            )

            st.write(
                missing_columns
            )

            st.stop()


        # -------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------

        with st.spinner(
            "Analyzing transactions..."
        ):

            model, preprocessor, config = (
                load_model_files()
            )


            # -------------------------------------------------
            # FEATURE ENGINEERING
            # -------------------------------------------------

            processed_df, _ = add_features(
                df.copy(),
                device_threshold=config[
                    "device_threshold"
                ]
            )


            # -------------------------------------------------
            # PREPARE FEATURES
            # -------------------------------------------------

            X = processed_df.drop(
                columns=[
                    "transaction_id",
                    "is_fraud"
                ],
                errors="ignore"
            )


            # -------------------------------------------------
            # PREPROCESS DATA
            # -------------------------------------------------

            X_processed = (
                preprocessor.transform(X)
            )


            # -------------------------------------------------
            # FRAUD PROBABILITY
            # -------------------------------------------------

            fraud_probability = (
                model.predict_proba(
                    X_processed
                )[:, 1]
            )


            # -------------------------------------------------
            # FRAUD THRESHOLD
            # -------------------------------------------------

            fraud_threshold = config[
                "fraud_threshold"
            ]


            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            predictions = (
                fraud_probability >= fraud_threshold
            ).astype(int)


            # -------------------------------------------------
            # RISK LEVEL FUNCTION
            # -------------------------------------------------

            def get_risk_level(probability):

                if probability < 0.30:

                    return "LOW"

                elif probability < 0.70:

                    return "MEDIUM"

                else:

                    return "HIGH"


            risk_levels = [
                get_risk_level(probability)
                for probability in fraud_probability
            ]


            # -------------------------------------------------
            # RISK FACTORS FUNCTION
            # -------------------------------------------------

            def get_risk_factors(row):

                factors = []


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
                if (
                    row["device_trust_score"]
                    < config["device_threshold"]
                ):

                    factors.append(
                        "Low device trust"
                    )


                # High transaction velocity
                if row["velocity_last_24h"] >= 5:

                    factors.append(
                        "High transaction velocity"
                    )


                # Night transaction
                if row["transaction_hour"] < 6:

                    factors.append(
                        "Night-time transaction"
                    )


                # Early morning transaction
                if (
                    0
                    <= row["transaction_hour"]
                    <= 3
                ):

                    factors.append(
                        "Early-morning transaction"
                    )


                # Foreign + location mismatch
                if (
                    row["foreign_transaction"] == 1
                    and
                    row["location_mismatch"] == 1
                ):

                    factors.append(
                        "Foreign transaction with location mismatch"
                    )


                # High transaction amount
                if row["amount"] > 500:

                    factors.append(
                        "High transaction amount"
                    )


                # No risk indicators
                if not factors:

                    factors.append(
                        "No major risk indicators"
                    )


                return ", ".join(factors)


            risk_factors = df.apply(
                get_risk_factors,
                axis=1
            )


            # -------------------------------------------------
            # CREATE RESULTS DATAFRAME
            # -------------------------------------------------

            results = pd.DataFrame({

                "Transaction ID":
                    df["transaction_id"],

                "Fraud Probability":
                    (
                        fraud_probability * 100
                    ).round(2),

                "Prediction":
                    predictions,

                "Risk Level":
                    risk_levels,

                "Risk Factors":
                    risk_factors
            })


            # -------------------------------------------------
            # CONVERT PREDICTION TO TEXT
            # -------------------------------------------------

            results["Prediction"] = (
                results["Prediction"].map({

                    0: "NOT FRAUD",

                    1: "FRAUD"

                })
            )


        # =====================================================
        # SUMMARY
        # =====================================================

        st.divider()

        st.subheader(
            "📊 Analysis Summary"
        )


        total_transactions = len(
            results
        )

        fraud_count = (
            results["Prediction"]
            == "FRAUD"
        ).sum()

        safe_count = (
            results["Prediction"]
            == "NOT FRAUD"
        ).sum()


        # -----------------------------------------------------
        # SUMMARY CARDS
        # -----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Transactions",
                total_transactions
            )


        with col2:

            st.metric(
                "🚨 Fraud Detected",
                fraud_count
            )


        with col3:

            st.metric(
                "✅ Safe Transactions",
                safe_count
            )


        # =====================================================
        # PREDICTION RESULTS
        # =====================================================

        st.divider()

        st.subheader(
            "🔎 Prediction Results"
        )


        # Create display copy
        display_results = results.copy()


        # Add % symbol
        display_results[
            "Fraud Probability"
        ] = (
            display_results[
                "Fraud Probability"
            ].astype(str)
            + "%"
        )


        # -----------------------------------------------------
        # SHOW RESULTS TABLE
        # -----------------------------------------------------

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )


        # =====================================================
        # DOWNLOAD RESULTS
        # =====================================================

        st.divider()

        st.subheader(
            "📥 Download Results"
        )


        output_buffer = io.BytesIO()


        results.to_excel(
            output_buffer,
            index=False,
            engine="openpyxl"
        )


        st.download_button(

            label="⬇️ Download Prediction Results",

            data=output_buffer.getvalue(),

            file_name="prediction_results.xlsx",

            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),

            width="stretch"
        )


    # =========================================================
    # ERROR HANDLING
    # =========================================================

    except Exception as e:

        st.error(
            "An error occurred during prediction."
        )

        st.exception(e)