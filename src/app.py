import os
import io
import joblib
import pandas as pd
import streamlit as st

from feature_engineering import add_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "feature_config.pkl")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


st.title("💳 Credit Card Fraud Detection")
st.write(
    "Upload transaction data to identify potentially fraudulent transactions."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    config = joblib.load(CONFIG_PATH)

    return model, preprocessor, config


try:

    model, preprocessor, config = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📁 Upload Transaction File")

uploaded_file = st.file_uploader(
    "Choose an Excel file",
    type=["xlsx"]
)


# ============================================================
# MAIN PROCESS
# ============================================================

if uploaded_file is not None:

    try:

        # Read Excel
        df = pd.read_excel(uploaded_file)

        st.success(
            f"File loaded successfully: {len(df)} transactions"
        )

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

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            st.error(
                "Missing columns: "
                + ", ".join(missing_columns)
            )

            st.stop()


        st.success("Required columns verified.")


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "🔍 Analyze Transactions",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing transactions..."):

                # Feature engineering
                processed_df,_ = add_features(
                    df.copy(),
                    device_threshold=config["device_threshold"]
                )


                # Prepare model input
                X = processed_df.drop(
                    columns=["transaction_id", "is_fraud"],
                    errors="ignore"
                )


                # Preprocessing
                X_processed = preprocessor.transform(X)


                # Prediction
                probabilities = model.predict_proba(
                    X_processed
                )[:, 1]


                fraud_threshold = config["fraud_threshold"]

                predictions = (
                    probabilities >= fraud_threshold
                ).astype(int)


                # =================================================
                # CREATE RESULT TABLE
                # =================================================

                results = pd.DataFrame()

                results["Transaction ID"] = df[
                    "transaction_id"
                ]

                results["Fraud Probability"] = (
                    probabilities * 100
                ).round(2)

                results["Prediction"] = predictions


                # Convert prediction
                results["Prediction"] = results[
                    "Prediction"
                ].map(
                    {
                        0: "🟢 NOT FRAUD",
                        1: "🔴 FRAUD"
                    }
                )


                # Risk level
                def risk_level(probability):

                    if probability < 0.30:
                        return "🟢 LOW"

                    if probability < 0.70:
                        return "🟡 MEDIUM"

                    return "🔴 HIGH"


                results["Risk Level"] = [
                    risk_level(probability)
                    for probability in probabilities
                ]


                # =================================================
                # RISK FACTORS
                # =================================================

                def risk_factors(row):

                    factors = []

                    if row["foreign_transaction"] == 1:
                        factors.append("Foreign transaction")

                    if row["location_mismatch"] == 1:
                        factors.append("Location mismatch")

                    if row["velocity_last_24h"] >= 5:
                        factors.append(
                            "High transaction velocity"
                        )

                    if (
                        row["foreign_transaction"] == 1
                        and row["location_mismatch"] == 1
                    ):
                        factors.append(
                            "Foreign transaction with location mismatch"
                        )

                    if (
                        row["device_trust_score"]
                        <= config["device_threshold"]
                    ):
                        factors.append("Low device trust")

                    if row["amount"] >= 1000:
                        factors.append(
                            "High transaction amount"
                        )

                    if row["transaction_hour"] < 6:
                        factors.append(
                            "Night-time transaction"
                        )

                    if 0 <= row["transaction_hour"] <= 3:
                        factors.append(
                            "Early-morning transaction"
                        )

                    if not factors:
                        return "No major risk indicators"

                    return ", ".join(factors)


                results["Risk Factors"] = df.apply(
                    risk_factors,
                    axis=1
                )


                # =================================================
                # SUMMARY
                # =================================================

                total = len(results)

                fraud_count = int(
                    (predictions == 1).sum()
                )

                safe_count = int(
                    (predictions == 0).sum()
                )


                st.divider()

                st.subheader("📊 Analysis Summary")


                col1, col2, col3 = st.columns(3)


                with col1:
                    st.metric(
                        "Total Transactions",
                        total
                    )


                with col2:
                    st.metric(
                        "Fraud Detected",
                        fraud_count
                    )


                with col3:
                    st.metric(
                        "Safe Transactions",
                        safe_count
                    )


                if fraud_count > 0:

                    st.error(
                        f"⚠️ {fraud_count} potentially "
                        "fraudulent transaction(s) detected."
                    )

                else:

                    st.success(
                        "✅ No potentially fraudulent "
                        "transactions detected."
                    )


                # =================================================
                # TABLE
                # =================================================

                st.divider()

                st.subheader("📋 Transaction Results")


                display_results = results.copy()


                display_results[
                    "Fraud Probability"
                ] = display_results[
                    "Fraud Probability"
                ].apply(
                    lambda x: f"{x:.2f}%"
                )


                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True
                )


                # =================================================
                # DOWNLOAD
                # =================================================

                output = io.BytesIO()


                with pd.ExcelWriter(
                    output,
                    engine="openpyxl"
                ) as writer:

                    results.to_excel(
                        writer,
                        index=False,
                        sheet_name="Predictions"
                    )


                output.seek(0)


                st.download_button(
                    "⬇️ Download Results",
                    data=output,
                    file_name="prediction_results.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True
                )


    except Exception as e:

        st.error(
            "Something went wrong while processing the file."
        )

        st.exception(e)


else:

    st.info(
        "Upload an Excel transaction file to begin."
    )