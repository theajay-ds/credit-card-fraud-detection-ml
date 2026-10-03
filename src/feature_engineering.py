import numpy as np
import pandas as pd


def add_features(df, device_threshold=None):
    """
    Add the same features used during model training.
    """

    df = df.copy()

    # -----------------------------------------
    # 1. Night transaction
    # -----------------------------------------
    df["is_night"] = df["transaction_hour"].apply(
        lambda x: 1 if x < 6 else 0
    )

    # -----------------------------------------
    # 2. Early morning transaction
    # -----------------------------------------
    df["is_early_morning"] = df["transaction_hour"].apply(
        lambda x: 1 if 0 <= x <= 3 else 0
    )

    # -----------------------------------------
    # 3. High transaction velocity
    # -----------------------------------------
    df["high_velocity"] = df["velocity_last_24h"].apply(
        lambda x: 1 if x >= 5 else 0
    )

    # -----------------------------------------
    # 4. Low device trust
    # -----------------------------------------
    if device_threshold is None:
        device_threshold = df["device_trust_score"].quantile(0.25)

    df["low_device_trust"] = (
        df["device_trust_score"] <= device_threshold
    ).astype(int)

    # -----------------------------------------
    # 5. Foreign transaction + location mismatch
    # -----------------------------------------
    df["foreign_location_risk"] = (
        (df["foreign_transaction"] == 1)
        &
        (df["location_mismatch"] == 1)
    ).astype(int)

    # -----------------------------------------
    # 6. Log transaction amount
    # -----------------------------------------
    df["amount_log"] = np.log1p(df["amount"])

    # -----------------------------------------
    # 7. Amount relative to velocity
    # -----------------------------------------
    df["amount_per_velocity"] = (
        df["amount"] / (df["velocity_last_24h"] + 1)
    )

    return df, device_threshold