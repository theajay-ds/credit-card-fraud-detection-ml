import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data
df =pd.read_excel(r"C:\Users\Sanjay Gupta\Downloads\credit.xls")

print("dataset laoded suceesfully")
print("total tarnsacton:",len(df))

#function to calculte  fraud rate

def fraud_rate(column):

    result=(
        df.groupby(column)["is_fraud"]
        .agg(["count","sum","mean"])
        .reset_index()
    )
    result["fraud_rate"]=result["mean"]*100     
    result =result.drop(columns=["mean"])
    return result

    
    
    print("\n========== FRAUD RATE BY FOREIGN TRANSACTION ==========")

foreign_result = fraud_rate("foreign_transaction")

print(foreign_result)


plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=foreign_result,
    x="foreign_transaction",
    y="fraud_rate"
)

plt.title("Fraud Rate by Foreign Transaction", fontsize=15, fontweight="bold")
plt.xlabel("Foreign Transaction")
plt.ylabel("Fraud Rate (%)")

plt.xticks(
    [0, 1],
    ["Domestic", "Foreign"]
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f%%",
        padding=5
    )

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 2. FRAUD RATE BY LOCATION MISMATCH
# ==========================================

print("\n========== FRAUD RATE BY LOCATION MISMATCH ==========")

location_result = fraud_rate("location_mismatch")

print(location_result)


plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=location_result,
    x="location_mismatch",
    y="fraud_rate"
)

plt.title("Fraud Rate by Location Mismatch", fontsize=15, fontweight="bold")
plt.xlabel("Location Mismatch")
plt.ylabel("Fraud Rate (%)")

plt.xticks(
    [0, 1],
    ["No Mismatch", "Mismatch"]
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f%%",
        padding=5
    )

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 3. FRAUD RATE BY MERCHANT CATEGORY
# ==========================================

print("\n========== FRAUD RATE BY MERCHANT CATEGORY ==========")

merchant_result = fraud_rate("merchant_category")

print(merchant_result)


plt.figure(figsize=(10, 5))

ax = sns.barplot(
    data=merchant_result,
    x="merchant_category",
    y="fraud_rate"
)

plt.title("Fraud Rate by Merchant Category", fontsize=15, fontweight="bold")
plt.xlabel("Merchant Category")
plt.ylabel("Fraud Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f%%",
        padding=5
    )

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 4. FRAUD RATE BY TRANSACTION HOUR
# ==========================================

print("\n========== FRAUD RATE BY TRANSACTION HOUR ==========")

hour_result = fraud_rate("transaction_hour")

print(hour_result)


plt.figure(figsize=(12, 5))

ax = sns.barplot(
    data=hour_result,
    x="transaction_hour",
    y="fraud_rate"
)

plt.title("Fraud Rate by Transaction Hour", fontsize=15, fontweight="bold")
plt.xlabel("Transaction Hour")
plt.ylabel("Fraud Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3,
        fontsize=8
    )

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 5. FRAUD RATE BY DEVICE TRUST SCORE
# ==========================================

print("\n========== DEVICE TRUST SCORE ANALYSIS ==========")

device_result = (
    df.groupby("device_trust_score")["is_fraud"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

device_result["fraud_rate"] = device_result["mean"] * 100

print(device_result.sort_values("fraud_rate", ascending=False).head(10))


plt.figure(figsize=(10, 5))

sns.scatterplot(
    data=device_result,
    x="device_trust_score",
    y="fraud_rate",
    size="count",
    sizes=(30, 300)
)

plt.title("Device Trust Score vs Fraud Rate", fontsize=15, fontweight="bold")
plt.xlabel("Device Trust Score")
plt.ylabel("Fraud Rate (%)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 6. FRAUD RATE BY TRANSACTION VELOCITY
# ==========================================

print("\n========== FRAUD RATE BY TRANSACTION VELOCITY ==========")

velocity_result = fraud_rate("velocity_last_24h")

print(velocity_result)


plt.figure(figsize=(10, 5))

ax = sns.barplot(
    data=velocity_result,
    x="velocity_last_24h",
    y="fraud_rate"
)

plt.title(
    "Fraud Rate by Transactions in Previous 24 Hours",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Transactions in Previous 24 Hours")
plt.ylabel("Fraud Rate (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f%%",
        padding=4
    )

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
