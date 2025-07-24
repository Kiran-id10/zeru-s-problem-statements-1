import json, gzip, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
import gdown
gdrive_url = "https://drive.google.com/uc?id=1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS"
data_path = "aave_data.json"
if not os.path.exists(data_path):
    gdown.download(gdrive_url, data_path, quiet=False)
import gzip
import json

def load_wallet_data(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            for line in f:
                yield json.loads(line)
    else:
        with open(path, "r") as f:
            data = json.load(f)  # For plain .json file
            for item in data:
                yield item
def aggregate_features(txns):
    df = pd.DataFrame(txns).sort_values("timestamp")

    res = {
        "tx_per_day": len(df) / (((df["timestamp"].max() - df["timestamp"].min()) / 86400) + 1)
    }

    if "health_factor" in df.columns:
        res.update({
            "hf_mean": df["health_factor"].mean(),
            "hf_min": df["health_factor"].min(),
            "hf_std": df["health_factor"].std(),
        })
    else:
        res.update({
            "hf_mean": 0,
            "hf_min": 0,
            "hf_std": 0,
        })

    return res
for i, rec in enumerate(load_wallet_data(data_path)):
    print(f"Record {i+1}: {rec}")
    break
wallets = {}
for rec in load_wallet_data(data_path):
    wallets.setdefault(rec["userWallet"], []).append(rec)

rows = []
for wallet, txns in wallets.items():
    feats = aggregate_features(txns)
    feats["wallet"] = wallet
    rows.append(feats)

df = pd.DataFrame(rows).fillna(0)
features = [c for c in df.columns if c not in ["wallet", "had_liquidation"]]

iso = IsolationForest(contamination=0.01, random_state=42)
df["anomaly_score"] = -iso.fit_predict(df[features])
features.append("anomaly_score")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(300, 850))  # Credit score range
df["credit_score"] = scaler.fit_transform(df[["anomaly_score"]])

print(df.columns)
df["credit_score"] = (1 - df["anomaly_score"]) * 100
df[["wallet", "credit_score"]].to_csv("wallet_scores.csv", index=False)

from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Assuming your target is binary or categorical
df["target"] = df["anomaly_score"].apply(lambda x: 1 if x > 0 else 0)

# Train a model (e.g., Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df[features], df["target"])

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(df[features])

# SHAP summary plot
shap.summary_plot(shap_vals, df[features], show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.clf()

plt.hist(df.credit_score, bins=20, color="skyblue", edgecolor="black")
plt.title("Credit Score Distribution (0–1000)")
plt.xlabel("Score")
plt.ylabel("Wallet Count")
plt.savefig("score_distribution.png", bbox_inches="tight")
plt.show()


print("All steps complete. Outputs generated:\n")
print("- wallet_scores.csv (wallet → credit score)")
print("- shap_summary.png (SHAP value summary of feature importance)")
print("- score_distribution.png (Histogram of generated credit scores)")

# Display top 10 scores
print("\n  Wallet Credit Scores:")
print(df[["wallet", "credit_score"]].sort_values(by="credit_score", ascending=False))

shap.initjs()

import shap
shap.initjs()  # This line enables SHAP plots to show interactively in Jupyter

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(df[features])

# SHAP summary plot (interactive in notebook)
shap.summary_plot(shap_vals, df[features])

shap.summary_plot(shap_vals, df[features], show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.show()

# Scale anomaly scores to 300–850 range
scaler = MinMaxScaler(feature_range=(300, 850))
df["credit_score"] = scaler.fit_transform(df[["anomaly_score"]])

import matplotlib.pyplot as plt

# Histogram of credit scores
plt.figure(figsize=(10, 6))
plt.hist(df["credit_score"], bins=20, color="yellow", edgecolor="black")
plt.title("Credit Score Distribution (300–850)")
plt.xlabel("Credit Score")
plt.ylabel("Number of Wallets")
plt.grid(True)
plt.savefig("score_distribution.png", bbox_inches="tight")
plt.show()

import pandas as pd

# Make sure df has these columns: ['wallet', 'credit_score', 'tx_per_day', 'hf_mean', 'hf_min', 'hf_std', 'anomaly_score']

# Define score buckets
bins = [300, 400, 500, 600, 700, 800, 850]
labels = ['300-400', '400-500', '500-600', '600-700', '700-800', '800-850']

df["score_range"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=False)

# Analyze key metrics by score range
grouped = df.groupby("score_range")[["tx_per_day", "hf_mean", "hf_min", "hf_std", "anomaly_score"]].mean().reset_index()

print("=== Credit Score Range-wise Wallet Behavior ===\n")
print(grouped.to_string(index=False))

# Optional: Save to markdown-ready CSV for pasting in analysis.md
grouped.to_csv("wallet_behavior_analysis.csv", index=False)

# Define bins for score ranges (e.g., 300–400, 400–500, etc.)
score_bins = [300, 400, 500, 600, 700, 800, 850]
df["score_range"] = pd.cut(df["credit_score"], bins=score_bins)

# Group by score ranges and compute average statistics
group_stats = df.groupby("score_range")[["tx_per_day", "hf_mean", "hf_min", "hf_std"]].mean().reset_index()

# Print behavior of wallets in each score range
print("📊 Average wallet behavior in each credit score range:")
print(group_stats)

# Plot behavior
import seaborn as sns
plt.figure(figsize=(12, 6))
for col in ["tx_per_day", "hf_mean", "hf_min", "hf_std"]:
    sns.barplot(x="score_range", y=col, data=group_stats)
    plt.title(f"{col} by Credit Score Range")
    plt.xlabel("Credit Score Range")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Group by score ranges
score_bins = [300, 400, 500, 600, 700, 800, 850]
df["score_range"] = pd.cut(df["credit_score"], bins=score_bins)

grouped = df.groupby("score_range")["wallet"].count().reset_index().rename(columns={"wallet": "wallet_count"})

print("\n Wallets in Each Credit Score Range:")
print(grouped.to_string(index=False))

# Select behavioral columns
behavior_columns = ["wallet", "tx_per_day", "hf_mean", "hf_min", "hf_std", "anomaly_score", "credit_score"]

# Sort by credit score (optional)
wallet_behaviors = df[behavior_columns].sort_values(by="credit_score", ascending=False)

# Display full table (or limit to top N)
print("\n Wallet Behaviors:")
print(wallet_behaviors.to_string(index=False))

df["score_range"] = pd.cut(df["credit_score"], bins=[300, 400, 500, 600, 700, 800, 850])

behavior_summary = df.groupby("score_range")[["tx_per_day", "hf_mean", "hf_min", "hf_std"]].mean().reset_index()

print("\n📊 Average Wallet Behavior by Score Range:")
print(behavior_summary.to_string(index=False))

# Define lower range
lower_range = df[df["credit_score"] <= 500]

# Print basic stats
print(" Wallets in Lower Credit Score Range (300–500):", len(lower_range))
print(lower_range[["wallet", "credit_score"]].sort_values(by="credit_score"))

# Statistical summary of features
print("\n Statistical Summary (Lower Range):")
print(lower_range.describe())

# Plot transaction per day and health factor stats
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.hist(lower_range["tx_per_day"], bins=20, color="salmon", edgecolor="black")
plt.title("Tx Per Day (Lower Score)")
plt.xlabel("Transactions per Day")
plt.ylabel("Count")

plt.subplot(1, 3, 2)
plt.hist(lower_range["hf_mean"], bins=20, color="orange", edgecolor="black")
plt.title("Health Factor Mean (Lower Score)")
plt.xlabel("HF Mean")
plt.ylabel("Count")

plt.subplot(1, 3, 3)
plt.hist(lower_range["hf_std"], bins=20, color="red", edgecolor="black")
plt.title("HF Std Deviation (Lower Score)")
plt.xlabel("HF Std")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("lower_range_behavior.png", bbox_inches="tight")
plt.show()

# Define higher score range
higher_range = df[df["credit_score"] >= 700]

# Print basic info
print("📈 Wallets in Higher Credit Score Range (700–850):", len(higher_range))
print(higher_range[["wallet", "credit_score"]].sort_values(by="credit_score", ascending=False))

# Statistical summary of features
print("\n📊 Statistical Summary (Higher Range):")
print(higher_range.describe())

# Plot behavior metrics
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.hist(higher_range["tx_per_day"], bins=20, color="green", edgecolor="black")
plt.title("Tx Per Day (Higher Score)")
plt.xlabel("Transactions per Day")
plt.ylabel("Count")

plt.subplot(1, 3, 2)
plt.hist(higher_range["hf_mean"], bins=20, color="lime", edgecolor="black")
plt.title("Health Factor Mean (Higher Score)")
plt.xlabel("HF Mean")
plt.ylabel("Count")

plt.subplot(1, 3, 3)
plt.hist(higher_range["hf_std"], bins=20, color="cyan", edgecolor="black")
plt.title("HF Std Deviation (Higher Score)")
plt.xlabel("HF Std")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("higher_range_behavior.png", bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt

df["score_range"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=False)

# Count of wallets per range
df["score_range"].value_counts().sort_index().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Wallet Count by Credit Score Range")
plt.xlabel("Credit Score Range")
plt.ylabel("Number of Wallets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("wallet_count_by_score_range.png")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the scores
df_scores = pd.read_csv("wallet_scores.csv")

#  Display Top 10 Wallets by Credit Score
print("\n  Wallets by Credit Score:")
print(df_scores.sort_values(by="credit_score", ascending=False))

#  Display Bottom 10 Wallets by Credit Score
print("\n  Wallets by Credit Score:")
print(df_scores.sort_values(by="credit_score"))

#  Define credit score range bins
bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
labels = ["0–100", "100–200", "200–300", "300–400", "400–500", "500–600", "600–700", "700–800", "800–900", "900–1000"]

df_scores["score_range"] = pd.cut(df_scores["credit_score"], bins=bins, labels=labels, right=False)

#  Count of wallets per score range
range_counts = df_scores["score_range"].value_counts().sort_index()
print("\n Wallet Count per Score Range:")
print(range_counts)

#  Plot score range distribution
plt.figure(figsize=(10, 6))
range_counts.plot(kind="bar", color="teal", edgecolor="black")
plt.title("Distribution of Wallets Across Credit Score Ranges")
plt.xlabel("Credit Score Range")
plt.ylabel("Number of Wallets")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("score_range_distribution.png", bbox_inches="tight")
plt.show()
