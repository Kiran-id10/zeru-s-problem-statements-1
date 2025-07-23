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


for i, rec in enumerate(load_wallet_data(data_path)):
    print(f"Record {i+1}: {rec}")
    break

    return res


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


X_train, X_val, y_train, y_val = train_test_split(
    df[features], df.had_liquidation, test_size=0.2, stratify=df.had_liquidation, random_state=42
)

model = lgb.LGBMClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

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


print(" All steps complete. Outputs generated:\n")
print("- wallet_scores.csv (wallet → credit score)")
print("- shap_summary.png (SHAP value summary of feature importance)")
print("- score_distribution.png (Histogram of generated credit scores)")

# Display credit scores
print("\n Wallet Credit Scores:")
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

# Histogram of credit scores range between 
plt.figure(figsize=(10, 6))
plt.hist(df["credit_score"], bins=20, color="skyblue", edgecolor="black")
plt.title("Credit Score Distribution (300–850)")
plt.xlabel("Credit Score")
plt.ylabel("Number of Wallets")
plt.grid(True)
plt.savefig("score_distribution.png", bbox_inches="tight")
plt.show()



