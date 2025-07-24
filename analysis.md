# Wallet Credit Score Analysis Report

# Introduction
This report presents a comprehensive analysis of credit scores computed for individual blockchain wallets. The scoring mechanism is based on user behavior patterns derived from transactional data, health factor trends, and anomaly detection models. The intent is to develop a data-driven approach to identify trustworthy wallets, mitigate financial risk in DeFi ecosystems, and facilitate more informed decision-making in decentralized finance platforms.

# Objective
The primary goal of this project is to analyze, score, and categorize blockchain wallets based on their historical behavior. The analysis includes:

Score distribution across all wallets.

Behavioral insights across different score ranges.

Risk profiling and detection of anomalies.

Explanation of how each feature affects the credit score using SHAP values.

# Credit Score Ranges
To better understand wallet behaviors, we grouped the scores into several distinct ranges:

# Score Range	Description

0–100	              Extremely risky
101–200	            Very high risk
201–300	            High risk
301–400	            Medium risk
401–500           	Low risk
501–600           	Lower risk
601–700           	Fairly stable
701–800           	Trusted
801–850           	Highly trusted (Excellent)

# Score Distribution
The graph below shows the number of wallets in each score bucket. It helps visualize how many wallets fall into risky vs stable behavior categories.

# Observations:

Majority of wallets fall within the 300–600 range.

Very few wallets scored in the excellent range (800+).

Risky behavior (0–200) was observed in a small subset of wallets, indicating potential anomalies or bad actors.

# Behavior of Wallets by Score Range
0–200: Extremely Risky Wallets

# Characteristics:

Very low or irregular transaction frequency.

Low or volatile health factors.

Sudden spikes in debt or liquidation-related activity.

# Implications:

These wallets may belong to new or untrustworthy users.

Potentially abandoned or automated wallets.

Frequent liquidation events or usage of risky lending protocols.


# 201–400: High to Medium Risk Wallets
# Characteristics:

Unstable or inconsistent health metrics.

Moderate transaction activity but lacks diversification.

Occasional risky behavior detected.

# Implications:

These wallets are likely users with limited experience or unstable liquidity positions.

Need monitoring if used in DeFi lending/borrowing protocols.

 # 401–600: Low to Moderate Risk Wallets
 # Characteristics:

Balanced and frequent transactions.

Relatively stable health factor values.

Limited signs of risky borrowing or liquidation activity.

# Implications:

These are average DeFi users with decent history.

Likely to be semi-reliable users in the ecosystem.

# 601–850: Trusted and Reliable Wallets
# Characteristics:

High transaction frequency with steady inflow/outflow patterns.

Excellent or above-average health factor metrics.

No signs of liquidation or risky borrowing.

# Implications:

These wallets are likely long-term DeFi users, traders, or institutions.

Safe to engage in high-trust environments such as lending or DAOs.


