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

# Behavior of Wallets in Lower Credit Score Ranges (0–200)
Wallets falling into the 0–200 credit score range are considered extremely risky. These wallets demonstrate behavior patterns and financial activity that raise red flags in a decentralized finance (DeFi) ecosystem.

 # Key Characteristics:
Low Transaction Activity: These wallets typically have very few transactions or exhibit long periods of inactivity. This could indicate dormant wallets or abandoned accounts.

Irregular Patterns: When activity is present, it often appears in irregular spikes, suggesting unpredictable or speculative usage.

Poor Health Factor Metrics: Many of these wallets show low health factors (e.g., borrowing too close to the liquidation threshold), implying high risk in DeFi lending platforms.

Frequent Liquidations: A common trait is a history of liquidation events or usage of volatile assets as collateral without adequate backing.

High Anomaly Scores: These wallets often trigger anomaly detection models due to sudden shifts in behavior, balance, or borrowing.

Minimal On-Chain Reputation: They may be new to the ecosystem or lack diverse interactions with DeFi protocols, reducing their trustworthiness.

 # Potential Behavior Patterns:
# Behavior Type	Description
Speculative Trading	Sudden, short-term interactions with risky protocols or assets.
Abandoned Wallets	Wallets with no activity for long periods.
Flash Loan Exploits	Attempted usage of flash loans without proper collateral.
Bot Activity	Irregular, automated behavior suggesting the wallet is controlled by scripts or bots.
Low Collateral Ratios	Consistently borrowing against collateral just above liquidation thresholds.

#  Interpretation & Use-Cases:
Risk Mitigation: These wallets should be flagged for high risk in lending and borrowing scenarios.

Exclusion from Incentives: Platforms can avoid rewarding wallets in this range with staking rewards, airdrops, or governance rights.

Further Monitoring: May require real-time monitoring or behavior verification before trust-based interactions.

 # Example Impact:
If used in a DeFi lending protocol, wallets in this range could:

Be more likely to default on loans.

Trigger liquidation cascades in unstable markets.

Damage trust in undercollateralized systems.


<img width="1380" height="678" alt="image" src="https://github.com/user-attachments/assets/983cbb3a-f757-4d44-9bcd-4c6422dee888" />

# Behavior of Wallets in Higher Credit Score Ranges (700–850)
Wallets that fall within the 700–850 credit score range are considered highly reliable and financially healthy. These wallets demonstrate strong engagement with decentralized finance (DeFi) platforms while maintaining a cautious and sustainable financial strategy.

# Key Behavioral Traits:
Consistent and Stable Activity
These wallets show regular interaction with the blockchain—frequent, steady transaction patterns with lending, borrowing, and repaying. Activity is distributed over time and avoids sudden spikes or inactivity.

High Health Factor Maintenance
Wallets maintain high average and minimum health factors, ensuring they are well-collateralized and unlikely to be liquidated. This reflects conservative borrowing behavior.

Low or Zero Liquidation Events
High-score wallets have little to no liquidation history, indicating the user proactively manages collateral ratios and avoids risky market timing.

Engagement with Reputable Protocols
Wallets often interact with established protocols such as Aave, Compound, or Curve, preferring safety and long-term gains over speculative yields.

Low Anomaly Scores
Using unsupervised anomaly detection (e.g., Isolation Forest), these wallets are flagged as normal or low-risk based on their feature patterns.

Moderate Borrowing and Responsible Repayment
When borrowing occurs, it's usually within safe limits. Repayments are timely, and utilization ratios are moderate, reducing systemic risk.

High Creditworthiness
Their behavioral history suggests they could be candidates for future undercollateralized lending models due to their trust profile.

#  Implication of High Scores
These wallets represent ideal borrowers in a credit-based DeFi system.

Lenders or protocols could assign lower interest rates to such wallets, improving capital efficiency.

Wallets in this tier could be early adopters of DeFi credit scoring systems and may benefit from improved financial access.


<img width="1373" height="672" alt="image" src="https://github.com/user-attachments/assets/05150096-a990-4688-bff1-a9a712e6e8bf" />






