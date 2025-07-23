# Wallet Credit Scoring System 
This project implements a credit scoring system for DeFi wallets using transactional behavior from the AAVE protocol. The goal is to analyze on-chain behavior and predict wallet reliability using machine learning models and explainability techniques like SHAP.

# Problem Statement
In traditional finance, credit scores are widely used to assess the risk of lending to a borrower. In DeFi, such mechanisms are rare or non-existent. This project proposes a method to bridge that gap using machine learning to evaluate risk from blockchain-based transaction data.

# Method Chosen
Isolation Forest is used for detecting anomalies in wallet behavior based on aggregated features.

LightGBM is employed as the main classifier to predict liquidation events, which are used as proxy labels for risk scoring.

The anomaly score is scaled to derive a credit score between 300 and 850.

SHAP (SHapley Additive exPlanations) is used to interpret feature importance and model behavior.

Wallets are analyzed across score ranges to observe behavioral patterns.

# Architecture Overview
                 ┌────────────────────┐
                 │  AAVE Wallet Data  │
                 └────────┬───────────┘
                          │
                 ┌────────▼───────────┐
                 │  Feature Aggregator│
                 │ (txns per day, HF) │
                 └────────┬───────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ Isolation Forest (Anomaly Scoring) │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ LightGBM Classifier (Liquidation)  │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ Credit Score Scaler (300–850)      │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ SHAP for Feature Explanation       │
        └────────────────────────────────────┘
# Processing Flow
1 Data Loading
Load transaction-level data for each wallet using load_wallet_data().

2 Feature Engineering
Aggregate wallet-level features like:

tx_per_day

health_factor stats (mean, min, std)

3 Anomaly Detection
Use IsolationForest to score the wallets based on outlier behavior.

4 Model Training
Train a LightGBM model using the aggregated features and anomaly labels.

5 Credit Score Calculation
Convert anomaly scores into a 300–850 credit score scale using MinMaxScaler.

6 Explainability (SHAP)
Generate SHAP value plots to interpret the importance of features.

7 Behavioral Analysis
Analyze wallet behaviors across score ranges (e.g., 300–400, 400–500) to understand risk trends.

# Methodology
1 Data Collection
JSON-formatted AAVE protocol wallet data was downloaded using a Google Drive link.
Each entry includes wallet address, timestamp, transaction details, and optional health factors.

2 Feature Engineering
Computed features per wallet:
Transaction frequency (tx_per_day)
health_factor metrics: mean, min, std
Aggregated features to create a wallet-level dataset.

3 Anomaly Detection
Used Isolation Forest to identify wallets with unusual patterns (e.g., extremely high or low health factors).
Anomaly score added as a risk indicator.

4 Supervised Modeling
Trained a LightGBM classifier on the feature set to predict liquidation risk.
Used SHAP values to explain feature contributions.

5 Credit Scoring
Transformed anomaly scores to a credit score range (300–850) using MinMaxScaler.
Higher scores → safer wallets; lower scores → risky wallets.

# Score Distribution
Wallets are scored and grouped as follows:

300–500: High-risk wallets — low transaction activity, poor health factor, likely to be liquidated.

500–700: Moderate-risk wallets — consistent behavior with occasional risk.

700–850: Low-risk wallets — stable, frequent transactions and high health factor.

# Use Cases
Risk assessment for decentralized lending platforms.

Behavioral segmentation of wallet users.

Fraud/anomaly detection in blockchain finance.

Creditworthiness assessment in on-chain loan issuance.

# Future Work
Integrate with real-time on-chain APIs.

Expand features with liquidity, collateral types, and token behavior.

Introduce dynamic time-series modeling for prediction.

Use ensemble models for better accuracy.









