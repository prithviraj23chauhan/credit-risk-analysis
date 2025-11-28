📊 Credit Risk Analysis & Scoring System (FinTech – BharatEarns)

A complete end-to-end credit risk prediction project built using
Machine Learning, Deep Learning, and Model Explainability.
Designed for BharatEarns (FinTech) as part of an internship project & academic submission.

This project predicts whether a loan applicant is likely to default using multiple models, ensembling techniques, and an interactive Streamlit dashboard.

🚀 Project Highlights
✔ 1. Multiple Models Implemented

Machine Learning Models

Logistic Regression

Support Vector Machine (SVM)

Random Forest

LightGBM (LGBMClassifier)

Deep Learning Model

DeepFM (feature-aware neural network)

Ensembling Techniques

Soft Voting Ensemble

Stacking Ensemble (best performing)

✔ 2. Three Stages Evaluated

Each model is trained and evaluated at:

Stage	Description
Base	Original imbalanced data
Balanced	Class rebalancing (oversampling / weights)
Tuned	Hyperparameter optimization

All metrics are logged as:

Accuracy

Precision

Recall

F1 Score

ROC–AUC

Stored in results/*.csv.

✔ 3. Production-Grade ML Pipeline

All models saved as bundles (.pkl) with:

model

feature list

feature means

Ensures consistent predictions at inference time.

Config-driven design using:

config/models.yaml
src/pipeline.py
src/model_registry.py

✔ 4. Interactive Streamlit Dashboard

The app supports:

🔹 Single Customer Prediction

Input limited key fields

Remaining features autofilled using feature means

Outputs:

Default probability

Risk label (High/Low)

🔹 Batch Prediction

Upload a CSV with the same feature columns as training (feature_cols).

App provides:

Risk prediction for all rows

Count of high-risk customers

Average probability

Downloadable scored CSV

Sample file included:

data/sample_input.csv

🔹 Model Comparison Dashboard

Load evaluation metrics across Base / Balanced / Tuned

Plot metrics: Accuracy / Precision / Recall / F1 / AUC

Compare models visually

Select model in sidebar

🔹 Explainability (SHAP)

For tree-based models (Random Forest / LightGBM):

Global SHAP summary plot showing feature importance

Helps identify top drivers of credit default risk

🗂️ Project Structure
"""credit-risk-analysis/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── pipeline.py
│   ├── train_models.py
│   ├── model_registry.py
│   └── config.py
│
├── models/
│   └── credit_risk_*.pkl
│
├── results/
│   ├── results_base.csv
│   ├── results_balanced.csv
│   └── results_tuned.csv
│
├── data/
│   └── sample_input.csv
│
├── config/
│   └── models.yaml
│
├── notebooks/
│   └── Credit_Risk_Analysis.ipynb
│
├── requirements.txt
└── README.md
"""

🛠️ Installation & Setup
1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate     (Windows)

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
cd app
streamlit run streamlit_app.py

📈 Results Summary

Best models (Tuned Stage):

Stacking Ensemble

Random Forest Tuned

LightGBM Tuned

Showed the highest Recall and ROC–AUC, making them most effective for identifying high-risk defaulting customers.

🔍 Explainability (SHAP)

SHAP Summary Plot highlights:

EXT_SOURCE values

DAYS_BIRTH

AMT_CREDIT

Social Circle counts

Various one-hot encoded financial indicators

These features significantly influence default outcomes.

📄 Project Use Cases

This system can be used by fintech platforms for:

Loan approval decisions

Fraud risk detection

Customer segmentation

Underwriting automation

Monitoring portfolio risk
