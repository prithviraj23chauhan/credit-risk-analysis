# 📊 Credit Risk Analysis & Scoring System (FinTech – BharatEarns)

A complete end-to-end **Credit Risk Prediction System** developed for **BharatEarns (FinTech)** as part of an internship & academic project.

This system predicts whether a loan applicant is likely to **default**, using machine learning, deep learning, and ensembling techniques. It includes a fully interactive **Streamlit dashboard**, model comparison utilities, and SHAP-based explainability.

---

## 🚀 Key Features

### ✔ Multiple Machine Learning Models
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- LightGBM  

### ✔ Deep Learning Model
- **DeepFM** (feature-aware neural network)

### ✔ Ensemble Methods
- Soft Voting Ensemble  
- Stacking Ensemble (best-performing model)

---

## 🎯 Three Evaluation Stages

Each model is evaluated in three phases:

| Stage | Description |
|-------|-------------|
| **Base** | Original imbalanced dataset |
| **Balanced** | Class balancing using weights / oversampling |
| **Tuned** | Full hyperparameter optimization |

All metrics logged:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC–AUC  

Stored inside `results/` folder.

---

## 🛠 End-to-End Pipeline

### ✔ Model Bundles (`.pkl`)
Each model is saved as a bundle with:
- `model`  
- `feature_cols`  
- `feature_means`  

Ensures consistent inference even if user uploads incomplete CSVs.

### ✔ Config-Driven Architecture
The system loads models dynamically through:

```
config/models.yaml
src/model_registry.py
src/pipeline.py
```

---

## 🖥 Streamlit Dashboard

### 🔹 **1. Single Customer Prediction**
- Enter key customer fields  
- Remaining features auto-filled from training means  
- Outputs:
  - Default probability
  - Predicted label (High/Low Risk)

---

### 🔹 **2. Batch Prediction**
Upload a CSV with same features as training data.

The system displays:
- Predictions for each customer  
- Total applicants  
- High-risk count  
- Mean default probability  
- Downloadable CSV with predictions  

A ready-to-use example file is included:
```
data/sample_input.csv
```

---

### 🔹 **3. Model Comparison Dashboard**
Compare all models across **Base**, **Balanced**, and **Tuned** stages.

- Plot metrics: Accuracy / Precision / Recall / F1 / AUC  
- Select specific model to view progression  
- Helps interpret performance improvements

---

### 🔹 **4. Explainability (SHAP)**
For **tree-based models** (RF & LGBM):

- Global SHAP summary plot  
- Shows most influential features  
- Helps justify decisions for risk assessment  

---

## 📂 Project Structure

```
credit-risk-analysis/
│
├── app/
│   └── streamlit_app.py           # Main dashboard
│
├── src/
│   ├── pipeline.py                # Prediction pipeline
│   ├── train_models.py            # Training scripts
│   ├── model_registry.py          # Model loader
│   └── config.py                  # Config helpers
│
├── models/                        # Saved model bundles (.pkl)
│
├── results/
│   ├── results_base.csv
│   ├── results_balanced.csv
│   └── results_tuned.csv
│
├── data/
│   └── sample_input.csv           # Example batch input
│
├── config/
│   └── models.yaml                # Model settings & paths
│
├── notebooks/
│   └── Credit_Risk_Analysis.ipynb # Feature engineering + EDA
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Running

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate    # Windows
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Launch Streamlit App
```bash
cd app
streamlit run streamlit_app.py
```

---

## 📈 Results Summary

**Best Performing Models (Tuned Stage):**

| Model | Highlights |
|-------|------------|
| **Stacking Ensemble** | Best tradeoff of Precision/Recall/AUC |
| **LightGBM Tuned** | Highest AUC, strong recall |
| **Random Forest Tuned** | Excellent balanced performance |

These models are recommended for production deployment.

---

## 🔍 Key Insights from SHAP

Top global risk indicators include:
- `EXT_SOURCE_2`  
- `DAYS_BIRTH`  
- `AMT_CREDIT`  
- Social Circle Indicators  
- Regional & Housing Type features  
- Employment and Document verification flags  

---

## 📄 Use Cases

This project fits into real FinTech workflows:
- Credit scoring for loan approval  
- Risk-based pricing  
- Fraud detection  
- Customer segmentation  
- Underwriting automation  

---

