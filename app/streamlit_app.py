import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import yaml
import shap
import matplotlib.pyplot as plt


# Map config keys (from models.yaml) to the "clean" names used in results CSVs
CONFIG_TO_CLEAN_NAME = {
    "rf_tuned": "Random Forest",
    "lgbm_tuned": "LGBMClassifier",
    "stacking_tuned": "Stacking Ensemble",
    "voting_tuned": "Soft Voting Ensemble",
    "lr_tuned": "Logistic Regression",
    "svm_tuned": "SVM",
    "deepfm_tuned": "DeepFM",
    
}


FEATURE_DESCRIPTIONS = {
    # ---- Basic Demographic Features ----
    "SK_ID_CURR": "Unique customer identifier.",
    "NAME_CONTRACT_TYPE": "Type of loan contract (Cash loans / Revolving loans).",
    "CODE_GENDER": "Gender of client (M/F).",
    "FLAG_OWN_CAR": "Whether the client owns a car.",
    "FLAG_OWN_REALTY": "Whether the client owns real estate.",
    "CNT_CHILDREN": "Number of children the client has.",
    "AMT_INCOME_TOTAL": "Total annual income of the client.",
    "AMT_CREDIT": "Total credit amount requested by the client.",
    "AMT_ANNUITY": "Loan annuity (regular payment amount).",
    "AMT_GOODS_PRICE": "For consumer loans, the price of the goods to purchase.",

    # ---- External Risk Scores ----
    "EXT_SOURCE_1": "External risk score #1 (higher = lower risk).",
    "EXT_SOURCE_2": "External risk score #2 (higher = lower risk).",
    "EXT_SOURCE_3": "External risk score #3 (higher = lower risk).",

    # ---- Days Indicators (all negative) ----
    "DAYS_BIRTH": "Age of client in days (negative; -14000 ≈ 38 years).",
    "DAYS_EMPLOYED": "Days employed (negative; -1000 ≈ 3 years). A very large value (~365243) indicates unemployment.",
    "DAYS_REGISTRATION": "How many days before the application the client registered.",
    "DAYS_ID_PUBLISH": "Days since client changed identity documents.",
    "OWN_CAR_AGE": "Age of client's car in years.",
    "DAYS_LAST_PHONE_CHANGE": "Days since last phone number change.",

    # ---- Family & Housing ----
    "CNT_FAM_MEMBERS": "Total family members including children.",
    "NAME_FAMILY_STATUS": "Family status: Married / Single / Widow / Civil marriage.",
    "NAME_HOUSING_TYPE": "Housing type: Own / Rented / Parents apartment / Municipal / etc.",

    # ---- Education & Occupation ----
    "NAME_EDUCATION_TYPE": "Education level: Secondary / Higher / Academic degree.",
    "NAME_INCOME_TYPE": "Income category: Working / Pensioner / Commercial associate etc.",
    "OCCUPATION_TYPE": "Type of occupation (Profession).",

    # ---- Flags: Document Submission ----
    # (One-hot encoded)
    "FLAG_DOCUMENT_2": "Whether client submitted document #2.",
    "FLAG_DOCUMENT_3": "Client submitted document #3.",
    "FLAG_DOCUMENT_4": "Client submitted document #4.",
    "FLAG_DOCUMENT_5": "Client submitted document #5.",
    "FLAG_DOCUMENT_6": "Client submitted document #6.",
    "FLAG_DOCUMENT_7": "Client submitted document #7.",
    "FLAG_DOCUMENT_8": "Client submitted document #8.",
    "FLAG_DOCUMENT_9": "Client submitted document #9.",
    "FLAG_DOCUMENT_10": "Client submitted document #10.",
    "FLAG_DOCUMENT_11": "Client submitted document #11.",
    "FLAG_DOCUMENT_12": "Client submitted document #12.",
    "FLAG_DOCUMENT_13": "Client submitted document #13.",
    "FLAG_DOCUMENT_14": "Client submitted document #14.",
    "FLAG_DOCUMENT_15": "Client submitted document #15.",
    "FLAG_DOCUMENT_16": "Client submitted document #16.",
    "FLAG_DOCUMENT_17": "Client submitted document #17.",
    "FLAG_DOCUMENT_18": "Client submitted document #18.",
    "FLAG_DOCUMENT_19": "Client submitted document #19.",
    "FLAG_DOCUMENT_20": "Client submitted document #20.",
    "FLAG_DOCUMENT_21": "Client submitted document #21.",

    # ---- Contact & Phone Flags ----
    "FLAG_MOBIL": "Client has mobile phone.",
    "FLAG_EMP_PHONE": "Client has employer contact phone.",
    "FLAG_WORK_PHONE": "Client has work phone.",
    "FLAG_CONT_MOBILE": "Client can be reached on mobile phone.",
    "FLAG_PHONE": "Client has a phone number.",
    "FLAG_EMAIL": "Client provided email.",

    # ---- Region Information ----
    "REGION_RATING_CLIENT": "Home region rating relative to others.",
    "REGION_RATING_CLIENT_W_CITY": "Client region rating with city adjustment.",
    "REGION_POPULATION_RELATIVE": "Population of client region relative to country.",
    "REGION_CITY_NOT_WORK_CITY": "Client works outside city of residence.",
    "REGION_CITY_NOT_LIVE_CITY": "Client lives outside registered city.",

    # ---- Social Circle Averages ----
    "OBS_30_CNT_SOCIAL_CIRCLE": "Number of observations of client's social circle (30 days).",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Number of defaults among friends/associates (30 days).",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Observations over 60 days.",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Defaults in social circle over 60 days.",

    # ---- One-Hot Encoded Categorical Columns (examples) ----
    # These appear after get_dummies()
    "NAME_EDUCATION_TYPE_Higher education": "Client has higher education.",
    "NAME_EDUCATION_TYPE_Secondary / secondary special": "Client completed secondary education.",
    "NAME_CONTRACT_TYPE_Cash loans": "Contract type: Cash loan.",
    "NAME_CONTRACT_TYPE_Revolving loans": "Contract type: Revolving credit.",
    "NAME_FAMILY_STATUS_Married": "Client is married.",
    "NAME_HOUSING_TYPE_With parents": "Client lives with parents.",
    # ... include all one-hot categories your columns contain ...

    # ---- Ratio / Engineered Features (if present) ----
    "CREDIT_INCOME_RATIO": "Loan amount divided by annual income.",
    "ANNUITY_INCOME_RATIO": "Annuity divided by annual income.",
    "EMPLOYED_TO_AGE_RATIO": "Ratio of employment duration to age.",
    "INCOME_PER_FAMILY_MEMBER": "Income divided by number of family members.",
}


# ---------------- RESULTS LOADING (for metrics) ----------------

@st.cache_data
def load_results_all_stages():
    base_path = RESULTS_DIR / "results_base.csv"
    bal_path = RESULTS_DIR / "results_balanced.csv"
    tuned_path = RESULTS_DIR / "results_tuned.csv"

    try:
        df_base = pd.read_csv(base_path)
        df_bal = pd.read_csv(bal_path)
        df_tuned = pd.read_csv(tuned_path)
    except Exception:
        return None

    if "stage" not in df_base.columns:
        df_base["stage"] = "base"
    if "stage" not in df_bal.columns:
        df_bal["stage"] = "balanced"
    if "stage" not in df_tuned.columns:
        df_tuned["stage"] = "tuned"

    df_all = pd.concat([df_base, df_bal, df_tuned], ignore_index=True)

    # optional: clean model names
    if "model_clean" not in df_all.columns:
        name_map = {}
        for m in df_all["model"].unique():
            # strip "(Base)", "(Balanced)", "(Tuned)" etc
            if "(" in m:
                name_map[m] = m.split("(")[0].strip()
            else:
                name_map[m] = m
        df_all["model_clean"] = df_all["model"].map(name_map)

    return df_all


# ---------------- PATH SETUP ----------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

CONFIG_PATH = ROOT / "config" / "models.yaml"
RESULTS_DIR = ROOT / "results"


# ---------------- CONFIG + MODEL LOADING ----------------

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model_bundle(model_name: str):
    cfg = load_config()
    model_info = cfg["models"][model_name]
    model_path = ROOT / model_info["path"]

    bundle = joblib.load(model_path)
    # Ensure expected keys exist
    bundle.setdefault("feature_cols", None)
    bundle.setdefault("feature_means", {})
    bundle["model_name"] = model_name
    bundle["type"] = model_info.get("type", "sklearn")
    return bundle


def prepare_input_df(df: pd.DataFrame, feature_cols, feature_means):
    # Work on a copy
    df = df.copy()

    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = feature_means.get(col, 0.0)

    # Drop any extra columns
    df = df[feature_cols].copy()

    # Convert to numeric and coerce errors to NaN
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill any remaining NaNs with training means (fallback = 0.0)
    fill_map = {col: feature_means.get(col, 0.0) for col in feature_cols}
    df = df.fillna(value=fill_map)

    return df



def score_dataframe(df_raw: pd.DataFrame, model, feature_cols, feature_means, threshold: float):
    # Align columns with training
    df_prepared = prepare_input_df(df_raw, feature_cols, feature_means)

    # IMPORTANT: use numpy so sklearn doesn't complain about feature names
    X_array = df_prepared.to_numpy()
    proba = model.predict_proba(X_array)[:, 1]
    labels = (proba >= threshold).astype(int)

    result = df_raw.copy()
    result["default_probability"] = proba
    result["predicted_label"] = labels
    result["risk_level"] = result["predicted_label"].map({1: "High Risk", 0: "Low Risk"})

    # df_prepared is returned for SHAP (it has correct columns/order)
    return result, df_prepared



# ---------------- STREAMLIT LAYOUT ----------------

st.set_page_config(
    page_title="Credit Risk Analysis Dashboard",
    layout="wide",
)

st.title("📊 Credit Risk Analysis – Model Dashboard")

st.markdown(
    """
This dashboard compares multiple machine learning models for **credit risk prediction** and
allows you to score individual customers or full CSV files.

Models include Logistic Regression, SVM, Random Forest, LightGBM, DeepFM, and Ensemble methods (Voting, Stacking).
"""
)

cfg = load_config()
model_names = list(cfg["models"].keys())

# ---------- SIDEBAR SETTINGS ----------

st.sidebar.header("⚙️ Global Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose model",
    model_names,
    index=model_names.index(cfg.get("default_model", model_names[0])),
)

threshold = st.sidebar.slider(
    "Decision threshold (for classifying High Risk)",
    min_value=0.1,
    max_value=0.9,
    value=float(cfg.get("decision_threshold", 0.5)),
    step=0.01,
)

st.sidebar.markdown(
    f"""
**Active model:** `{selected_model_name}`  
**Threshold:** `{threshold:.2f}`
"""
)

bundle = load_model_bundle(selected_model_name)
model = bundle["model"]
feature_cols = bundle["feature_cols"]
feature_means = bundle.get("feature_means", {})

if feature_cols is None:
    st.error("This model bundle has no 'feature_cols'. Please re-save it with feature metadata.")
    st.stop()
df_all = load_results_all_stages()

if df_all is not None:
    model_clean = None

    # 1) try explicit mapping
    if selected_model_name in CONFIG_TO_CLEAN_NAME:
        model_clean = CONFIG_TO_CLEAN_NAME[selected_model_name]
    else:
        # 2) fallback heuristic - in case names don’t follow the mapping
        candidates = df_all["model_clean"].unique()
        for c in candidates:
            if c.lower() in selected_model_name.lower() or selected_model_name.lower() in c.lower():
                model_clean = c
                break

    if model_clean is not None:
        df_sel = df_all[df_all["model_clean"] == model_clean].copy()

        st.markdown(f"#### 📌 Performance of `{model_clean}` across stages")

        colm1, colm2, colm3 = st.columns(3)
        for stage_name, colm in zip(["base", "balanced", "tuned"], [colm1, colm2, colm3]):
            row = df_sel[df_sel["stage"] == stage_name]
            if not row.empty:
                r = row.iloc[0]
                with colm:
                    st.metric(
                        label=f"{stage_name.capitalize()} – F1",
                        value=f"{r['f1']:.3f}",
                        help=f"Acc: {r['accuracy']:.3f}, Recall: {r['recall']:.3f}, AUC: {r['roc_auc']:.3f}",
                    )
            else:
                with colm:
                    st.metric(
                        label=f"{stage_name.capitalize()} – F1",
                        value="N/A",
                    )

        df_plot = df_sel[["stage", "accuracy", "precision", "recall", "f1", "roc_auc"]].set_index("stage")
        st.line_chart(df_plot)
    else:
        st.info("No matching evaluation metrics found for this model in `results/`.")
else:
    st.info("Add `results_base.csv`, `results_balanced.csv`, `results_tuned.csv` under `results/` to see model metrics.")

st.markdown(f"### ✅ Active model: `{selected_model_name}`")

import numpy as np  # make sure this import exists at top

# Suggested defaults for manual single-customer inputs
DEFAULT_MANUAL_FEATURES = [
    "AMT_CREDIT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]

# Only keep features that actually exist in this model
available_manual_defaults = [f for f in DEFAULT_MANUAL_FEATURES if f in feature_cols]

st.sidebar.subheader("👤 Single Prediction Settings")

manual_features = st.sidebar.multiselect(
    "Features to manually edit in single-customer form",
    options=feature_cols,
    default=available_manual_defaults,
    help="These features will show up in the single-customer form. All other features are kept at their typical (mean) values.",
)


# ---------------- TABS ----------------

tab_overview, tab_single, tab_batch, tab_explain = st.tabs(
    ["📈 Overview", "👤 Single Prediction", "📁 Batch Prediction", "🧠 SHAP Explainability"]
)


# ========== TAB 1: OVERVIEW (METRICS + PLOTS) ==========

with tab_overview:
    st.subheader("Model Performance Summary")

    df_all = load_results_all_stages()
    if df_all is None:
        st.info("No results CSVs found in `results/`. Add results_base.csv, results_balanced.csv, and results_tuned.csv to use this tab.")
    else:
        st.markdown("#### Comparison Table (All Stages)")
        st.dataframe(df_all)

        st.markdown("#### Metric Comparison Across Stages")
        metric_to_plot = st.selectbox(
            "Select metric to plot",
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
            index=3,  # default f1
        )

        models = df_all["model_clean"].unique()
        stages = ["base", "balanced", "tuned"]

        base_vals, bal_vals, tuned_vals = [], [], []
        for m in models:
            for lst, stage in zip([base_vals, bal_vals, tuned_vals], stages):
                val = df_all[
                    (df_all["model_clean"] == m) & (df_all["stage"] == stage)
                ][metric_to_plot]
                lst.append(val.values[0] if len(val) else np.nan)

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - width, base_vals, width, label="Base")
        ax.bar(x,         bal_vals,  width, label="Balanced")
        ax.bar(x + width, tuned_vals, width, label="Tuned")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel(metric_to_plot.upper())
        ax.set_title(f"{metric_to_plot.upper()} vs Stage")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)

# ========== TAB 2: SINGLE CUSTOMER PREDICTION ==========

import shap
import numpy as np

# --------------------------------------------
#  AUTO SHAP TOOLTIP + CATEGORIZATION SYSTEM
# --------------------------------------------


def compute_feature_importance_for_tooltips(model, X_sample):
    """
    Returns SHAP-based tooltips for tree models
    and generic tooltips for non-tree models (Stacking, Logistic, SVM, DeepFM).
    """

    # ---- 1. Detect if model is tree-based ----
    # Applies only to: RandomForest, LightGBM, XGBoost, GradientBoosting, etc.
    model_name = type(model).__name__.lower()
    tree_keywords = ["forest", "gbm", "lightgbm", "lgbm", "xgb", "gradientboost"]

    if not any(word in model_name for word in tree_keywords):
        # This prevents SHAP failures & keeps app stable
        return {
            feat: f"Model input feature: {feat}. (SHAP not available for {model_name})"
            for feat in X_sample.columns
        }

    # ---- 2. Tree model → run SHAP ----
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # If model has two outputs (list), get positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Compute mean-absolute SHAP importance
        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(X_sample.columns, mean_abs))

        # Sort by importance
        sorted_feats = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        tooltips = {}
        for feat, val in sorted_feats:
            if val == 0:
                tooltips[feat] = f"{feat}: low impact on model prediction."
            else:
                tooltips[feat] = (
                    f"High {feat} changes default risk. "
                    f"SHAP importance score: {val:.4f}."
                )

        return tooltips

    except Exception:
        # SHAP failed → fallback generic descriptions
        return {
            feat: f"Model input feature: {feat}. (SHAP error or unsupported model type)"
            for feat in X_sample.columns
        }



def categorize_features(feature_cols):
    """Organizes features into human-readable groups."""
    categories = {
        "Demographics": [],
        "Financial": [],
        "Contact Flags": [],
        "Documents": [],
        "Housing & Family": [],
        "Social Circle": [],
        "One-Hot Categories": [],
        "Advanced": [],
    }

    for feat in feature_cols:
        f = feat.lower()

        if any(x in f for x in ["birth", "gender", "days_", "id", "age"]):
            categories["Demographics"].append(feat)

        elif any(x in f for x in ["amt_", "income", "ext_source", "credit"]):
            categories["Financial"].append(feat)

        elif "flag_document" in f:
            categories["Documents"].append(feat)

        elif any(x in f for x in ["flag_phone", "flag_cont", "flag_work", "flag_mobil", "flag_email"]):
            categories["Contact Flags"].append(feat)

        elif any(x in f for x in ["family", "housing"]):
            categories["Housing & Family"].append(feat)

        elif "social" in f:
            categories["Social Circle"].append(feat)

        elif any(prefix in f for prefix in [
            "name_", "occupation", "organization", "education", "income_type"
        ]):
            categories["One-Hot Categories"].append(feat)

        else:
            categories["Advanced"].append(feat)

    return categories


# ========================================================
#                  SINGLE PREDICTION TAB  
# ========================================================

with tab_single:

    st.subheader("Single-Customer Risk Prediction (Smart Mode)")
    st.markdown("""
    This section automatically:
    - Picks most important features (via SHAP)
    - Generates tooltips explaining each feature
    - Organizes inputs into categories
    - Hides rarely-used features inside an expandable section
    """)

    # --------- 1. Load SHAP Importance Tooltips ----------
    sample_df = pd.DataFrame([feature_means])  # 1-row fake sample for SHAP ref
    tooltips = compute_feature_importance_for_tooltips(model, sample_df[feature_cols])

    # --------- 2. Categorize all features ----------
    categories = categorize_features(feature_cols)

    # --------- 3. User selects editable feature count ----------
    num_features = st.slider(
        "How many top important features should be editable?",
        min_value=3, max_value=15, value=6
    )

    # Sort features by SHAP importance
    sorted_by_importance = sorted(tooltips.items(), key=lambda x: len(x[1]), reverse=False)
    editable_features = [feat for feat, _ in sorted_by_importance][:num_features]

    st.markdown("### 🧠 Editable Key Features (Top SHAP Influencers)")

    user_input = {}

    # --------- Editable features input section ----------
    for feat in editable_features:
        mean_val = feature_means.get(feat, 0.0)
        std_val = abs(mean_val) * 0.2 if mean_val != 0 else 1.0

        user_input[feat] = st.number_input(
            feat,
            value=float(mean_val),
            min_value=float(mean_val - 3 * std_val),
            max_value=float(mean_val + 3 * std_val),
            help=tooltips.get(feat, "Feature impact unknown.")
        )

    # ---------- Collapsible Advanced Section ----------
    with st.expander("⚙️ Advanced Model Features"):
        st.markdown("These are rarely used or one-hot encoded features.")
        for category, feats in categories.items():
            if category == "Advanced":
                st.markdown(f"#### {category}")

                for feat in feats:
                    default_val = feature_means.get(feat, 0.0)
                    user_input[feat] = st.number_input(
                        feat,
                        value=float(default_val),
                        help=tooltips.get(feat, "No SHAP data.")
                    )

    # ---------- Predict Button ----------
    if st.button("Predict Risk"):

        df_single = pd.DataFrame([user_input])
        result_single, _ = score_dataframe(
            df_single, model, feature_cols, feature_means, threshold
        )

        row = result_single.iloc[0]
        proba = row["default_probability"]
        label = row["predicted_label"]

        risk_text = "🟥 High Risk" if label == 1 else "🟩 Low Risk"

        st.markdown(f"### Prediction: **{risk_text}**")
        st.write(f"**Default Probability:** {proba:.3f}")

        st.markdown("### Full Model Input")
        st.dataframe(result_single)

# with tab_single:
#     st.subheader("Single-Customer Risk Prediction")

#     st.markdown(
#         """
#         Enter key features for a single customer. All other model features will be filled
#         with their typical (mean) values from the training data.

#         You can choose which features to edit in the sidebar (under *Single Prediction Settings*).
#         """
#     )

#     if not manual_features:
#         st.warning(
#             "No manual features selected. Go to the sidebar and choose at least one feature "
#             "to edit under 'Single Prediction Settings'."
#         )
#     else:
#         # Arrange inputs in rows of up to 3 columns
#         user_input = {}

#         n_cols = min(3, len(manual_features))
#         rows = (len(manual_features) + n_cols - 1) // n_cols

#         idx = 0
#         for _ in range(rows):
#             cols = st.columns(n_cols)
#             for col in cols:
#                 if idx >= len(manual_features):
#                     break
#                 feat = manual_features[idx]
#                 idx += 1

#                 mean_val = feature_means.get(feat, 0.0)

#                 # Basic plausible range: mean ± 3 * (20% of |mean|) or fallback
#                 std_val = 0.2 * abs(mean_val) if mean_val != 0 else 1.0
#                 min_v = mean_val - 3 * std_val
#                 max_v = mean_val + 3 * std_val

#                 help_text = FEATURE_DESCRIPTIONS.get(feat, "Model input feature.")

#                 with col:
#                     # number_input enforces min/max but still lets you type;
#                     # that's fine because any out-of-range value is clamped.
#                     user_input[feat] = st.number_input(
#                         label=feat,
#                         value=float(mean_val),
#                         min_value=float(min_v),
#                         max_value=float(max_v),
#                         help=help_text,
#                     )

#         # Advanced: allow overriding any other feature (hidden features)
#         with st.expander("Advanced: override additional (hidden) features", expanded=False):
#             hidden_features = [f for f in feature_cols if f not in manual_features]
#             adv_input = {}
#             if hidden_features:
#                 selected_hidden_feat = st.selectbox(
#                     "Choose a hidden feature to override",
#                     options=hidden_features,
#                     help="These features are normally kept at typical (mean) values.",
#                 )
#                 default_val = feature_means.get(selected_hidden_feat, 0.0)
#                 adv_val = st.number_input(
#                     f"New value for {selected_hidden_feat}",
#                     value=float(default_val),
#                     help=FEATURE_DESCRIPTIONS.get(selected_hidden_feat, "Hidden model feature."),
#                     key=f"adv_{selected_hidden_feat}",
#                 )
#                 if st.button("Apply override for this feature"):
#                     adv_input[selected_hidden_feat] = adv_val
#                     st.info(
#                         f"Override applied: {selected_hidden_feat} = {adv_val}. "
#                         "This will be used in the next prediction."
#                     )

#         if st.button("Predict Risk for This Customer"):
#             # Start from manual inputs
#             full_input = user_input.copy()
#             # Apply any advanced overrides
#             full_input.update(adv_input)

#             df_single = pd.DataFrame([full_input])

#             # Use the existing scoring function
#             result_single, _ = score_dataframe(
#                 df_single, model, feature_cols, feature_means, threshold
#             )
#             row = result_single.iloc[0]
#             proba = row["default_probability"]
#             label = row["predicted_label"]
#             risk_text = (
#                 "High Risk (Default Likely)" if label == 1 else "Low Risk (Default Unlikely)"
#             )

#             st.success(f"Prediction: **{risk_text}**")
#             st.write(f"Default probability: **{proba:.3f}**")
#             st.write("Full scored row:")
#             st.dataframe(result_single)


# ========== TAB 3: BATCH PREDICTION (CSV) ==========
with tab_batch:
    st.subheader("Batch Prediction for CSV")

    uploaded_file = st.file_uploader(
        "Upload a CSV file with customer features (same columns as training features).",
        type=["csv"],
        key="batch_uploader"
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
        else:
            st.success(f"Loaded {len(df_input)} rows from `{uploaded_file.name}`")

            if st.checkbox("Show raw input data", value=False, key="show_raw_batch"):
                st.dataframe(df_input.head())

            # 🔴 IMPORTANT: this uses score_dataframe
            result_df, df_prepared = score_dataframe(
                df_input, model, feature_cols, feature_means, threshold
            )

            st.markdown("### Predictions (first 100 rows)")
            st.dataframe(result_df.head(100))

            total = len(result_df)
            high_risk = int((result_df["predicted_label"] == 1).sum())
            low_risk = total - high_risk
            avg_proba = float(result_df["default_probability"].mean())

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", total)
            col2.metric("High Risk (1)", high_risk)
            col3.metric("Avg Default Probability", f"{avg_proba:.3f}")

            csv_download = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_download,
                file_name="credit_risk_scored.csv",
                mime="text/csv",
            )

            # 🔴 IMPORTANT: store prepared data for SHAP tab
            st.session_state["last_scored_prepared"] = df_prepared
    else:
        st.info("Upload a CSV to generate batch predictions.")



# ========== TAB 4: SHAP EXPLAINABILITY ==========
with tab_explain:
    st.subheader("SHAP Explainability")

    st.markdown(
        """
SHAP helps explain which features contribute most to the model's predictions.

For performance and compatibility, explanations here are limited to **tree-based models**
such as Random Forest and LightGBM.
        """
    )

    # Only allow SHAP for tree-based models
    is_tree_model = any(
        key in selected_model_name.lower()
        for key in ["rf", "random", "lgbm", "lightgbm"]
    )

    if not is_tree_model:
        st.warning(
            "SHAP explanations are currently enabled only for tree-based models "
            "(Random Forest / LightGBM). Select one of those models in the sidebar."
        )
    else:
        if "last_scored_prepared" not in st.session_state:
            st.info(
                "To see SHAP explanations, first go to **Batch Prediction** and upload a CSV. "
                "The app will reuse that preprocessed data here."
            )
        else:
            X_for_shap = st.session_state["last_scored_prepared"]
            # sample to speed up
            if len(X_for_shap) > 500:
                X_sample = X_for_shap.sample(500, random_state=42)
            else:
                X_sample = X_for_shap

            st.markdown(f"Using a sample of **{len(X_sample)}** rows for SHAP computation.")

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                st.markdown("### Global Feature Importance (SHAP Summary Plot)")
                st.markdown("### Global Feature Importance (SHAP Summary Plot)")

                # Let SHAP draw on its own figure
                shap.summary_plot(
                    shap_values[1] if isinstance(shap_values, list) else shap_values,
                    X_sample,
                    show=False
                )

                # Get the current figure created by SHAP
                fig1 = plt.gcf()
                st.pyplot(fig1, clear_figure=True)


            except Exception as e:
                st.error(f"SHAP encountered an error: {e}")
