import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Corporate Expense Fraud Analytics",
    layout="wide"
)

st.title("💼 Corporate Expense Fraud Analytics")
st.write("EDA + Explainable Machine Learning (Audit-Correct Logic)")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Expense CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Upload the dataset to begin.")
    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------
df.replace(["?", "NA"], np.nan, inplace=True)

# Clean amount
df["amount"] = df["amount"].astype(str).str.replace("₹", "")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Clean date
df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")

# Logical defaults
df["receipt_attached"].fillna("No", inplace=True)
df["audit_flag"].fillna("Clear", inplace=True)
df["violation_type"].fillna("None", inplace=True)

# --------------------------------------------------
# 🚨 FORCE REMOVE ANY OLD DERIVED COLUMNS
# --------------------------------------------------
for col in ["over_limit", "missing_receipt"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# --------------------------------------------------
# ✅ CORRECT FEATURE ENGINEERING (AUTHORITATIVE)
# --------------------------------------------------
df["over_limit"] = df["violation_type"].eq("Over Limit")
df["missing_receipt"] = df["violation_type"].eq("No Receipt")

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
st.subheader("📄 Cleaned Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# --------------------------------------------------
# VALIDATION CHECK (VERY IMPORTANT)
# --------------------------------------------------
st.subheader("✅ Logic Validation Check")
validation = df.groupby("violation_type")[["over_limit", "missing_receipt"]].mean()
st.dataframe(validation, use_container_width=True)

# --------------------------------------------------
# EDA SECTION
# --------------------------------------------------
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("Expense Type Distribution")
    fig, ax = plt.subplots()
    df["expense_type"].value_counts().plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.write("Audit Flag Distribution")
    fig, ax = plt.subplots()
    df["audit_flag"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# Over-limit rate
st.subheader("📈 Over-Limit Rate by Expense Type")
over_rate = df.groupby("expense_type")["over_limit"].mean()

fig, ax = plt.subplots()
over_rate.plot(kind="bar", ax=ax)
ax.set_ylabel("Over-Limit Rate")
plt.xticks(rotation=45)
st.pyplot(fig)

# Missing receipt vs audit
st.subheader("📉 Missing Receipt vs Audit Outcome")
receipt_ct = pd.crosstab(df["missing_receipt"], df["audit_flag"], normalize="index")

fig, ax = plt.subplots()
receipt_ct.plot(kind="bar", stacked=True, ax=ax)
ax.set_ylabel("Proportion")
st.pyplot(fig)

# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------
st.subheader("🤖 Machine Learning – Fraud Prediction")

ml_df = df.dropna(subset=["amount", "audit_flag"])

ml_df["audit_flag"] = ml_df["audit_flag"].map({
    "Clear": 0,
    "Suspect": 1
})

features = [
    "amount",
    "policy_limit",
    "over_limit",
    "missing_receipt"
]

X = ml_df[features]
y = ml_df["audit_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
st.write("### 📋 Model Evaluation")
st.text(classification_report(y_test, y_pred))

# Feature importance
st.write("### 🔍 Feature Importance")
importance = pd.Series(model.coef_[0], index=features)

fig, ax = plt.subplots()
importance.sort_values().plot(kind="barh", ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# PREDICTION PREVIEW
# --------------------------------------------------
st.subheader("🔎 Sample Predictions")

ml_df["prediction"] = model.predict(X)
ml_df["prediction_label"] = ml_df["prediction"].map({
    0: "Clear",
    1: "Suspect"
})

st.dataframe(
    ml_df[
        [
            "emp_id",
            "expense_type",
            "amount",
            "policy_limit",
            "violation_type",
            "over_limit",
            "missing_receipt",
            "prediction_label"
        ]
    ].head(25),
    use_container_width=True
)
