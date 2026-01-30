import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Expense Fraud Analytics", layout="wide")

st.title("💼 Corporate Expense Fraud Analytics")
st.write("EDA + Machine Learning Dashboard")

# -------------------------
# Upload Dataset
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Expense CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------
    # DATA CLEANING
    # -------------------------
    df.replace(["?", "NA"], np.nan, inplace=True)

    df["amount"] = df["amount"].astype(str).str.replace("₹", "")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")

    df["receipt_attached"].fillna("No", inplace=True)
    df["audit_flag"].fillna("Clear", inplace=True)
    df["violation_type"].fillna("None", inplace=True)

    # Feature Engineering
    df["over_limit"] = df["amount"] > df["policy_limit"]
    df["missing_receipt"] = df["receipt_attached"] == "No"

    # -------------------------
    # RAW DATA VIEW
    # -------------------------
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head(20))

    # -------------------------
    # EDA SECTION
    # -------------------------
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Expense Type Distribution")
        fig1, ax1 = plt.subplots()
        df["expense_type"].value_counts().plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.write("Audit Flag Distribution")
        fig2, ax2 = plt.subplots()
        df["audit_flag"].value_counts().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

    st.subheader("📈 Policy Violation by Expense Type")
    violation_rate = df.groupby("expense_type")["over_limit"].mean()

    fig3, ax3 = plt.subplots()
    violation_rate.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Violation Rate")
    st.pyplot(fig3)

    # -------------------------
    # MACHINE LEARNING
    # -------------------------
    st.subheader("🤖 Fraud Prediction (Machine Learning)")

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

    st.write("### 📋 Model Evaluation")
    st.text(classification_report(y_test, y_pred))

    # -------------------------
    # PREDICTION OUTPUT
    # -------------------------
    st.subheader("🔍 Sample Predictions")

    ml_df["prediction"] = model.predict(X)
    ml_df["prediction_label"] = ml_df["prediction"].map({
        0: "Clear",
        1: "Suspect"
    })

    st.dataframe(
        ml_df[
            ["emp_id", "amount", "policy_limit", "over_limit",
             "missing_receipt", "prediction_label"]
        ].head(20)
    )

else:
    st.info("👆 Please upload the CSV file to start analysis.")
