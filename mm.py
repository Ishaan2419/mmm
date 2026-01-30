import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

st.set_page_config(page_title="Employee Burnout Analytics", layout="wide")

st.title("🧠 Employee Workload & Burnout Risk Analytics")

# Upload file
file = st.file_uploader("Upload Employee Burnout CSV", type=["csv"])

if file is None:
    st.info("Upload the CSV file to begin analysis.")
    st.stop()

df = pd.read_csv(file)

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📄 Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# -------------------------------
# EDA
# -------------------------------
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("Burnout Distribution")
    fig, ax = plt.subplots()
    df["burnout_flag"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

with col2:
    st.write("Stress Level Distribution")
    fig, ax = plt.subplots()
    df["self_reported_stress"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

st.write("### Average Weekly Hours by Burnout Status")
st.dataframe(df.groupby("burnout_flag")["weekly_hours"].mean())

st.write("### Burnout Rate by Department")
burnout_rate = df.groupby("department")["burnout_flag"].apply(
    lambda x: (x == "Yes").mean()
)

fig, ax = plt.subplots()
burnout_rate.plot(kind="bar", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------------
# MACHINE LEARNING
# -------------------------------
st.subheader("🤖 Machine Learning – Burnout Prediction")

ml_df = df.copy()
ml_df["burnout_flag"] = ml_df["burnout_flag"].map({"No": 0, "Yes": 1})
ml_df["weekend_work"] = ml_df["weekend_work"].map({"No": 0, "Yes": 1})
ml_df["self_reported_stress"] = ml_df["self_reported_stress"].map(
    {"Low": 0, "Medium": 1, "High": 2}
)

features = [
    "weekly_hours",
    "overtime_hours",
    "meetings_per_week",
    "weekend_work",
    "sick_leaves",
    "self_reported_stress"
]

X = ml_df[features]
y = ml_df["burnout_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("### 📋 Model Evaluation")
st.text(classification_report(y_test, y_pred))

# -------------------------------
# PREDICTION PREVIEW
# -------------------------------
st.subheader("🔍 Sample Predictions")

ml_df["prediction"] = model.predict(X)
ml_df["prediction_label"] = ml_df["prediction"].map({0: "No Burnout", 1: "Burnout"})

st.dataframe(
    ml_df[
        [
            "emp_id",
            "weekly_hours",
            "overtime_hours",
            "self_reported_stress",
            "prediction_label"
        ]
    ].head(20),
    use_container_width=True
)
