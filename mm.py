import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

st.markdown(
    """
    <style>

    [data-testid="stFileUploader"] button {
        background-color: #FF8C00 !important;
        color: #0E1117 !important;
        border-radius: 8px;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label {
        color: #0E1117 !important;
        font-weight: 500;
    }
     [data-testid="stFileUploader"] {
        background-color: #0E1117 !important;
        border: 2px dashed #FF8C00 !important;
        border-radius: 14px;
        padding: 18px;
    }

    /* Main app background */
    .stApp {
        background-color: #FFF4DE;
    }

    /* Header visible (DO NOT HIDE) */
    header[data-testid="stHeader"] {
        background-color: #FFF4DE;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }

    [data-testid="stSidebar"] * {
        color: #FFF3E0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.header("Behavior-Driven Addiction Risk Analysis")
st.write("Upload the Dataset to perform EDA and Machine Learning")

st.sidebar.header("Upload Dataset")
upload_file = st.sidebar.file_uploader("Upload The CSV File", type="csv")

if upload_file is not None:
    df = pd.read_csv(upload_file)

    st.subheader("Dataset Preview")
    st.write("File uploaded successfully")

    df.replace(["?", "NA"], np.nan, inplace=True)
    df["daily_screen_time"] = pd.to_numeric(df["daily_screen_time"], errors="coerce")
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
    df["dependency_score"] = pd.to_numeric(df["dependency_score"], errors="coerce")

    df["daily_screen_time"] = df["daily_screen_time"].fillna(df["daily_screen_time"].median())
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
    df["mood_level"] = df["mood_level"].fillna(df["mood_level"].mode()[0])
    df["profession"] = df["profession"].fillna(df["profession"].mode()[0])

    df.loc[df["age_group"].isna() & (df['age'].between(10, 19)), "age_group"] = "Teen"
    df.loc[df["age_group"].isna() & (df["age"].between(20, 29)), "age_group"] = "Young Adult"
    df.loc[df["age_group"].isna() & (df["age"] >= 30), "age_group"] = "Adult"

    st.subheader("Missing Values Summary")
    st.write(df.isna().sum())

    rows = st.slider("Select number of rows to display", 5, 50, 5)
    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(rows))

    st.subheader("Exploratory Data Analysis (EDA)")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="age_group", hue="addiction_risk", palette="inferno", ax=ax[0])
    ax[0].set_title("Addiction Risk Distribution by Age Group")
    ax[0].set_xlabel("Age Group")
    ax[0].set_ylabel("Count")

    avg_sleep = df.groupby("addiction_risk")["sleep_hours"].mean()
    avg_sleep.plot(kind="barh", color="#FF8C00", ax=ax[1])
    ax[1].set_title("Average Sleep Hours by Addiction Risk")
    ax[1].set_xlabel("Average Sleep Hours")
    ax[1].set_ylabel("Addiction Risk")

    st.pyplot(fig)

    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))

    dc = df["detox_needed"].value_counts()
    ax2[0].pie(dc, labels=dc.index, autopct="%1.1f%%", startangle=90, colors=["red", "orange"])
    ax2[0].set_title("Distribution of Users by Detox Requirement")

    avg_screen_time = df.groupby("age_group")["daily_screen_time"].mean()
    avg_screen_time.plot(kind="bar", ax=ax2[1], color="#FF8C00")
    ax2[1].set_title("Average Daily Screen Time by Age Group")
    ax2[1].set_xlabel("Age Group")
    ax2[1].set_ylabel("Average Screen Time (Hours)")

    st.pyplot(fig2)

    df["night_usage_bin"] = pd.cut(df['night_usage_minutes'], bins=range(0, 241, 30))
    avg_sleep = df.groupby("night_usage_bin")["sleep_hours"].mean().reset_index()
    avg_sleep = avg_sleep.sort_values("night_usage_bin")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.plot(avg_sleep["night_usage_bin"].astype(str),
             avg_sleep["sleep_hours"], marker="o", color="#FF8C00")
    ax3.set_title("Night Usage vs Average Sleep Hours")
    ax3.set_xlabel("Night Usage Minutes")
    ax3.set_ylabel("Average Sleep Hours")
    st.pyplot(fig3)

    st.subheader("Distribution of Numerical Features")

    num_cols = ["daily_screen_time", "night_usage_minutes",
                "notifications_count", "phone_pickups",
                "sleep_hours", "dependency_score"]

    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.kdeplot(data=df, x=col, fill=True, color="#FF8C00", ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    df[num_cols].boxplot(ax=ax)
    ax.set_title("Outlier Analysis using Boxplot")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(df[num_cols].corr(), annot=True, ax=ax, cmap="coolwarm")
    ax.set_title("Correlation Heatmap of Numerical Features")
    st.pyplot(fig)

    st.subheader("Encoding Categorical Variables")

    df["addiction_risk_enc"] = df["addiction_risk"].map({"Low": 0, "Medium": 1, "High": 2})
    df["detox_needed_enc"] = df["detox_needed"].map({"No": 0, "Yes": 1})

    st.write(df[["addiction_risk", "addiction_risk_enc",
                 "detox_needed", "detox_needed_enc"]].head())

    features = ["age", "daily_screen_time", "night_usage_minutes",
                "notifications_count", "phone_pickups",
                "sleep_hours", "dependency_score"]

    selected_features = st.multiselect(
        "Select Features (Model Input)", features, default=features)

    target = st.selectbox(
        "Select Target Variable (Prediction Output)",
        ["addiction_risk_enc", "detox_needed_enc"]
    )

    if st.button("Train & Evaluate Models"):
        x = df[selected_features]
        y = df[target]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=1)

        ss = StandardScaler()
        x_trainss = ss.fit_transform(x_train)
        x_testss = ss.transform(x_test)

        lr = LogisticRegression()
        lr.fit(x_trainss, y_train)

        acc = lr.score(x_testss, y_test)
        st.success(f"Logistic Regression Accuracy: {acc:.2%}")
        st.text("Logistic Regression Classification Report")
        st.text(classification_report(y_test, lr.predict(x_testss)))

        rf = RandomForestClassifier(n_estimators=200, random_state=2)
        rf.fit(x_train, y_train)

        acc = rf.score(x_test, y_test)
        st.success(f"Random Forest Accuracy: {acc:.2%}")
        st.text("Random Forest Classification Report")
        st.text(classification_report(y_test, rf.predict(x_test)))

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(rf, x, y, cv=5, scoring="f1_macro")

        st.subheader("Cross-Validation Performance")
        st.write("F1 Scores (5-Fold):", scores)
        st.write("Mean F1 Score:", scores.mean())

else:
    st.info("Please upload a CSV file to begin analysis.")

