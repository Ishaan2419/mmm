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

st.title("Machine Learning App")
st.write("Upload the Dataset")


st.sidebar.header("Upload Dataset")
upload_file=st.sidebar.file_uploader("Upload The CSV File",type="csv")


if upload_file is not None:
    df=pd.read_csv(upload_file)
    st.subheader(" Data Preview")
    st.write("File Uploaded successfully")

    df.replace(["?","NA"],np.nan,inplace=True)
    df["daily_screen_time"]=pd.to_numeric(df["daily_screen_time"],errors="coerce")
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
    df["dependency_score"]=pd.to_numeric(df["dependency_score"],errors="coerce")
    df["daily_screen_time"] = df["daily_screen_time"].fillna(df["daily_screen_time"].median())
    df["sleep_hours"]=df["sleep_hours"].fillna(df["sleep_hours"].median())
    df['mood_level']=df['mood_level'].fillna(df['mood_level'].mode()[0])
    df["profession"]=df["profession"].fillna(df["profession"].mode()[0])

    df.loc[df["age_group"].isna() & (df['age'].between(10,19)),"age_group"]="Teen"
    df.loc[df["age_group"].isna() & (df["age"].between(20,29)),"age_group"]="Young Adult"
    df.loc[df["age_group"].isna() & (df["age"]>=30),"age_group"]="Adult"

   

    st.subheader("Missing Values")
    st.write(df.isna().sum())
    
    rows = st.slider("Select number of rows to display", 5, 50, 5)
    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(rows))

    st.subheader("Analysis")


    fig,ax=plt.subplots(1,2,figsize=(14,5))
    sns.countplot(data=df,x="age_group",hue="addiction_risk",palette="inferno",ax=ax[0])
    ax[0].set_title("Addiction risk by Age")

    avg_sleep = df.groupby("addiction_risk")["sleep_hours"].mean()
    avg_sleep.plot(kind="barh",color="#FF8C00",ax=ax[1])
    ax[1].set_title("Addiction risk by sleeping hours")
    
    st.pyplot(fig)

    fig2,ax2=plt.subplots(1,2,figsize=(14,5))
    dc=df["detox_needed"].value_counts()
    ax2[0].pie(dc,labels=dc.index,autopct="%1.1f%%",startangle=90,colors=["red","orange"])
    ax2[0].set_title("Distribution of Users by Detox Requirement")

    df["night_usage_bin"]=pd.cut(df['night_usage_minutes'],bins=range(0,241,30))
    avg_sleep=(df.groupby("night_usage_bin")["sleep_hours"].mean().reset_index())
    avg_sleep = avg_sleep.sort_values("night_usage_bin")
    ax2[1].plot(
    avg_sleep["night_usage_bin"].astype(str),
    avg_sleep["sleep_hours"],
    marker="o"
    )
    ax2[1].set_title("Night Usage vs Sleep Hours")
    ax2[1].set_xlabel("Night Usage Minutes")
    ax2[1].set_ylabel("Average Sleep Hours")
    st.pyplot(fig2)


    st.subheader("Average Screen Time by Age Group")
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    avg_screen_time = df.groupby("age_group")["daily_screen_time"].mean()
    avg_screen_time.plot(kind="bar",ax=ax3,color="#4C72B0")
    ax3.set_ylabel("Average Screen Time (Hours)")
    ax3.set_xlabel("Age Group")
    ax3.set_title("Average Screen Time by Age Group")
    st.pyplot(fig3)

    st.subheader("Distribution of Numerical Features (KDE)")
    num_cols = ["daily_screen_time","night_usage_minutes","notifications_count","phone_pickups","sleep_hours","dependency_score"]
    
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.kdeplot(data=df,x=col,fill=True,ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        st.pyplot(fig)

    fig,ax=plt.subplots(figsize=(14,5))
    df[num_cols].boxplot(ax=ax)
    st.pyplot(fig)

    fig,ax=plt.subplots(figsize=(14,5))
    sns.heatmap(df[num_cols].corr(),annot=True,ax=ax)
    st.pyplot(fig)

    st.subheader("Encoding Categorical Variables")
    df["addiction_risk_enc"] = df["addiction_risk"].map({"Low": 0, "Medium": 1, "High": 2})
    
    df["detox_needed_enc"] = df["detox_needed"].map({"No": 0, "Yes": 1})
    
    st.write(df[["addiction_risk", "addiction_risk_enc","detox_needed", "detox_needed_enc"]].head())
    features = ["age","daily_screen_time","night_usage_minutes","notifications_count","phone_pickups","sleep_hours","dependency_score"]

    
    selected_features = st.multiselect("Select Features (Input)",features,default=features)
    target = st.selectbox("Select Target (What to predict)", df.columns, index=len(df.columns)-1)

    if st.button("Train & Predict"):
        x=df[selected_features]
        y=df[target]

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_states=1)
        
        ss=StandardScaler()
        x_train=ss.fit_transform(x_train)
        x_test=ss.transform(x_test)

        lr=LogisticRegression()
        lr.fit(x_train,y_train)

        acc = model.score(x_test, y_test)
        st.success(f"Model Accuracy: {acc:.2%}")
        
        y_pred=lr.predict(x_test)
        print(classification_report(y_test,y_pred))



        rf = RandomForestClassifier( n_estimators=200, random_state=2) 
        rf.fit(x_train, y_train)

        acc = model.score(x_test, y_test)
        st.success(f"Model Accuracy: {acc:.2%}")

        
        y_pred_rf = rf.predict(x_test)
        print(classification_report(y_test,y_pred_rf))

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(rf, x, y, cv=5, scoring="f1_macro")
        scores, scores.mean()
    



else:
    st.info("Please upload a CSV file")





    
    
    
