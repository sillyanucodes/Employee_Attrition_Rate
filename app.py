import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

# Title
st.title("🧩 Employee Attrition Prediction Dashboard")

st.write("""
This dashboard allows HR teams to:
- View attrition data distribution
- Predict the likelihood of an employee leaving
""")

# Load Model
model = pickle.load(open("C:\\Users\\Anushka\\Downloads\\Employee_Attrition_Project\\rf_model.pkl", "rb"))
# Load Data
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["EDA", "Predict Attrition"])

# --------------- EDA SECTION ---------------
if option == "EDA":
    st.header("📊 Exploratory Data Analysis")

    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df.head())

    st.subheader("Attrition Distribution")
    attr_counts = df['Attrition'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(attr_counts, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("Attrition by Department")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Department', hue='Attrition', data=df, ax=ax2)
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    st.subheader("Attrition by Overtime")
    fig3, ax3 = plt.subplots()
    sns.countplot(x='OverTime', hue='Attrition', data=df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Monthly Income vs Attrition")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=ax4)
    st.pyplot(fig4)

# --------------- PREDICTION SECTION ---------------
elif option == "Predict Attrition":
    st.header("🔮 Predict Employee Attrition Risk")

    st.write("Input employee details below:")

    age = st.slider("Age", 18, 60, 30)
    distance_from_home = st.slider("Distance From Home (km)", 1, 30, 5)
    education = st.slider("Education (1-5)", 1, 5, 3)
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
    num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
    overtime = st.selectbox("OverTime", ['Yes', 'No'])
    total_working_years = st.slider("Total Working Years", 0, 40, 5)
    years_at_company = st.slider("Years at Company", 0, 20, 3)
    years_in_current_role = st.slider("Years in Current Role", 0, 15, 2)

    # Encoding
    marital_status_encoded = {'Single': 2, 'Married': 1, 'Divorced': 0}[marital_status]
    overtime_encoded = 1 if overtime == 'Yes' else 0

    input_data = np.array([[age, distance_from_home, education, environment_satisfaction,
                            job_satisfaction, marital_status_encoded, monthly_income,
                            num_companies_worked, overtime_encoded, total_working_years,
                            years_at_company, years_in_current_role]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ High Risk: This employee is likely to leave. (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"✅ Low Risk: This employee is likely to stay. (Probability: {prediction_proba:.2f})")
