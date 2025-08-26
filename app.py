
import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline (must be saved separately)
MODEL_PATH = "LightGBM_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💰", layout="centered")
st.title("💰 Employee Salary Prediction App")
st.markdown("Predict employee **salary** based on demographic and company details.")

# --- Input fields ---
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 70, 30)
    age_joined = st.number_input("Age when joined", 18, 70, 25)
    years_company = st.number_input("Years in company", 0, 40, 5)
    prior_exp = st.number_input("Prior years experience", 0, 40, 2)
with col2:
    annual_bonus = st.number_input("Annual Bonus", 0, 100000, 5000)
    company = st.selectbox("Company", ["A", "B", "C"])
    department = st.selectbox("Department", ["HR", "Tech", "Sales"])
    employment = st.selectbox("Employment type", ["full_time", "part_time", "contractor"])

# --- Build single-row DataFrame ---
input_df = pd.DataFrame([{
    "age": age,
    "age_when_joined": age_joined,
    "years_in_the_company": years_company,
    "prior_years_experience": prior_exp,
    "annual_bonus": annual_bonus,
    "company": company,
    "department": department,
    "full_time": 1 if employment == "full_time" else 0,
    "part_time": 1 if employment == "part_time" else 0,
    "contractor": 1 if employment == "contractor" else 0,
}])

if st.button("🔮 Predict Salary"):
    salary_pred = model.predict(input_df)[0]
    st.success(f"Estimated Salary: **${salary_pred:,.0f}**")

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ by Emr7y | Model: LightGBM | Data: Kaggle")
