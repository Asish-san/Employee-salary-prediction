import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import time

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Load model components
# Real-time currency conversion (USD to INR)
def get_usd_to_inr_rate():
    """
    Fetches the latest USD to INR conversion rate from a public API and caches it for 1 hour.
    """
    if not hasattr(get_usd_to_inr_rate, "cache") or time.time() - get_usd_to_inr_rate.cache_time > 3600:
        try:
            response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
            data = response.json()
            rate = data["rates"]["INR"]
            get_usd_to_inr_rate.cache = rate
            get_usd_to_inr_rate.cache_time = time.time()
        except Exception:
            # Fallback to previous cached value or default
            rate = getattr(get_usd_to_inr_rate, "cache", 83)
    else:
        rate = get_usd_to_inr_rate.cache
    return rate
try:
    model_data = load_model()
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
except:
    st.error("❗ Please retrain the model first by running all cells in the notebook.")
    st.stop()

# Page configuration
st.set_page_config(page_title="Employee Salary Predicton", page_icon="💼", layout="wide")

# 🌈 Custom CSS Styling (Creative Redesign)
st.markdown("""
<style>
body {
    font-family: 'Montserrat', 'Segoe UI', sans-serif;
    background: linear-gradient(120deg, #f5f7fa 60%, #f59e42 100%);
    min-height: 100vh;
}
.main-header {
    background: linear-gradient(90deg, #f43f5e 0%, #f59e42 100%);
    padding: 3rem 1.5rem 2rem 1.5rem;
    border-radius: 24px;
    margin-bottom: 2.5rem;
    text-align: center;
    color: #fff;
    box-shadow: 0 12px 32px rgba(244,63,94,0.12);
    border: 3px solid #fff;
    animation: popIn 1s cubic-bezier(.68,-0.55,.27,1.55) 1;
}
@keyframes popIn {
    0% { transform: scale(0.8); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}
.metric-card {
    background: linear-gradient(135deg, #f59e42 60%, #f43f5e 100%);
    padding: 2rem 1.2rem;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(245,158,66,0.12);
    margin-bottom: 2rem;
    color: #fff;
    border: 2px solid #fff;
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: scale(1.03) rotate(-2deg);
    box-shadow: 0 16px 32px rgba(244,63,94,0.18);
}
.metric-card h2 {
    font-size: 2.5rem;
    color: #fff;
    margin: 1rem 0 0;
    letter-spacing: 2px;
    text-shadow: 0 2px 12px #f59e42;
}
.prediction-card {
    background: linear-gradient(135deg, #f43f5e 0%, #f59e42 100%);
    color: #fff;
    padding: 2.5rem 1.5rem;
    border-radius: 28px;
    text-align: center;
    box-shadow: 0 16px 32px rgba(244,63,94,0.18);
    margin: 2.5rem 0;
    border: 3px dashed #fff;
    animation: fadeIn 1.2s cubic-bezier(.68,-0.55,.27,1.55) 1;
}
.github-link {
    display: none !important;
}
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
.prediction-card h1 {
    font-size: 3rem;
    margin: 1rem 0 0.5rem 0;
    color: #fff;
    text-shadow: 0 2px 12px #f59e42;
}
.prediction-card h2 {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    color: #fff;
}
.prediction-card p {
    font-size: 1.2rem;
    margin-top: 0.5rem;
    color: #fff;
}
.stButton > button {
    background: linear-gradient(90deg, #f43f5e 0%, #f59e42 100%);
    color: #fff;
    border: none;
    padding: 1.2rem 2.5rem;
    border-radius: 50px;
    font-size: 1.2rem;
    font-weight: 700;
    width: 100%;
    transition: box-shadow 0.2s, transform 0.2s;
    box-shadow: 0 4px 16px rgba(244,63,94,0.18);
    letter-spacing: 1.5px;
    cursor: pointer;
    animation: popIn 1s cubic-bezier(.68,-0.55,.27,1.55) 1;
}
.stButton > button:hover {
    transform: scale(1.06) translateY(-2px);
    box-shadow: 0 12px 32px rgba(245,158,66,0.22);
    background: linear-gradient(90deg, #f59e42 0%, #f43f5e 100%);}
.salary-animate {
    animation: popSalary 1.2s cubic-bezier(.17,.67,.83,.67) forwards;
}
.stTextInput > div > input {
    background: #fffbe6;
    border-radius: 12px;
    border: 2px solid #f59e42;
    padding: 0.7rem 1rem;
    font-size: 1.1rem;
    color: #f43f5e;
    box-shadow: 0 2px 8px rgba(245,158,66,0.08);
}
.stSlider > div {
    background: #fffbe6;
    border-radius: 12px;
    border: 2px solid #f59e42;
    box-shadow: 0 2px 8px rgba(245,158,66,0.08);
}
.stSelectbox > div {
    background: #fffbe6;
    border-radius: 12px;
    border: 2px solid #f59e42;
    box-shadow: 0 2px 8px rgba(245,158,66,0.08);
}
</style>
""", unsafe_allow_html=True)

# Main Heading
st.markdown("""
<div class="main-header">
    <h1>🤵 Employee Salary Prediction</h1>
    <p>Estimate salaries based on input feature</p>
</div>
""", unsafe_allow_html=True)

# 🧾 Form Inputs and Info Panel
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📋 Input Employee Details")
    with st.form("salary_predictior"):
        age = st.slider("👤 Age", 18, 70, 30)
        gender = st.selectbox("⚧ Gender", sorted(['Male', 'Female']))
        education = st.selectbox("🎓 Education Level", sorted(["Bachelor's", "Master's", "PhD"]))
        job_title = st.selectbox("💼 Job Title", sorted([
            'Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate',
            'Director', 'Marketing Analyst', 'Product Manager', 'Sales Manager',
            'Marketing Coordinator'
        ]))
        experience = st.slider("📈 Years of Experience", 0.0, 50.0, 5.0, step=0.5)

        submitted = st.form_submit_button("🔮 Predict Salary")
    # 🎈 Balloon effect when salary is predicted
    st.balloons()

    st.markdown("### 🔧 Features Used")
    for feature in feature_names:
        st.markdown(f"- {feature}")

    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Fill all the details.
    2. Click 'Predict Salary'.
    3. View predicted salary with insights.
    """)

# 🔮 Prediction
if submitted:
    try:
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education],
            'Job Title': [job_title],
            'Years of Experience': [experience]
        })

        # Encode categorical features
        input_encoded = input_data.copy()
        for col in ['Gender', 'Education Level', 'Job Title']:
            if col in label_encoders:
                input_encoded[col] = label_encoders[col].transform(input_data[col])

        # Match column order
        input_final = input_encoded[feature_names]

        # Scale input
        input_scaled = scaler.transform(input_final)

        # Predict salary
        prediction = model.predict(input_scaled)[0]

        # 🌟 Show Prediction (USD and INR)
        usd_to_inr = get_usd_to_inr_rate()
        inr_prediction = prediction * usd_to_inr

        st.markdown(
            f'<div class="prediction-card"><h2>💰 Predicted Annual Salary</h2><h1>USD ${prediction:,.0f}</h1><h1>INR ₹{inr_prediction:,.0f}</h1><p>Based on the employee details</p><p style="font-size:0.9rem;color:#f59e42;">(Live USD to INR rate: {usd_to_inr:.2f})</p></div>',
            unsafe_allow_html=True
        )

        # 📋 Input Summary
        st.markdown("### 📌 Summary of Inputs")
        s1, s2 = st.columns(2)
        with s1:
            st.write(f"**👤 Age:** {age}")
            st.write(f"**⚧ Gender:** {gender}")
            st.write(f"**🎓 Education:** {education}")
        with s2:
            st.write(f"**💼 Job Title:** {job_title}")
            st.write(f"**📈 Experience:** {experience} years")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Ensure the model was trained using these input features.")

# 🔻 Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.9rem; color: #775;'>Created by Asish Kumar | Streamlit Web App</div>", unsafe_allow_html=True)
