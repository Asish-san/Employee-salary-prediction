import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

@st.cache_data
def load_rank():
    with open("model_rank.txt", "r") as f:
        return float(f.read())

# Load model components
try:
    model_data = load_model()
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    score = load_rank()
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
    background: #f5f7fa;
}
.main-header {
    background: radial-gradient(circle at 20% 40%, #f59e42 60%, #f43f5e 100%);
    padding: 2.5rem 1rem;
    border-radius: 18px;
    margin-bottom: 2.5rem;
    text-align: center;
    color: #fff;
    box-shadow: 0 8px 24px rgba(244,63,94,0.08);
    border: 2px solid #f59e42;
}
.metric-card {
    background: linear-gradient(120deg, #f59e42 60%, #f43f5e 100%);
    padding: 1.8rem 1rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(245,158,66,0.07);
    margin-bottom: 1.8rem;
    color: #fff;
    border: 1.5px solid #f43f5e;
}
.metric-card h2 {
    font-size: 2.2rem;
    color: #fff;
    margin: 0.7rem 0 0;
    letter-spacing: 2px;
    text-shadow: 0 2px 8px #f59e42;
}
.prediction-card {
    background: repeating-linear-gradient(135deg, #f43f5e 0px, #f59e42 40px, #f43f5e 80px);
    color: #fff;
    padding: 2.2rem 1.2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 24px rgba(244,63,94,0.15);
    margin: 2.2rem 0;
    border: 2px dashed #fff;
}
.stButton > button {
    background: linear-gradient(90deg, #f59e42 60%, #f43f5e 100%);
    color: #fff;
    border: none;
    padding: 1rem 2.2rem;
    border-radius: 40px;
    font-size: 1.15rem;
    font-weight: 700;
    width: 100%;
    transition: box-shadow 0.2s, transform 0.2s;
    box-shadow: 0 2px 8px rgba(244,63,94,0.12);
    letter-spacing: 1px;
}
.stButton > button:hover {
    transform: scale(1.04) translateY(-2px);
    box-shadow: 0 8px 24px rgba(245,158,66,0.18);
    background: linear-gradient(90deg, #f43f5e 60%, #f59e42 100%);
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

    """, unsafe_allow_html=True)

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
st.markdown("""
<div style="text-align: center; font-size: 0.9rem; color: #775;">
     Created by Asish Kumar | 🕸️ Streamlit
</div>
""", unsafe_allow_html=True)
