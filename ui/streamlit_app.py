import streamlit as st
import requests

st.set_page_config(
    page_title="Explainable Loan Default Predictor",
    layout="wide"
)

st.title("Explainable Loan Default Prediction System")

st.header("Applicant Financial Information")

credit_income = st.number_input("Credit-to-Income Ratio")
annuity_income = st.number_input("Annuity-to-Income Ratio")
age = st.number_input("Age (Years)")
employment = st.number_input("Employment Duration (Years)")

if st.button("Analyze Risk"):
    
    input_data = {
        "CREDIT_TO_INCOME": credit_income,
        "ANNUITY_TO_INCOME": annuity_income,
        "AGE_YEARS": age,
        "EMPLOYMENT_YEARS": employment
    }

    predict_response = requests.post(
        "http://localhost:8000/predict",
        json=input_data
    )

    explain_response = requests.post(
        "http://localhost:8000/explain",
        json=input_data
    )

    if predict_response.status_code == 200:

        prediction_data = predict_response.json()
        explanation_data = explain_response.json()
        
        st.subheader("Prediction Result")
        st.write("Risk Category:", explanation_data["risk_category"])
        prob = explanation_data["probability"] * 100

        if prob < 0.01:
            st.write("Probability of Default: < 0.01%")
        else:
            st.write(f"Probability of Default: {prob:.2f}%")



        
        st.subheader("Text Explanation")
        st.write(explanation_data["text_explanation"])
        
        st.subheader("Top Risk-Increasing Factors")
        st.json(explanation_data["structured_explanation"]["top_risk_increasing_features"])
        
        st.subheader("Top Risk-Reducing Factors")
        st.json(explanation_data["structured_explanation"]["top_risk_reducing_features"])
