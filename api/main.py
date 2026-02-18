from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from src.model_explainability import generate_structured_explanation
from src.text_explanation_engine import generate_text_explanation

app = FastAPI(title="Explainable Loan Default API")

# 🔹 Load model artifact
artifact = joblib.load("models/final_model_locked.pkl")

model = artifact["model"]
threshold = artifact["threshold"]

feature_names = joblib.load("models/feature_names.pkl")

import shap

explainer = shap.LinearExplainer(
    model,
    np.zeros((1, len(feature_names)))
)

# 🔹 Input Schema
class ApplicantData(BaseModel):
    CREDIT_TO_INCOME: float
    ANNUITY_TO_INCOME: float
    AGE_YEARS: float
    EMPLOYMENT_YEARS: float


# 🔹 Risk Categorization
def categorize_risk(probability: float):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"

def build_full_feature_vector(data: ApplicantData):
    """
    Reconstruct full 219 feature vector.
    Unknown features are filled with 0.
    """

    # Initialize full zero vector
    full_vector = np.zeros(len(feature_names))

    # Map your known features
    feature_mapping = {
        "CREDIT_TO_INCOME": data.CREDIT_TO_INCOME,
        "ANNUITY_TO_INCOME": data.ANNUITY_TO_INCOME,
        "AGE_YEARS": data.AGE_YEARS,
        "EMPLOYMENT_YEARS": data.EMPLOYMENT_YEARS
    }

    for i, name in enumerate(feature_names):
        if name in feature_mapping:
            full_vector[i] = feature_mapping[name]

    return full_vector.reshape(1, -1)


# 🔹 Predict Endpoint
@app.post("/predict")
def predict(data: ApplicantData):

    input_array = build_full_feature_vector(data)

    prob = model.predict_proba(input_array)[0][1]

    return {
        "probability": float(prob),
        "risk_category": categorize_risk(prob)
    }



# 🔹 Explain Endpoint
@app.post("/explain")
def explain(data: ApplicantData):

    input_array = build_full_feature_vector(data)

    prob = model.predict_proba(input_array)[0][1]

    structured_explanation = generate_structured_explanation(
        model,
        explainer,
        input_array,
        0,
        feature_names
    )

    text_explanation = generate_text_explanation(structured_explanation)

    return {
        "probability": float(prob),
        "risk_category": categorize_risk(prob),
        "structured_explanation": structured_explanation,
        "text_explanation": text_explanation
    }


# 🔹 Health Endpoint
@app.get("/health")
def health():
    return {"status": "API running"}
