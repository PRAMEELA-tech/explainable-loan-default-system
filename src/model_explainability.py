import shap
import numpy as np


def generate_global_shap(model, X_sample):
    """
    Generates SHAP values for a dataset sample.
    Returns explainer and shap values.
    """
    explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def generate_local_shap(explainer, sample):
    """
    Generates SHAP values for a single sample.
    """
    shap_values_single = explainer.shap_values(sample.reshape(1, -1))
    return shap_values_single


import numpy as np

def generate_structured_explanation(model, explainer, X_sample, index, feature_names):
    shap_values = explainer.shap_values(X_sample)
    
    shap_vals = shap_values[index]
    feature_vals = X_sample[index]

    explanation_data = []

    for i in range(len(feature_names)):
        explanation_data.append({
            "feature": feature_names[i],
            "value": float(feature_vals[i]),
            "shap_contribution": float(shap_vals[i])
        })

    positive_impact = sorted(
        [f for f in explanation_data if f["shap_contribution"] > 0],
        key=lambda x: x["shap_contribution"],
        reverse=True
    )

    negative_impact = sorted(
        [f for f in explanation_data if f["shap_contribution"] < 0],
        key=lambda x: x["shap_contribution"]
    )

    structured_output = {
        "base_value": float(explainer.expected_value),
        "prediction_probability": float(
            model.predict_proba(X_sample[index].reshape(1, -1))[0][1]
        ),
        "top_risk_increasing_features": positive_impact[:5],
        "top_risk_reducing_features": negative_impact[:5]
    }

    return structured_output
