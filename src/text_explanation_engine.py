# src/text_explanation_engine.py

def categorize_risk(probability: float) -> str:
    """
    Categorizes default probability into human-readable risk levels.
    Deterministic thresholds.
    """
    if probability >= 0.75:
        return "Very High Risk"
    elif probability >= 0.50:
        return "High Risk"
    elif probability >= 0.25:
        return "Moderate Risk"
    else:
        return "Low Risk"


FEATURE_NAME_MAP = {
    "CREDIT_TO_INCOME": "credit-to-income ratio",
    "ANNUITY_TO_INCOME": "annuity burden",
    "AGE_YEARS": "applicant age",
    "EMPLOYMENT_YEARS": "employment duration",
}


def map_feature_name(feature_name: str) -> str:
    """
    Converts internal feature name to human-friendly term.
    Falls back to original if not mapped.
    """
    return FEATURE_NAME_MAP.get(feature_name, feature_name)


def generate_text_explanation(structured_explanation: dict) -> str:
    """
    Generates a deterministic, human-readable explanation
    from a structured SHAP explanation object.
    """

    base_value = structured_explanation["base_value"]
    probability = structured_explanation["prediction_probability"]
    top_positive = structured_explanation["top_risk_increasing_features"]
    top_negative = structured_explanation["top_risk_reducing_features"]

    # Categorize risk
    risk_category = categorize_risk(probability)

    # Limit to top 3 drivers
    top_positive = top_positive[:3]
    top_negative = top_negative[:3]

    # Convert feature names
    positive_features = [map_feature_name(f["feature"]) for f in top_positive]
    negative_features = [map_feature_name(f["feature"]) for f in top_negative]

    # Build risk-increasing sentence
    if positive_features:
        risk_increase_sentence = (
            "The default risk is primarily driven by "
            + ", ".join(positive_features)
            + "."
        )
    else:
        risk_increase_sentence = "No significant risk-increasing factors were identified."

        # Build risk-reducing sentence
    if negative_features:
        risk_reduce_sentence = (
            "The risk is partially mitigated by "
            + ", ".join(negative_features)
            + "."
        )
    else:
        risk_reduce_sentence = "No significant risk-reducing factors were identified."


    return f"""
The applicant is categorized as {risk_category}.
The predicted probability of default is {probability:.2f}.

{risk_increase_sentence}

{risk_reduce_sentence}

Overall, the decision is based on additive contributions of financial capacity indicators, credit exposure metrics, and stability-related factors.
"""
