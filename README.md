# Explainable Loan Default Prediction System

An end-to-end Explainable AI system for predicting loan default risk using Logistic Regression and SHAP-based interpretability.

This project includes:

- Data preprocessing pipeline
- Model training & evaluation
- Threshold optimization
- SHAP explainability
- Structured explanation engine
- FastAPI backend
- Streamlit frontend

---

## Project Architecture

User → Streamlit UI → FastAPI → Model → SHAP → Text Explanation → JSON Response

---

## Features

- 219 engineered features
- Logistic Regression (locked production model)
- Threshold optimized for high recall (0.40)
- SHAP global & local explanations
- Deterministic text explanation engine
- REST API endpoints:
  - `/predict`
  - `/explain`
  - `/health`

---

## Repository Structure

```
api/
src/
ui/
models/
notebooks/
reports/
requirements.txt
README.md
```

---

## ⚙️ Installation Guide (For New System)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/PRAMEELA-tech/explainable-loan-default-system.git
cd explainable-loan-default-system
```


### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Backend API

From project root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

You should see:

```
Uvicorn running on http://0.0.0.0:8000
```

Open Swagger UI:

```
http://localhost:8000/docs
```

Test endpoints:
- `/predict`
- `/explain`

---

## 🖥 Running the Streamlit UI

Open a new terminal (keep API running):

```bash
streamlit run ui/streamlit_app.py
```

Streamlit will open:

```
http://localhost:8501
```

Enter:
- Credit-to-Income Ratio
- Annuity-to-Income Ratio
- Age
- Employment Duration

Click **Analyze Risk**.

---

## API Endpoints

### POST `/predict`

Returns:

```json
{
  "probability": 0.42,
  "risk_category": "Medium Risk"
}
```

---

### POST `/explain`

Returns:

```json
{
  "probability": 0.42,
  "risk_category": "Medium Risk",
  "structured_explanation": {...},
  "text_explanation": "The applicant is categorized as..."
}
```

---

## Model Details

- Model: Logistic Regression
- Features: 219 engineered features
- Threshold: 0.40
- Recall (Default class): 0.81
- ROC-AUC: 0.749

---

## Important Notes

- Dataset is excluded from repository.
- Model artifacts are included.
- Project runs without retraining.

---

## Author

Ginni Prameela 
B.Tech CSE  
Explainable AI Project
