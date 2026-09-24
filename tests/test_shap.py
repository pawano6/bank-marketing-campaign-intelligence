import pandas as pd
import joblib

from app.explainability import get_shap_explanation


# Load trained model
model = joblib.load(
    "models/bank_marketing_pipeline.joblib"
)


# Example customer
customer = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": 10,
    "previous": 1,
    "poutcome": "success"
}


# Convert customer into DataFrame
input_data = pd.DataFrame([customer])


# Get SHAP explanation
result = get_shap_explanation(
    model,
    input_data,
    top_n=5
)


print("\n========== SHAP EXPLANATION ==========\n")
print(result.to_string(index=False))