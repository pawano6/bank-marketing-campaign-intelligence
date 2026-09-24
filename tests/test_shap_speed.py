import time
import joblib
import pandas as pd

from app.explainability import get_shap_explanation


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(
    "models/bank_marketing_pipeline.joblib"
)

print("Model loaded.")


# ============================================================
# TEST CUSTOMER
# ============================================================

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
    "poutcome": "success",
}

input_data = pd.DataFrame(
    [customer]
)


# ============================================================
# TEST SHAP
# ============================================================

print("\nCalculating SHAP...")

start_time = time.time()

shap_result = get_shap_explanation(
    model,
    input_data,
    top_n=5
)

end_time = time.time()


# ============================================================
# RESULT
# ============================================================

print("\n========== SHAP RESULT ==========")

print(
    shap_result.to_string(
        index=False
    )
)

print(
    "\nSHAP execution time:",
    round(
        end_time - start_time,
        2
    ),
    "seconds"
)