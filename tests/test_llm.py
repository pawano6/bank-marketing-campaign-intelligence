import pandas as pd
import joblib

from app.explainability import (
    get_shap_explanation,
    make_shap_readable
)

from app.campaign_rules import (
    generate_campaign_rules
)

from app.llm_service import (
    generate_campaign_insight
)


# --------------------------------------------------
# 1. Load trained model
# --------------------------------------------------

model = joblib.load(
    "models/bank_marketing_pipeline.joblib"
)


# --------------------------------------------------
# 2. Customer data
# --------------------------------------------------

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


input_data = pd.DataFrame([customer])


# --------------------------------------------------
# 3. ML prediction
# --------------------------------------------------

probability = model.predict_proba(
    input_data
)[0, 1]

threshold = 0.59

prediction = (
    "yes"
    if probability >= threshold
    else "no"
)


# --------------------------------------------------
# 4. SHAP explanation
# --------------------------------------------------

shap_explanation = get_shap_explanation(
    model,
    input_data,
    top_n=5
)


# --------------------------------------------------
# 5. Human-readable SHAP
# --------------------------------------------------

readable_signals = make_shap_readable(
    shap_explanation,
    customer
)


# --------------------------------------------------
# 6. Campaign rules
# --------------------------------------------------

rules = generate_campaign_rules(
    prediction=prediction,
    probability=probability,
    threshold=threshold,
    customer_data=customer,
    shap_explanation=shap_explanation
)


# --------------------------------------------------
# 7. Gemini insight
# --------------------------------------------------

ai_insight = generate_campaign_insight(
    prediction=prediction,
    probability=probability,
    threshold=threshold,
    customer_data=customer,
    readable_signals=readable_signals,
    campaign_rules=rules
)


# --------------------------------------------------
# 8. Display complete result
# --------------------------------------------------

print("\n========== ML RESULT ==========")

print("Prediction:", prediction)

print(
    "Probability:",
    round(probability, 4)
)

print(
    "Threshold:",
    threshold
)


print("\n========== HUMAN-READABLE SHAP ==========")

for signal in readable_signals:
    print(signal)


print("\n========== CAMPAIGN RULES ==========")

print(
    "Priority:",
    rules["priority"]
)

print(
    "Recommended Action:",
    rules["recommended_action"]
)


print("\n========== GEMINI AI INSIGHT ==========")

print(ai_insight)