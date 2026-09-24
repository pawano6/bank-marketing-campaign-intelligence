"""
FastAPI backend for the Bank Marketing Campaign Prediction system.

Architecture:

Customer Input
      ↓
Random Forest
      ↓
Prediction + Probability
      ↓
SHAP Explanation
      ↓
Deterministic Campaign Rules
      ↓
Gemini Business Explanation
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    CustomerData,
    PredictionResponse,
    CampaignInsightResponse,
)

from app.explainability import (
    get_shap_explanation,
    make_shap_readable,
)

from app.campaign_rules import (
    generate_campaign_rules,
)

from app.llm_service import (
    generate_campaign_insight,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "bank_marketing_pipeline.joblib"

CONFIG_PATH = BASE_DIR / "models" / "deployment_config.joblib"


# ============================================================
# LOAD MODEL AND CONFIGURATION
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

except Exception as exc:

    raise RuntimeError(
        f"Could not load production model from {MODEL_PATH}: {exc}"
    )


try:

    config = joblib.load(CONFIG_PATH)

except Exception as exc:

    raise RuntimeError(
        f"Could not load deployment configuration from {CONFIG_PATH}: {exc}"
    )


if "threshold" not in config:

    raise RuntimeError(
        "deployment_config.joblib does not contain a 'threshold' value."
    )


THRESHOLD = float(config["threshold"])


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Bank Marketing Campaign Prediction API",
    description=(
        "Production API for bank marketing campaign prediction, "
        "SHAP explainability, deterministic campaign rules, "
        "and Gemini-generated business insights."
    ),
    version="1.0.0",
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "message": "Bank Marketing Campaign Prediction API",
        "status": "running",
        "threshold": THRESHOLD,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerData):
    """
    Generate a model prediction and probability.
    """

    try:

        customer_dict = customer.model_dump()

        customer_df = pd.DataFrame([customer_dict])

        # Probability of positive class
        probability = float(
            model.predict_proba(customer_df)[0, 1]
        )

        # Apply production threshold
        prediction = (
            "yes"
            if probability >= THRESHOLD
            else "no"
        )

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            threshold=THRESHOLD,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# ============================================================
# CAMPAIGN INSIGHT ENDPOINT
# ============================================================

@app.post(
    "/campaign-insight",
    response_model=CampaignInsightResponse,
)
def campaign_insight(customer: CustomerData):
    """
    Complete campaign intelligence pipeline:

    1. Random Forest prediction
    2. SHAP explanation
    3. Human-readable SHAP signals
    4. Deterministic campaign rules
    5. Gemini explanation
    """

    try:

        # ----------------------------------------------------
        # 1. Customer data
        # ----------------------------------------------------

        customer_dict = customer.model_dump()

        customer_df = pd.DataFrame(
            [customer_dict]
        )

        # ----------------------------------------------------
        # 2. Model prediction
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(customer_df)[0, 1]
        )

        prediction = (
            "yes"
            if probability >= THRESHOLD
            else "no"
        )

        # ----------------------------------------------------
        # 3. SHAP explanation
        # ----------------------------------------------------

        shap_explanation = get_shap_explanation(
            model,
            customer_df,
        )

        # ----------------------------------------------------
        # 4. Convert SHAP values into readable signals
        # ----------------------------------------------------

        readable_signals = make_shap_readable(
            shap_explanation,
            customer_dict,
        )

        # ----------------------------------------------------
        # 5. Deterministic campaign rules
        # ----------------------------------------------------

        campaign_rules = generate_campaign_rules(
            prediction=prediction,
            probability=probability,
            threshold=THRESHOLD,
            customer_data=customer_dict,
            shap_explanation=readable_signals,
        )

        # ----------------------------------------------------
        # 6. Gemini explanation
        # ----------------------------------------------------

        ai_insight = generate_campaign_insight(
            prediction=prediction,
            probability=probability,
            threshold=THRESHOLD,
            priority=campaign_rules["priority"],
            recommended_action=campaign_rules[
                "recommended_action"
            ],
            model_signals=(
                campaign_rules["positive_signals"]
                + campaign_rules["negative_signals"]
            ),
            customer_data=customer_dict,
        )

        # ----------------------------------------------------
        # 7. Return final result
        # ----------------------------------------------------

        return CampaignInsightResponse(
            prediction=prediction,
            probability=probability,
            threshold=THRESHOLD,
            priority=campaign_rules["priority"],
            positive_signals=campaign_rules["positive_signals"],
            negative_signals=campaign_rules["negative_signals"],
            recommended_action=campaign_rules[
                "recommended_action"
            ],
            ai_insight=ai_insight,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Campaign insight generation failed: {str(exc)}",
        )