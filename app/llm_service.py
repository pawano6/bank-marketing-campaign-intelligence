"""
Gemini service.

Gemini is used ONLY to convert already-generated ML results
and SHAP explanations into a business-readable explanation.

Gemini does NOT:
- make predictions
- calculate probabilities
- choose campaign priority
- create campaign rules
- invent customer information
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


def _get_client():
    """
    Create a Gemini client only when it is actually needed.

    This allows the FastAPI application and ML prediction
    endpoints to work even if Gemini is not configured.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Add it to your .env file before using Gemini insights."
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000),
    )


def generate_campaign_insight(
    prediction,
    probability,
    threshold,
    priority,
    recommended_action,
    model_signals,
    customer_data,
):
    """
    Generate a short business explanation using Gemini.

    Gemini receives the outputs of the ML system and explains them.
    It does not make a new prediction or campaign decision.
    """

    client = _get_client()

    # ---------------------------------------------------------
    # Format model signals
    # ---------------------------------------------------------

    if model_signals:
        signals_text = "\n".join(
            f"- {signal}" for signal in model_signals
        )
    else:
        signals_text = "- No major SHAP signals were identified."

    # ---------------------------------------------------------
    # Special handling for pdays
    # ---------------------------------------------------------

    pdays_value = customer_data.get("pdays")

    if pdays_value == 999:
        pdays_context = (
            "pdays = 999, meaning no previous contact is recorded."
        )
    else:
        pdays_context = (
            f"pdays = {pdays_value}."
        )

    # ---------------------------------------------------------
    # Gemini prompt
    # ---------------------------------------------------------

    prompt = f"""
You are a business analyst explaining the result of a bank marketing
campaign prediction system.

IMPORTANT:

The machine learning model has ALREADY made the prediction.

You must NOT:
- make a new prediction
- change the prediction
- change the probability
- change the campaign priority
- change the recommended action
- invent customer information
- invent customer preferences
- claim that a feature causes the prediction
- create a customer persona
- recommend a communication channel unless explicitly provided
- recommend campaign timing or frequency
- create new campaign rules

Your job is ONLY to explain the supplied result clearly.

MODEL RESULT
Prediction: {prediction}
Probability: {probability:.4f}
Classification threshold: {threshold:.2f}

DETERMINISTIC CAMPAIGN DECISION
Priority: {priority}
Recommended action: {recommended_action}

MODEL EXPLANATION SIGNALS
{signals_text}

CUSTOMER CONTEXT
Age: {customer_data.get("age")}
Job: {customer_data.get("job")}
Marital status: {customer_data.get("marital")}
Education: {customer_data.get("education")}
Default: {customer_data.get("default")}
Balance: {customer_data.get("balance")}
Housing loan: {customer_data.get("housing")}
Personal loan: {customer_data.get("loan")}
Contact type: {customer_data.get("contact")}
Campaign contacts: {customer_data.get("campaign")}
Previous contacts: {customer_data.get("previous")}
Previous campaign outcome: {customer_data.get("poutcome")}
{pdays_context}

IMPORTANT INTERPRETATION RULES

- SHAP signals describe model behavior, not causality.
- Do not say a feature "caused" the result.
- Do not invent information that is not provided.
- If pdays = 999, describe it as no previous contact recorded.
- Do not infer customer preferences from demographics.
- Do not create a persona.
- Do not create new campaign rules.
- Keep the supplied priority and recommended action exactly as provided.

OUTPUT FORMAT

PRIORITY
<repeat the supplied priority>

INSIGHT
<brief explanation of the prediction using the supplied model signals>

RECOMMENDED ACTION
<repeat the supplied recommended action>

APPROACH
<one short practical explanation based only on the available information>

Keep the entire response under 100 words.
"""

    # ---------------------------------------------------------
    # Gemini API call
    # ---------------------------------------------------------

    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    return response.output_text.strip()