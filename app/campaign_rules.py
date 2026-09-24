"""
Deterministic campaign decision rules.

The machine learning model makes the prediction.
SHAP provides model explanation signals.
This module converts those results into campaign priority
and a recommended action.

Gemini does NOT make campaign decisions.
"""

from typing import Any, Dict


def generate_campaign_rules(
    prediction: str,
    probability: float,
    threshold: float,
    customer_data: Dict[str, Any],
    shap_explanation,
) -> Dict[str, Any]:
    """
    Generate deterministic campaign rules.

    The SHAP readable output is expected to be a list
    of dictionaries containing:

        feature
        impact
        shap_value
    """

    # =========================================================
    # 1. CAMPAIGN PRIORITY
    # =========================================================

    if probability >= 0.70:

        priority = "HIGH"

    elif probability >= threshold:

        priority = "MEDIUM"

    else:

        priority = "LOW"

    # =========================================================
    # 2. RECOMMENDED ACTION
    # =========================================================

    if prediction.lower() == "yes":

        recommended_action = (
            "Prioritize this customer for targeted campaign outreach."
        )

    else:

        recommended_action = (
            "Do not prioritize this customer for targeted outreach."
        )

    # =========================================================
    # 3. EXTRACT SHAP SIGNALS
    # =========================================================

    positive_signals = []
    negative_signals = []

    if shap_explanation is None:

        shap_rows = []

    elif isinstance(shap_explanation, list):

        shap_rows = shap_explanation

    else:

        # Safety fallback in case a DataFrame is supplied.
        try:

            shap_rows = shap_explanation.to_dict(
                "records"
            )

        except AttributeError:

            shap_rows = []

    # =========================================================
    # 4. PROCESS READABLE SHAP OUTPUT
    # =========================================================

    for row in shap_rows:

        if not isinstance(row, dict):
            continue

        feature = str(
            row.get("feature", "")
        ).strip()

        impact = str(
            row.get("impact", "")
        ).lower().strip()

        if not feature:
            continue

        # -----------------------------------------------------
        # Positive SHAP signal
        # -----------------------------------------------------

        if impact == "positive":

            positive_signals.append(feature)

        # -----------------------------------------------------
        # Negative SHAP signal
        # -----------------------------------------------------

        elif impact == "negative":

            negative_signals.append(feature)

    # =========================================================
    # 5. RETURN CAMPAIGN DECISION
    # =========================================================

    return {
        "priority": priority,

        "recommended_action": recommended_action,

        "positive_signals": positive_signals,

        "negative_signals": negative_signals,
    }