# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

import numpy as np
import pandas as pd
import shap


# ============================================================
# SHAP EXPLAINER CACHE
# ============================================================

_explainer = None
_explainer_model_id = None


# ============================================================
# GET SHAP EXPLANATION
# ============================================================

def get_shap_explanation(
    model_pipeline,
    input_data,
    top_n=5
):
    """
    Generate local SHAP explanation for one customer.

    Returns the top SHAP features contributing to
    the model prediction.
    """

    global _explainer
    global _explainer_model_id

    # --------------------------------------------------------
    # Extract pipeline components
    # --------------------------------------------------------

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    # --------------------------------------------------------
    # Transform input using the SAME preprocessing pipeline
    # used by the trained model
    # --------------------------------------------------------

    transformed_data = preprocessor.transform(
        input_data
    )

    # Convert sparse matrix to dense
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    transformed_data = np.asarray(
        transformed_data,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # --------------------------------------------------------
    # Create / reuse TreeExplainer
    # --------------------------------------------------------

    model_id = id(model)

    if (
        _explainer is None
        or _explainer_model_id != model_id
    ):

        _explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent"
        )

        _explainer_model_id = model_id

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = _explainer.shap_values(
        transformed_data,
        check_additivity=False
    )

    # --------------------------------------------------------
    # Handle different SHAP output formats
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        # Binary classification
        values = shap_values[1][0]

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            values = shap_array[0, :, 1]

        elif shap_array.ndim == 2:

            values = shap_array[0]

        else:

            raise ValueError(
                f"Unexpected SHAP shape: "
                f"{shap_array.shape}"
            )

    # --------------------------------------------------------
    # Build explanation dataframe
    # --------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": values
        }
    )

    # Absolute SHAP value = local importance
    explanation["importance"] = (
        explanation["shap_value"].abs()
    )

    # Sort strongest signals first
    explanation = explanation.sort_values(
        "importance",
        ascending=False
    )

    # Keep top N
    top_features = (
        explanation
        .head(top_n)
        .copy()
    )

    # Direction of influence
    top_features["direction"] = np.where(
        top_features["shap_value"] > 0,
        "increases prediction",
        "decreases prediction"
    )

    return (
        top_features[
            [
                "feature",
                "shap_value",
                "direction"
            ]
        ]
        .reset_index(drop=True)
    )


# ============================================================
# CONVERT SHAP TO HUMAN-READABLE FORMAT
# ============================================================

def make_shap_readable(
    shap_explanation,
    customer_data
):
    """
    Convert transformed SHAP feature names into
    business-readable customer signals.

    Important:
    - Only the customer's actual categorical value
      is displayed.
    - pdays=999 is interpreted as no previous contact.
    """

    readable_signals = []

    # --------------------------------------------------------
    # Numeric feature labels
    # --------------------------------------------------------

    numeric_labels = {

        "age":
            "Age",

        "balance":
            "Account balance",

        "day":
            "Campaign day",

        "campaign":
            "Current campaign contacts",

        "pdays":
            "Days since previous contact",

        "previous":
            "Previous contacts"
    }

    # --------------------------------------------------------
    # Categorical feature labels
    # --------------------------------------------------------

    categorical_labels = {

        "job":
            "Job",

        "marital":
            "Marital status",

        "education":
            "Education",

        "default":
            "Credit default",

        "housing":
            "Housing loan",

        "loan":
            "Personal loan",

        "contact":
            "Contact channel",

        "month":
            "Campaign month",

        "poutcome":
            "Previous campaign outcome"
    }

    # --------------------------------------------------------
    # Process each SHAP feature
    # --------------------------------------------------------

    for _, row in shap_explanation.iterrows():

        feature = row["feature"]

        shap_value = float(
            row["shap_value"]
        )

        # ----------------------------------------------------
        # Determine impact
        # ----------------------------------------------------

        impact = (
            "positive"
            if shap_value > 0
            else "negative"
        )

        # ----------------------------------------------------
        # Remove preprocessing prefixes
        # ----------------------------------------------------

        clean_feature = (
            feature
            .replace("num__", "", 1)
            .replace("cat__", "", 1)
        )

        # ====================================================
        # SPECIAL CASE: PDAYS = 999
        # ====================================================

        if clean_feature == "pdays":

            pdays_value = customer_data["pdays"]

            if pdays_value == 999:

                readable_feature = (
                    "No previous contact recorded"
                )

            else:

                readable_feature = (
                    f"Days since previous contact = "
                    f"{pdays_value}"
                )

            readable_signals.append(
                {
                    "feature":
                        readable_feature,

                    "impact":
                        impact,

                    "shap_value":
                        round(
                            shap_value,
                            4
                        )
                }
            )

            continue

        # ====================================================
        # NUMERIC FEATURES
        # ====================================================

        if clean_feature in numeric_labels:

            readable_feature = (
                f"{numeric_labels[clean_feature]} = "
                f"{customer_data[clean_feature]}"
            )

            readable_signals.append(
                {
                    "feature":
                        readable_feature,

                    "impact":
                        impact,

                    "shap_value":
                        round(
                            shap_value,
                            4
                        )
                }
            )

            continue

        # ====================================================
        # CATEGORICAL FEATURES
        # ====================================================

        if "_" in clean_feature:

            original_feature, category = (
                clean_feature.split(
                    "_",
                    1
                )
            )

            if original_feature in categorical_labels:

                actual_value = str(
                    customer_data[
                        original_feature
                    ]
                )

                # ------------------------------------------------
                # IMPORTANT:
                # Ignore inactive one-hot categories.
                #
                # Example:
                # customer contact = cellular
                #
                # Ignore:
                # contact_unknown
                # contact_telephone
                #
                # Keep:
                # contact_cellular
                # ------------------------------------------------

                if category != actual_value:
                    continue

                readable_feature = (
                    f"{categorical_labels[original_feature]} = "
                    f"{category}"
                )

                readable_signals.append(
                    {
                        "feature":
                            readable_feature,

                        "impact":
                            impact,

                        "shap_value":
                            round(
                                shap_value,
                                4
                            )
                    }
                )

                continue

        # ====================================================
        # FALLBACK
        # ====================================================

        readable_signals.append(
            {
                "feature":
                    clean_feature,

                "impact":
                    impact,

                "shap_value":
                    round(
                        shap_value,
                        4
                    )
            }
        )

    return readable_signals