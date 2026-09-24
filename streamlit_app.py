import streamlit as st
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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Campaign Intelligence",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/bank_marketing_pipeline.joblib"
    )

    config = joblib.load(
        "models/deployment_config.joblib"
    )

    return model, config


model, config = load_model()

THRESHOLD = config["threshold"]


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bank Campaign Intelligence")

st.caption(
    "AI-powered customer targeting using Machine Learning, "
    "SHAP explainability and campaign rules"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("👤 Customer Profile")


age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)


job = st.sidebar.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ],
    index=4
)


marital = st.sidebar.selectbox(
    "Marital Status",
    [
        "married",
        "single",
        "divorced"
    ]
)


education = st.sidebar.selectbox(
    "Education",
    [
        "primary",
        "secondary",
        "tertiary",
        "unknown"
    ],
    index=2
)


default = st.sidebar.selectbox(
    "Credit Default",
    [
        "no",
        "yes"
    ]
)


balance = st.sidebar.number_input(
    "Account Balance",
    value=1500
)


housing = st.sidebar.selectbox(
    "Housing Loan",
    [
        "no",
        "yes"
    ],
    index=1
)


loan = st.sidebar.selectbox(
    "Personal Loan",
    [
        "no",
        "yes"
    ]
)


contact = st.sidebar.selectbox(
    "Contact",
    [
        "cellular",
        "telephone",
        "unknown"
    ]
)


day = st.sidebar.number_input(
    "Campaign Day",
    min_value=1,
    max_value=31,
    value=15
)


month = st.sidebar.selectbox(
    "Campaign Month",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec"
    ],
    index=4
)


campaign = st.sidebar.number_input(
    "Current Campaign Contacts",
    min_value=1,
    value=2
)


pdays = st.sidebar.number_input(
    "Days Since Previous Contact",
    value=10
)


previous = st.sidebar.number_input(
    "Previous Contacts",
    min_value=0,
    value=1
)


poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome",
    [
        "unknown",
        "failure",
        "other",
        "success"
    ],
    index=3
)


# ============================================================
# CUSTOMER DATA
# ============================================================

customer = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.sidebar.button(
    "🔍 Analyze Customer",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYZE CUSTOMER
# ============================================================

if analyze:

    # Create dataframe from CURRENT sidebar values
    input_data = pd.DataFrame(
        [customer]
    )

    with st.spinner(
        "Running ML model and generating explanation..."
    ):

        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        probability = model.predict_proba(
            input_data
        )[0, 1]

        prediction = (
            "yes"
            if probability >= THRESHOLD
            else "no"
        )


        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        shap_explanation = get_shap_explanation(
            model,
            input_data,
            top_n=5
        )


        # ----------------------------------------------------
        # HUMAN-READABLE SHAP
        # ----------------------------------------------------

        readable_signals = make_shap_readable(
            shap_explanation,
            customer
        )


        # ----------------------------------------------------
        # CAMPAIGN RULES
        # ----------------------------------------------------

        rules = generate_campaign_rules(
            prediction=prediction,
            probability=probability,
            threshold=THRESHOLD,
            customer_data=customer,
            shap_explanation=shap_explanation
        )


    # ========================================================
    # SAVE ONLY CURRENT ANALYSIS
    # ========================================================

    st.session_state["analyzed"] = True

    # Save a COPY so later sidebar changes cannot modify it
    st.session_state["customer"] = customer.copy()

    st.session_state["prediction"] = prediction

    st.session_state["probability"] = probability

    st.session_state["shap"] = readable_signals

    st.session_state["rules"] = rules

    # Remove old Gemini response
    st.session_state.pop(
        "ai_insight",
        None
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.get(
    "analyzed",
    False
):

    prediction = (
        st.session_state["prediction"]
    )

    probability = (
        st.session_state["probability"]
    )

    readable_signals = (
        st.session_state["shap"]
    )

    rules = (
        st.session_state["rules"]
    )

    analyzed_customer = (
        st.session_state["customer"]
    )


    # ========================================================
    # KPI CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Prediction",
            prediction.upper()
        )


    with col2:

        st.metric(
            "Probability",
            f"{probability * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Decision Threshold",
            f"{THRESHOLD * 100:.0f}%"
        )


    st.divider()


    # ========================================================
    # ANALYZED CUSTOMER
    # ========================================================

    st.subheader(
        "👤 Analyzed Customer"
    )

    # Display values directly.
    # No dataframe -> avoids canvas rendering artifact.

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.caption("Age")
        st.write(analyzed_customer["age"])

        st.caption("Job")
        st.write(analyzed_customer["job"])

        st.caption("Marital Status")
        st.write(analyzed_customer["marital"])

        st.caption("Education")
        st.write(analyzed_customer["education"])


    with col2:
        st.caption("Credit Default")
        st.write(analyzed_customer["default"])

        st.caption("Account Balance")
        st.write(analyzed_customer["balance"])

        st.caption("Housing Loan")
        st.write(analyzed_customer["housing"])

        st.caption("Personal Loan")
        st.write(analyzed_customer["loan"])


    with col3:
        st.caption("Contact")
        st.write(analyzed_customer["contact"])

        st.caption("Campaign Day")
        st.write(analyzed_customer["day"])

        st.caption("Campaign Month")
        st.write(analyzed_customer["month"])

        st.caption("Current Campaign Contacts")
        st.write(analyzed_customer["campaign"])


    with col4:
        st.caption("Days Since Previous Contact")
        st.write(analyzed_customer["pdays"])

        st.caption("Previous Contacts")
        st.write(analyzed_customer["previous"])

        st.caption("Previous Campaign Outcome")
        st.write(analyzed_customer["poutcome"])


    st.divider()


    # ========================================================
    # CAMPAIGN DECISION
    # ========================================================

    st.subheader(
        "🎯 Campaign Decision"
    )

    priority = rules["priority"]


    if priority == "HIGH":

        st.success(
            f"Priority: {priority}"
        )

    elif priority == "MEDIUM":

        st.warning(
            f"Priority: {priority}"
        )

    else:

        st.info(
            f"Priority: {priority}"
        )


    st.write(
        f"**Recommended Action:** "
        f"{rules['recommended_action']}"
    )


    st.divider()


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.subheader(
        "📊 Why did the model make this prediction?"
    )


    positive_signals = [
        signal
        for signal in readable_signals
        if signal["impact"] == "positive"
    ]


    negative_signals = [
        signal
        for signal in readable_signals
        if signal["impact"] == "negative"
    ]


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # POSITIVE SIGNALS
    # --------------------------------------------------------

    with col1:

        st.markdown(
            "### 🟢 Positive Signals"
        )


        if positive_signals:

            for signal in positive_signals:

                st.success(
                    signal["feature"]
                )

        else:

            st.write(
                "No strong positive signals."
            )


    # --------------------------------------------------------
    # NEGATIVE SIGNALS
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "### 🔴 Negative Signals"
        )


        if negative_signals:

            for signal in negative_signals:

                st.error(
                    signal["feature"]
                )

        else:

            st.write(
                "No strong negative signals."
            )


    st.divider()


    # ========================================================
    # MODEL SIGNAL DETAILS
    # ========================================================

    st.subheader(
        "📋 Model Signal Details"
    )


    for signal in readable_signals:

        feature = signal["feature"]

        impact = signal["impact"]

        shap_value = signal["shap_value"]


        if impact == "positive":

            icon = "🟢"

            direction = "Increases prediction"

        else:

            icon = "🔴"

            direction = "Decreases prediction"


        st.markdown(
            f"""
**{icon} {feature}**

`{direction}` · SHAP value: `{shap_value:+.4f}`
"""
        )

        st.divider()


    # ========================================================
    # AI CAMPAIGN ASSISTANT
    # ========================================================

    st.subheader(
        "🤖 AI Campaign Assistant"
    )


    st.info(
        "The Random Forest model makes the prediction. "
        "SHAP explains the important signals. "
        "Gemini only converts these results into a "
        "business-oriented explanation."
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Gemini is generated only from the LAST ANALYZED customer
    # --------------------------------------------------------

    generate_ai = st.button(
        "✨ Generate AI Campaign Insight",
        use_container_width=True
    )


    # ========================================================
    # GEMINI GENERATION
    # ========================================================

    if generate_ai:

        customer_data = (
            st.session_state["customer"]
        )

        prediction = (
            st.session_state["prediction"]
        )

        probability = (
            st.session_state["probability"]
        )

        readable_signals = (
            st.session_state["shap"]
        )

        rules = (
            st.session_state["rules"]
        )


        with st.spinner(
            "Generating AI campaign insight..."
        ):

            try:

                ai_insight = generate_campaign_insight(
                    prediction=prediction,
                    probability=probability,
                    threshold=THRESHOLD,
                    customer_data=customer_data,
                    readable_signals=readable_signals,
                    campaign_rules=rules
                )


                st.session_state[
                    "ai_insight"
                ] = ai_insight


            except Exception as e:

                st.session_state[
                    "ai_insight"
                ] = None


                st.error(
                    "Gemini could not generate the insight."
                )


                st.caption(
                    f"Error: {str(e)}"
                )


    # ========================================================
    # DISPLAY GEMINI RESULT
    # ========================================================

    if (
        "ai_insight"
        in st.session_state
        and st.session_state["ai_insight"]
    ):

        st.markdown(
            "### ✨ AI-Generated Campaign Insight"
        )


        st.markdown(
            st.session_state["ai_insight"]
        )


# ============================================================
# INITIAL STATE MESSAGE
# ============================================================

else:

    st.info(
        "👈 Enter customer information in the sidebar "
        "and click **Analyze Customer** to begin."
    )