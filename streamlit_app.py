import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Bank Campaign Intelligence",
    page_icon="🏦",
    layout="wide",
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://bank-marketing-campaign-intelligence.onrender.com",
).rstrip("/")


def call_campaign_api(customer_data):
    url = f"{API_BASE_URL}/campaign-insight"

    response = requests.post(
        url,
        json=customer_data,
        timeout=120,
    )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"FastAPI returned HTTP {response.status_code}: {detail}"
        )

    return response.json()


st.title("🏦 Bank Campaign Intelligence")
st.caption(
    "AI-powered customer targeting using Machine Learning, "
    "SHAP explainability and deterministic campaign rules"
)
st.info(
    "Architecture: Streamlit → FastAPI → Random Forest + SHAP + "
    "Campaign Rules + Gemini"
)

st.sidebar.header("👤 Customer Profile")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown",
    ],
    index=4,
)

marital = st.sidebar.selectbox(
    "Marital Status", ["married", "single", "divorced"]
)

education = st.sidebar.selectbox(
    "Education", ["primary", "secondary", "tertiary", "unknown"], index=2
)

default = st.sidebar.selectbox("Credit Default", ["no", "yes"])
balance = st.sidebar.number_input("Account Balance", value=1500)
housing = st.sidebar.selectbox("Housing Loan", ["no", "yes"], index=1)
loan = st.sidebar.selectbox("Personal Loan", ["no", "yes"])
contact = st.sidebar.selectbox(
    "Contact", ["cellular", "telephone", "unknown"]
)
day = st.sidebar.number_input(
    "Campaign Day", min_value=1, max_value=31, value=15
)
month = st.sidebar.selectbox(
    "Campaign Month",
    [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ],
    index=4,
)
campaign = st.sidebar.number_input(
    "Current Campaign Contacts", min_value=1, value=2
)
pdays = st.sidebar.number_input("Days Since Previous Contact", value=10)
previous = st.sidebar.number_input("Previous Contacts", min_value=0, value=1)
poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome",
    ["unknown", "failure", "other", "success"],
    index=3,
)

customer = {
    "age": int(age),
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": int(balance),
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": int(day),
    "month": month,
    "campaign": int(campaign),
    "pdays": int(pdays),
    "previous": int(previous),
    "poutcome": poutcome,
}

analyze = st.sidebar.button(
    "🔍 Analyze Customer",
    type="primary",
    use_container_width=True,
)

if analyze:
    with st.spinner(
        "Calling production FastAPI and generating campaign insight..."
    ):
        try:
            result = call_campaign_api(customer)
            st.session_state["analyzed"] = True
            st.session_state["customer"] = customer.copy()
            st.session_state["result"] = result
        except requests.exceptions.Timeout:
            st.error(
                "The API request timed out. The Render service may be "
                "starting up or Gemini may be taking longer than expected."
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to the FastAPI backend: {exc}")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

if st.session_state.get("analyzed", False):
    result = st.session_state["result"]
    analyzed_customer = st.session_state["customer"]

    prediction = result["prediction"]
    probability = result["probability"]
    threshold = result["threshold"]
    priority = result["priority"]
    positive_signals = result.get("positive_signals", [])
    negative_signals = result.get("negative_signals", [])
    recommended_action = result["recommended_action"]
    ai_insight = result.get("ai_insight", "")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", prediction.upper())
    with col2:
        st.metric("Probability", f"{probability * 100:.2f}%")
    with col3:
        st.metric("Decision Threshold", f"{threshold * 100:.0f}%")

    st.divider()
    st.subheader("👤 Analyzed Customer")

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
    st.subheader("🎯 Campaign Decision")

    if priority == "HIGH":
        st.success(f"Priority: {priority}")
    elif priority == "MEDIUM":
        st.warning(f"Priority: {priority}")
    else:
        st.info(f"Priority: {priority}")

    st.write(f"**Recommended Action:** {recommended_action}")

    st.divider()
    st.subheader("📊 Why did the model make this prediction?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟢 Positive Signals")
        if positive_signals:
            for signal in positive_signals:
                st.success(signal)
        else:
            st.write("No strong positive signals.")

    with col2:
        st.markdown("### 🔴 Negative Signals")
        if negative_signals:
            for signal in negative_signals:
                st.error(signal)
        else:
            st.write("No strong negative signals.")

    st.divider()
    st.subheader("🤖 AI Campaign Assistant")
    st.info(
        "The Random Forest model makes the prediction. "
        "SHAP explains the important signals. "
        "Deterministic rules assign campaign priority. "
        "Gemini converts these results into a business-readable explanation."
    )

    if ai_insight:
        st.markdown("### ✨ AI-Generated Campaign Insight")
        st.markdown(ai_insight)

else:
    st.info(
        "👈 Enter customer information in the sidebar "
        "and click **Analyze Customer** to begin."
    )

st.caption(f"Backend API: {API_BASE_URL}")
