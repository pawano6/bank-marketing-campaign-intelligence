# Bank Marketing Campaign Intelligence

An end-to-end machine learning application that predicts whether a bank customer is likely to subscribe to a term deposit and converts the model output into an actionable campaign insight.

The project combines **Random Forest classification, SHAP explainability, deterministic business rules, FastAPI, Streamlit, and Gemini** in a production-style workflow.

## Project Architecture

```text
Customer Input
      ↓
Random Forest
      ↓
Prediction + Probability
      ↓
SHAP
      ↓
Deterministic Campaign Rules
      ↓
Gemini
      ↓
Business-readable Explanation
```

### Design Principle

- **Random Forest** makes the prediction.
- **SHAP** explains the model's important signals.
- **Deterministic rules** assign campaign priority and recommended action.
- **Gemini** only converts the existing model results and explanations into business-readable language.
- Gemini does **not** make predictions or campaign decisions.

## Key Results

The final model was evaluated on an untouched test set.

| Metric | Result |
|---|---:|
| Accuracy | 90.36% |
| Precision | 57.89% |
| Recall | 59.88% |
| F1 Score | 58.87% |
| ROC-AUC | 95.59% |
| Production Threshold | 0.59 |

The production classification threshold was selected as **0.59** rather than relying on the default 0.50 threshold.

## Features

### Machine Learning
- Exploratory data analysis
- Feature analysis and engineering
- Model comparison
- Random Forest classification
- Hyperparameter tuning
- Probability threshold selection
- Final untouched test evaluation

### Explainable AI
- SHAP TreeExplainer
- Top model signals for each prediction
- Positive and negative model signals
- Human-readable feature explanations
- Special handling for categorical features and campaign-related fields

### Business Decision Layer
The system converts the model probability into campaign priority:

- **HIGH**: probability >= 0.70
- **MEDIUM**: probability >= 0.59 and < 0.70
- **LOW**: probability < 0.59

The recommendation is deterministic and based on the model prediction:

- Prediction = `yes` → prioritize targeted campaign outreach
- Prediction = `no` → do not prioritize targeted outreach

### Generative AI
Gemini receives the already-computed prediction, probability, SHAP signals, campaign priority, recommendation, and customer context.

It produces a concise business explanation without changing the underlying ML decision.

## Production Model

The production pipeline is stored in:

```text
models/bank_marketing_pipeline.joblib
```

Deployment configuration, including the production threshold, is stored in:

```text
models/deployment_config.joblib
```

The production model excludes `duration` because call duration is only known during/after a campaign interaction and is therefore not appropriate for pre-campaign customer targeting.

## Project Structure

```text
project/
├── app/
│   ├── campaign_rules.py
│   ├── explainability.py
│   ├── llm_service.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── ...
│
├── models/
│   ├── bank_marketing_pipeline.joblib
│   └── deployment_config.joblib
│
├── notebooks/
│   └── ...
│
├── tests/
│   └── ...
│
├── .env
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SHAP
- FastAPI
- Pydantic
- Streamlit
- Google Gemini API
- Joblib
- python-dotenv

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd project
```

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

A template is provided in:

```text
.env.example
```

Never commit the real `.env` file or your API key to GitHub.

## Run the FastAPI Backend

From the project root:

```powershell
python -m uvicorn app.main:app --reload
```

The API will run locally at:

```text
http://127.0.0.1:8000
```

Useful endpoints:

```text
GET  /
GET  /health
POST /predict
POST /campaign-insight
```

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Run the Streamlit Application

Open a second terminal, activate the virtual environment, and run:

```powershell
python -m streamlit run streamlit_app.py
```

Streamlit will provide the local application URL in the terminal.

The UI allows a user to:

1. Enter customer information.
2. Generate a prediction.
3. View the subscription probability.
4. View campaign priority.
5. Inspect SHAP model signals.
6. View the deterministic recommended action.
7. Generate a Gemini-powered business explanation.

## API Example

Example customer request:

```json
{
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
  "campaign": 1,
  "pdays": 999,
  "previous": 0,
  "poutcome": "unknown"
}
```

A prediction response contains:

```json
{
  "prediction": "yes",
  "probability": 0.72,
  "threshold": 0.59
}
```

The exact probability and prediction depend on the saved production model and customer input.

## Explainability

For each prediction, SHAP is used to identify the features that contributed most strongly to the model output.

The application separates signals into:

- Positive model signals
- Negative model signals

These signals describe **model behavior and association**, not causality.

For example, a feature appearing as a positive SHAP signal means that the feature pushed the model output toward the positive class for that specific prediction. It does not mean that the feature caused the customer to subscribe.

## Campaign Intelligence Layer

The campaign rules are intentionally deterministic.

The model output is first converted into a probability and prediction using the production threshold.

The rule layer then assigns:

```text
Probability >= 0.70
        → HIGH

0.59 <= Probability < 0.70
        → MEDIUM

Probability < 0.59
        → LOW
```

This keeps business decisions transparent and reproducible.

## Gemini Guardrails

Gemini is deliberately placed after the ML and rule-based layers.

It is instructed not to:

- make a new prediction
- change the model probability
- change campaign priority
- change the recommended action
- invent customer information
- infer customer preferences
- claim that a feature caused the prediction
- create new campaign rules

Its purpose is communication, not decision-making.

## Testing

The application was tested through the FastAPI endpoints, including:

- Root endpoint
- Health check
- Prediction endpoint
- Campaign insight endpoint
- SHAP explanation flow
- Deterministic campaign rules
- Gemini explanation flow

The production model successfully loads from the saved Joblib artifact and uses the configured threshold of `0.59`.

## Reproducibility

The project pins the production Scikit-learn version:

```text
scikit-learn==1.7.2
```

This helps avoid model deserialization compatibility problems between the environment used to train/save the model and the environment used to run the application.

## Important Notes

- The saved model artifacts are required for the application to run.
- `.env` should remain local and must not be committed.
- `duration` is excluded from production inference because it is not available before/during the targeting decision.
- SHAP explanations should be interpreted as model explanations, not causal explanations.
- Gemini is an explanation layer and does not replace the predictive model or deterministic business rules.

## Future Improvements

Possible extensions include:

- Model monitoring
- Prediction logging
- Batch campaign scoring
- Authentication for the API
- Docker deployment
- Cloud deployment
- Automated CI/CD testing
- Drift monitoring
- Campaign performance tracking

## Author

**Pawan Singh**

Machine Learning | Explainable AI | FastAPI | Streamlit | Generative AI
