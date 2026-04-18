# SmartLender — AI-driven Credit Risk Intelligence & Repayment Risk Estimation System

An AI-driven credit risk intelligence system for smarter and safer lending decisions.

SmartLender is transformed from a simple loan-approval demo into a production-oriented credit risk engine that estimates repayment risk, explains key drivers, and supports risk-aware decision-making via API and a decision dashboard UI. This README documents the system capabilities, API contract, code layout, and quick-run instructions for hackathon and fintech evaluation.

## Production Upgrade Summary

- Outputs probability-based risk (not only labels): `risk_probability` (0–1) and a `risk_level` (Low/Medium/High).
- Lightweight explainability: returns top 3 `risk_drivers` per applicant using model importance + simple heuristics.
- Business logic layer: `recommended_action` (Approve / Review Manually / Reject) derived from risk thresholds.
- Extensible placeholders for placement‑risk modeling: `placement_risk_extension` includes mock employment probability and salary-range estimate.
- Modular, maintainable code: `model_loader.py`, `predictor.py`, `utils/encoding.py`, `utils/risk_logic.py` keep the Flask app thin.

## System capabilities (what judges should look for)

- Risk probabilities and levels (deterministic thresholds):
	- Low: risk_probability < 0.3
	- Medium: 0.3 ≤ risk_probability ≤ 0.7
	- High: risk_probability > 0.7
- Structured JSON response containing: `prediction`, `risk_probability`, `risk_level`, `risk_drivers`, `recommended_action`, and `placement_risk_extension`.
- Simple, auditable explainability (transparent heuristics + feature importances).

## API contract

Endpoint: `POST /predict`

Request JSON (applicant features):

```json
{
	"Gender": "Male",
	"Married": "Yes",
	"Dependents": "0",
	"Education": "Graduate",
	"Self_Employed": "No",
	"ApplicantIncome": 5000,
	"CoapplicantIncome": 0,
	"LoanAmount": 100,
	"Loan_Amount_Term": 360,
	"Credit_History": 1,
	"Property_Area": "Urban"
}
```

Response JSON (structured):

```json
{
	"prediction": "Rejected",
	"risk_probability": 0.82,
	"risk_level": "High",
	"risk_drivers": [
		"Low Credit History",
		"High Loan Amount",
		"Low Applicant Income"
	],
	"recommended_action": "Reject",
	"placement_risk_extension": {
		"employment_probability": 0.4,
		"salary_range_estimate": "2000-3000"
	}
}
```

Notes:
- `risk_probability` is computed as `1 - P(approved)` using `predict_proba()` from the RandomForest model (deployed model class label `1` → approved).
- `risk_drivers` are generated from model feature importances combined with simple domain heuristics for clear human-readable explanations.

## Explainability (how drivers are computed)

- The system uses a transparent, two-step method for per-case explanations:
	1. Rank features by `model.feature_importances_` (when available).
	2. Apply simple heuristics (e.g., low credit history, loan amount above median, applicant income below median) to generate human-friendly phrases.

This approach is intentionally lightweight and deterministic for hackathon judges; it is easy to replace with SHAP/LIME in a follow-up.

## Code layout (key files)

- `Code_Files/flask_app.py` — thin Flask app (serves UI and `POST /predict`).
- `Code_Files/model_loader.py` — centralized model loading logic.
- `Code_Files/predictor.py` — prediction service: encodes input, calls model, computes probabilities, risk level, drivers, and recommended action.
- `Code_Files/utils/encoding.py` — input schema and encoders.
- `Code_Files/utils/risk_logic.py` — risk thresholds, driver heuristics, and placement-risk placeholder logic.
- `Code_Files/Static/` — web UI (`index.html`, `script.js`, `style.css`) updated to a decision-dashboard view.
- `Code_Files/loan_prediction.csv` — sample dataset used for development and medians used in heuristics.

## Model & evaluation

- Pipeline highlights: preprocessing, class-imbalance handling (SMOTE), model selection (Random Forest deployed, XGBoost experimented), cross-validation and metrics including ROC‑AUC and F1-score. Best observed ROC‑AUC in experiments: 0.87.
- The deployable model file: `Code_Files/model_randomforest.pkl`. The Flask app lazily loads this file at runtime.

## Run locally (quick)

1. Ensure `Code_Files/model_randomforest.pkl` is present. If you do not have a model, use the notebooks in `Code_Files/` to retrain and save the model.
2. Create a virtual environment and install runtime dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

3. Start the Flask service:

```bash
cd Code_Files
python flask_app.py
```

4. Open the decision dashboard at `http://127.0.0.1:5000/` or POST to `/predict` programmatically.

## Quick smoke test (python)

Run the following from repository root to exercise the `predictor` directly (uses `Code_Files` in `sys.path`):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'Code_Files'))
from predictor import predict

sample = { ... }
print(predict(sample))
```

## Limitations

- Models are trained on a static sample dataset — production performance requires richer features and ongoing monitoring.
- Not a real-time scoring pipeline; integration with streaming financial signals would be needed for live risk updates.
- Explainability is lightweight (heuristics + importances); add SHAP/LIME for deeper attribution when required.

## Future work & extension

- Integrate real-time data sources and credit bureau feeds for fresher risk estimates.
- Add SHAP/LIME explainability and an interactive rationale panel for reviewers.
- Extend to placement‑risk modeling: the repo includes a placeholder `placement_risk_extension` showing employment probability and salary estimate to demonstrate extensibility.
- Add Dockerfile, CI, model versioning and monitoring for production readiness.

## Contribution & Git

- I updated the README to reflect the production-grade upgrade and pushed code changes to the repository. To reproduce locally:

```bash
git add -A
git commit -m "Upgrade: production-grade Credit Risk Intelligence (risk probs, explainability, modular)"
git push
```

## License & contact

Provided for hackathon and educational use. For questions or collaboration, contact `smartlender@example.com`.
