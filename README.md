# SmartCareer — Placement Risk Engine for Education Loans

AI-driven system linking career outcomes with loan repayment risk

---

## Problem Statement

- Education loans are often approved without visibility into the student's employability or placement timeline.
- Placement delays and low starting salaries can lead directly to early repayment issues and higher delinquency rates.
- Lenders currently lack readily available predictive tools to estimate a student's post-graduation employability and its impact on repayment.

---

## Solution Overview

- Predicts placement probability at 3 / 6 / 12 months and estimates a plausible starting salary range.
- Produces explainable risk outputs (probability, level, top drivers) tied to recommended actions for lenders.
- Designed to support lender decision-making (human-in-the-loop): risk-based review, interventions, or adjusted pricing.

---

## What We Built (CRITICAL)

This is a working, demo-ready implementation intended to demonstrate a production approach. Key components:

- Random Forest model (deployed as a pickled artifact).
- SMOTE used during training to mitigate class imbalance for delayed-placement cases.
- Flask API exposing a `POST /predict` endpoint for inference.
- Risk outputs returned per request: `prediction`, `risk_probability` (0–1), `risk_level` (Low/Medium/High), `risk_drivers` (top-3 explanations), and `recommended_action`.

We avoid exaggerated claims — results shown in examples are demo/pilot estimates intended for validation with real data.

---

## System Architecture

Pipeline (conceptual):

Input → Preprocessing → Model → Risk Engine → Decision Layer → API / UI

- Modular structure: `model_loader.py` (model lifecycle), `predictor.py` (inference + schema), `utils/encoding.py`, `utils/risk_logic.py`.
- Explainability layer produces human-readable drivers; business rules map probabilities → recommended actions.

---

## Features

- Risk probability scoring (numeric 0–1) and risk level labels (Low/Medium/High).
- Explainability: deterministic top‑3 drivers per case (model importances + heuristics).
- Decision recommendation: simple business actions (Approve / Review / Reject).
- Modular codebase enabling easy replacement of model or explainability method.

---

## Sample Output (JSON)

Example API response (example/demo values):

```json
{
	"prediction": "Placed",
	"placement_probabilities": {"3m": 0.42, "6m": 0.78, "12m": 0.95},
	"estimated_salary_range": "₹4,00,000 - ₹6,00,000",
	"risk_probability": 0.21,
	"risk_level": "Low",
	"risk_drivers": [
		"Low internships",
		"CGPA below program median",
		"Institute Tier: 3"
	],
	"recommended_action": "Approve — offer standard terms; suggest placement support",
	"meta": {"model_version": "rf_v1_demo", "note": "example/demo values"}
}
```

---

## Tech Stack

- Backend: Python, Flask
- ML: scikit-learn (RandomForest), imbalanced-learn (SMOTE), NumPy, pandas
- Frontend: simple static dashboard (HTML/CSS/JS) in `Code_Files/Static`

---

## Run Locally

1. Ensure `Code_Files/model_randomforest.pkl` is in place. If missing, see notebooks in `Code_Files/` to retrain.

2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
```

3. Start the API:

```powershell
cd Code_Files
python flask_app.py
# Service listens on http://127.0.0.1:5000 by default
```

4. Example request (curl):

```bash
curl -X POST http://127.0.0.1:5000/predict \
	-H "Content-Type: application/json" \
	-d '{"CGPA":7.2,"Internships":0,"Institute_Tier":3,"ApplicantIncome":4000}'
```

---

## Limitations

- Models are trained on a demo dataset and results are pilot/estimated. Real-world performance requires richer data, validation, and monitoring.
- Not designed as a real-time streaming scorer in the current form — integration work is required for live signals.
- Explainability is intentionally lightweight (feature importances + heuristics); add SHAP/LIME for deeper attribution in production.

---

## Future Scope

- Integrate real placement data and credit bureau feeds for stronger signals and calibration.
- Replace lightweight explainability with SHAP/LIME and add an interactive explanation panel.
- Add CI, Docker, model versioning, and monitoring (data drift, performance alerts).
- Pilot with lenders (1,000+ student cohort) to validate business impact and calibrate thresholds.

---

## Important Rules (documentation tone)

- No emojis; keep language professional and human.
- Avoid fake or unverified claims — label figures as `demo`, `estimated`, or `pilot-ready` where appropriate.
- Use clear, concise bullets and prioritize transparency for judges and reviewers.

---

If you want, I can also add a short API reference and a sample `sample_request.json` in the repository.
