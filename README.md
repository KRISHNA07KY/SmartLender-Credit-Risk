# SmartLender — Loan Approval (Credit Risk) Prediction

SmartLender is an end-to-end machine learning project that predicts loan approval (credit risk) for applicants. It includes a trained Random Forest model wrapped by a small Flask API and a simple web frontend so users can interactively submit applicant data and receive approval predictions.

**Key features:**
- Predict loan approval (Approved / Rejected) using a Random Forest model.
- Simple web UI and a REST `/predict` endpoint for programmatic use.
- Example dataset and Jupyter notebooks for preprocessing and model development.

**Tech stack:**
- Python, Flask, NumPy, scikit-learn (model training & inference)
- HTML / CSS / JavaScript (frontend)

## Repository structure

- `Code_Files/` — core project code and assets
	- `flask_app.py` — Flask application exposing `/` and `/predict`
	- `loan_prediction.csv` — example dataset used for training and exploration
	- `datapreprocessing.ipynb` — preprocessing notebook
	- `ML_final (1).ipynb` — model development notebook
	- `Static/` — frontend files used by the Flask app (`index.html`, `script.js`, `style.css`)
- `index.html`, `script.js`, `style.css` — top-level static examples (alternative front-end)

## Dataset

The dataset `Code_Files/loan_prediction.csv` contains typical loan-application features such as `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, and the target `Loan_Status`.

## Model details

The project applies reproducible preprocessing, performs class imbalance handling (SMOTE), and evaluates candidate models (Random Forest and XGBoost) using ROC-AUC and F1-score (best observed ROC-AUC: 0.87). The Flask app loads a pickled Random Forest model (`model_randomforest.pkl`) and performs simple encoding before prediction. Encoding and input order used in `flask_app.py`:

- Gender: `Male` → 1, `Female` → 0
- Married: `Yes` → 1, `No` → 0
- Dependents: `0` → 0, `1` → 1, `2` → 2, `3+` → 3
- Education: `Graduate` → 1, `Not Graduate` → 0
- Self_Employed: `Yes` → 1, `No` → 0
- Property_Area: `Urban` → 2, `Semiurban` → 1, `Rural` → 0

Input vector order (as used for prediction):
`[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]`

## Run locally

1. Make sure the trained model `model_randomforest.pkl` is present in the same folder as `flask_app.py` (typically `Code_Files/`).
2. (Optional) Create and activate a virtual environment and install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install flask numpy scikit-learn pandas
```

3. Start the Flask app:

```bash
cd Code_Files
python flask_app.py
```

4. Open the web UI in a browser at `http://127.0.0.1:5000/` (or use the top-level `index.html` files for a static demo).

## API usage

Endpoint: `POST /predict`

Request JSON (example):

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

Response example:

```json
{ "prediction": "Approved" }
```

## Notes & next steps

- If you want to retrain the model, use the notebooks in `Code_Files/` to preprocess `loan_prediction.csv`, train, evaluate, and re-save a new `model_randomforest.pkl`.
- Add a `requirements.txt` to pin Python dependencies for reproducible setup.

## License & contact

This repository is provided for educational purposes. For questions or collaboration, contact `smartlender@example.com`.
