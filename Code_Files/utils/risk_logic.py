import os
from typing import Dict, List

import pandas as pd


def _data_csv_path():
    # CSV is expected to live in Code_Files/ (one level up from utils)
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base, 'loan_prediction.csv')


def get_dataset_stats() -> Dict[str, float]:
    """Compute simple median statistics used by heuristics. Returns dict of medians.

    If dataset is missing or parsing fails, medians will be None.
    """
    stats = {
        'ApplicantIncome_median': None,
        'CoapplicantIncome_median': None,
        'LoanAmount_median': None,
        'Loan_Amount_Term_median': None
    }
    csv_path = _data_csv_path()
    try:
        df = pd.read_csv(csv_path)
        for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                med = series.median(skipna=True)
                stats[f'{col}_median'] = float(med) if pd.notna(med) else None
    except Exception:
        # keep defaults None
        pass
    return stats


def compute_risk_level(risk_prob: float) -> str:
    if risk_prob is None:
        return 'Unknown'
    if risk_prob < 0.3:
        return 'Low'
    if risk_prob <= 0.7:
        return 'Medium'
    return 'High'


def recommend_action(risk_prob: float) -> str:
    if risk_prob is None:
        return 'Review Manually'
    if risk_prob < 0.3:
        return 'Approve'
    if risk_prob <= 0.7:
        return 'Review Manually'
    return 'Reject'


def generate_risk_drivers(model, input_dict: Dict, feature_names: List[str], stats: Dict, top_n: int = 3) -> List[str]:
    """Return top risk driver strings using a simple importance+heuristic approach.

    - Use model.feature_importances_ to rank features (if available)
    - For each top feature, apply a small heuristic to produce a human-friendly phrase
    - If fewer than `top_n` risk-phrases are identified, fill with top feature/value pairs
    """
    drivers = []
    try:
        import numpy as np

        fi = getattr(model, 'feature_importances_', None)
        if fi is None:
            indices = list(range(len(feature_names)))
        else:
            indices = list(np.argsort(fi)[::-1])

        for idx in indices:
            fname = feature_names[idx]
            val = input_dict.get(fname)
            phrase = None

            # small, transparent heuristics used for dashboard explanation
            if fname == 'Credit_History':
                try:
                    if int(float(val)) == 0:
                        phrase = 'Low Credit History'
                except Exception:
                    pass

            elif fname == 'LoanAmount':
                med = stats.get('LoanAmount_median')
                try:
                    if med is not None and float(val) >= med:
                        phrase = 'High Loan Amount'
                except Exception:
                    pass

            elif fname == 'ApplicantIncome':
                med = stats.get('ApplicantIncome_median')
                try:
                    if med is not None and float(val) <= med:
                        phrase = 'Low Applicant Income'
                except Exception:
                    pass

            elif fname == 'CoapplicantIncome':
                med = stats.get('CoapplicantIncome_median')
                try:
                    if med is not None and float(val) <= med:
                        phrase = 'Low Coapplicant Income'
                except Exception:
                    pass

            elif fname == 'Dependents':
                try:
                    if int(float(val)) >= 2:
                        phrase = 'Multiple Dependents'
                except Exception:
                    pass

            elif fname == 'Self_Employed':
                if str(val).lower() in ['yes', '1', 'true']:
                    phrase = 'Self Employed (income variability)'

            elif fname == 'Education':
                if str(val).lower() in ['not graduate', '0']:
                    phrase = 'Not Graduate'

            elif fname == 'Property_Area':
                if str(val).lower() == 'rural':
                    phrase = 'Rural Property Area'

            if phrase:
                drivers.append(phrase)
            if len(drivers) >= top_n:
                break

        # Fill with top feature/value text if we don't have enough risk-phrases
        if len(drivers) < top_n:
            for idx in indices:
                fname = feature_names[idx]
                val = input_dict.get(fname)
                text = f"{fname}: {val}"
                if text not in drivers:
                    drivers.append(text)
                if len(drivers) >= top_n:
                    break

    except Exception:
        # on any error, return empty list
        return []

    return drivers


def placement_risk_extension(input_dict: Dict, stats: Dict) -> Dict:
    """Simple placeholder logic to demonstrate extensibility toward placement-risk modeling.

    Returns:
      - employment_probability: mock 0-1 score
      - salary_range_estimate: a rough estimate string
    """
    try:
        ai = float(input_dict.get('ApplicantIncome', 0) or 0)
    except Exception:
        ai = 0.0

    med = stats.get('ApplicantIncome_median')
    if med is None:
        emp_prob = 0.6 if ai > 0 else 0.5
    else:
        emp_prob = 0.8 if ai >= med else 0.4

    # salary range estimate: +/- 20% of reported applicant income (mock)
    low = int(ai * 0.8)
    high = int(ai * 1.2) if ai > 0 else 0
    salary_range = f"{low}-{high}" if high > 0 else "unknown"

    return {'employment_probability': round(float(emp_prob), 2), 'salary_range_estimate': salary_range}
