from typing import Dict

from model_loader import load_model
from utils.encoding import encode_input, get_feature_names
from utils.risk_logic import (
    get_dataset_stats,
    compute_risk_level,
    recommend_action,
    generate_risk_drivers,
    placement_risk_extension,
)


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        # lazy load so module import doesn't fail if the model isn't present yet
        _MODEL = load_model()
    return _MODEL


def predict(input_dict: Dict):
    """Run inference for a single applicant dictionary.

    Returns a structured dict with:
      - prediction (Approved/Rejected)
      - risk_probability (0..1)
      - risk_level (Low/Medium/High)
      - risk_drivers (list of top 3 strings)
      - recommended_action
      - placement_risk_extension (placeholder dict)

    If the model file is missing, raises FileNotFoundError which the Flask app will report.
    """
    model = _get_model()
    feature_names = get_feature_names()
    stats = get_dataset_stats()

    # encode input
    X = encode_input(input_dict)

    # predict probability and label
    proba = model.predict_proba(X)

    # figure out which column corresponds to approved (class==1)
    classes = list(model.classes_)
    try:
        approved_idx = classes.index(1)
    except ValueError:
        # fallback: assume second column is positive class when available
        approved_idx = 1 if proba.shape[1] > 1 else 0

    prob_approved = float(proba[0][approved_idx])
    risk_probability = round(1.0 - prob_approved, 3)
    risk_level = compute_risk_level(risk_probability)
    recommended = recommend_action(risk_probability)
    pred_raw = model.predict(X)[0]
    prediction = 'Approved' if int(pred_raw) == 1 else 'Rejected'

    # explainability (lightweight)
    risk_drivers = generate_risk_drivers(model, input_dict, feature_names, stats, top_n=3)

    # placement-risk placeholder
    placement = placement_risk_extension(input_dict, stats)

    return {
        'prediction': prediction,
        'risk_probability': risk_probability,
        'risk_level': risk_level,
        'risk_drivers': risk_drivers,
        'recommended_action': recommended,
        'placement_risk_extension': placement,
    }
