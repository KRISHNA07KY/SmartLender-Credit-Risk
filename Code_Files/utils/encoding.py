import numpy as np

FEATURE_ORDER = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area'
]

ENCODE_MAP = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
}


def get_feature_names():
    return FEATURE_ORDER.copy()


def encode_input(data: dict):
    """Encode input dict into numpy array of shape (1, n_features).

    Raises KeyError if a required field is missing, ValueError for invalid values.
    """
    arr = []
    for f in FEATURE_ORDER:
        if f not in data:
            raise KeyError(f)
        v = data[f]

        # allow numeric values passed directly
        if isinstance(v, (int, float)):
            arr.append(float(v))
            continue

        # categorical fields with mapping
        if f in ENCODE_MAP:
            if v in ENCODE_MAP[f]:
                arr.append(float(ENCODE_MAP[f][v]))
                continue
            # allow numeric string
            try:
                arr.append(float(v))
                continue
            except Exception:
                raise ValueError(f"Cannot encode field {f} value {v}")

        # numeric fields
        try:
            arr.append(float(v))
        except Exception:
            raise ValueError(f"Cannot convert field {f} value {v} to float")

    return np.array([arr], dtype=float)
