import os
import pickle


def load_model(model_path: str = None):
    """Load and return the pickled model. Raises FileNotFoundError if missing."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model_randomforest.pkl')
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as fh:
        model = pickle.load(fh)
    return model
