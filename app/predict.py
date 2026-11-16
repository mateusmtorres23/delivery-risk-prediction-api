import pickle
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


MODEL_PATH = Path(__file__).resolve().parent / "model" / "delivery_risk.pkl"


def load_model() -> Pipeline:
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def predict_late_probability(
    model: Pipeline, features: dict[str, int | float | str | None]
) -> float:
    order = pd.DataFrame([features], columns=model.feature_names_in_)
    late_class = model.classes_.tolist().index(1)
    return float(model.predict_proba(order)[0, late_class])
