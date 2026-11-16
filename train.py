import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.predict import MODEL_PATH


ROOT = Path(__file__).resolve().parent
ABT_PATH = ROOT / "data" / "processed" / "logistic_abt.csv"

# Match the final refit in notebooks/03_modeling.ipynb.
REFIT_CUTOFF = pd.Timestamp("2018-06-01")
DATE_COLUMNS = ["order_approved_at", "order_delivered_customer_date"]
NUMERIC_FEATURES = [
    "days_from_approval_to_estimate",
    "item_count",
    "distinct_product_count",
    "distinct_seller_count",
    "price_sum",
    "freight_sum",
    "total_weight_g",
    "total_volume_cm3",
    "max_seller_distance_km",
]
STATE_FEATURES = ["customer_state"]
FEATURE_COLUMNS = NUMERIC_FEATURES + STATE_FEATURES


def train_model(features: pd.DataFrame, labels: pd.Series) -> Pipeline:
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Training requires both on-time (0) and late (1) labels.")

    numeric_preprocessing = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    preprocessing = ColumnTransformer([
        ("numeric", numeric_preprocessing, NUMERIC_FEATURES),
        ("state", OneHotEncoder(handle_unknown="ignore"), STATE_FEATURES),
    ])
    model = make_pipeline(preprocessing, LogisticRegression(C=0.1, max_iter=500))
    return model.fit(features, labels)


def export_model(model: Pipeline, sample_orders: pd.DataFrame) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_probabilities = model.predict_proba(sample_orders)

    with TemporaryDirectory(dir=MODEL_PATH.parent) as directory:
        temporary_path = Path(directory) / MODEL_PATH.name
        with temporary_path.open("wb") as model_file:
            pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

        with temporary_path.open("rb") as model_file:
            reloaded_model = pickle.load(model_file)

        reloaded_probabilities = reloaded_model.predict_proba(sample_orders)
        if not (expected_probabilities == reloaded_probabilities).all():
            raise RuntimeError("Predictions changed after reloading the model.")

        temporary_path.replace(MODEL_PATH)


def main() -> None:
    orders = pd.read_csv(
        ABT_PATH,
        usecols=DATE_COLUMNS + FEATURE_COLUMNS + ["is_late"],
        parse_dates=DATE_COLUMNS,
    )
    refit_orders = orders.loc[
        (orders["order_approved_at"] < REFIT_CUTOFF)
        & (orders["order_delivered_customer_date"] < REFIT_CUTOFF)
    ]
    features = refit_orders[FEATURE_COLUMNS]
    model = train_model(features, refit_orders["is_late"])
    export_model(model, features.head(3))
    print(f"Trained on {len(refit_orders):,} orders and verified the pickle round trip.")
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
