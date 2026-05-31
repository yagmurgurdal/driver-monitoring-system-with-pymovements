from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="logistic_regression",
    label="Logistic Regression",
    builder=lambda random_state, _n_jobs: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=4000,
                    class_weight="balanced",
                ),
            ),
        ]
    ),
)
