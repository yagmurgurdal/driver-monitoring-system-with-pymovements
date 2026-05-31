from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="knn",
    label="KNN",
    builder=lambda _random_state, n_jobs: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance",
                    n_jobs=n_jobs,
                ),
            ),
        ]
    ),
)
