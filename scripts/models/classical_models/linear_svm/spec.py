from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="linear_svm",
    label="Linear SVM",
    builder=lambda random_state, _n_jobs: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    random_state=random_state,
                    max_iter=6000,
                    class_weight="balanced",
                ),
            ),
        ]
    ),
)
