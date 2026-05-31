from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="rbf_svm",
    label="RBF SVM",
    builder=lambda random_state, _n_jobs: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=False,
                    random_state=random_state,
                ),
            ),
        ]
    ),
)
