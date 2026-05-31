from sklearn.tree import DecisionTreeClassifier

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="decision_tree",
    label="Decision Tree",
    builder=lambda random_state, _n_jobs: DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
    ),
)
