from sklearn.ensemble import GradientBoostingClassifier

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="gradient_boosting",
    label="Gradient Boosting",
    builder=lambda random_state, _n_jobs: GradientBoostingClassifier(
        random_state=random_state,
    ),
    use_balanced_sample_weight=True,
)
