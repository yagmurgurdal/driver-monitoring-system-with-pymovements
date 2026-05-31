from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="adaboost",
    label="AdaBoost",
    builder=lambda random_state, _n_jobs: AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            random_state=random_state,
        ),
        n_estimators=300,
        learning_rate=0.05,
        random_state=random_state,
    ),
    use_balanced_sample_weight=True,
)
