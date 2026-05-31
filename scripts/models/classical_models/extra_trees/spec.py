from sklearn.ensemble import ExtraTreesClassifier

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="extra_trees",
    label="Extra Trees",
    builder=lambda random_state, n_jobs: ExtraTreesClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    ),
)
