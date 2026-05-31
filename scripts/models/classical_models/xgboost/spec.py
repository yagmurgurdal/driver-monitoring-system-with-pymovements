from xgboost import XGBClassifier

from scripts.models.classical_models.common import ModelSpec


MODEL_SPEC = ModelSpec(
    key="xgboost",
    label="XGBoost",
    builder=lambda random_state, n_jobs: XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    ),
)
