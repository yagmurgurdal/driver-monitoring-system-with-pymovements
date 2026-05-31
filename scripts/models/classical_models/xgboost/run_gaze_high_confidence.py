from scripts.models.classical_models.scenario_runner import PROJECT_ROOT, run_named_model_cli


if __name__ == "__main__":
    run_named_model_cli(
        description="Run the XGBoost gaze high-confidence experiment.",
        model_key="xgboost",
        feature_set="gaze",
        use_high_confidence=True,
        default_window_root=str(PROJECT_ROOT / "window_dataset_with_gaze"),
        default_output_dir=str(PROJECT_ROOT / "results" / "xgboost_gaze_high_confidence"),
    )
