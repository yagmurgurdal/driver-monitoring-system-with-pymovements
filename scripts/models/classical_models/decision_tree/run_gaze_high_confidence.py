from scripts.models.classical_models.scenario_runner import PROJECT_ROOT, run_named_model_cli


if __name__ == "__main__":
    run_named_model_cli(
        description="Run the Decision Tree gaze high-confidence experiment.",
        model_key="decision_tree",
        feature_set="gaze",
        use_high_confidence=True,
        default_window_root=str(PROJECT_ROOT / "window_dataset_with_gaze"),
        default_output_dir=str(PROJECT_ROOT / "results" / "decision_tree_gaze_high_confidence"),
    )
