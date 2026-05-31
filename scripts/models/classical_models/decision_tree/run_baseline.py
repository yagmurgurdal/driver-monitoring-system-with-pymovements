from scripts.models.classical_models.scenario_runner import PROJECT_ROOT, run_named_model_cli


if __name__ == "__main__":
    run_named_model_cli(
        description="Run the Decision Tree baseline experiment.",
        model_key="decision_tree",
        feature_set="baseline",
        use_high_confidence=False,
        default_window_root=str(PROJECT_ROOT / "window_dataset"),
        default_output_dir=str(PROJECT_ROOT / "results" / "decision_tree_baseline"),
    )
