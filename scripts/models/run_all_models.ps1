param(
    [ValidateSet("all", "random_forest", "adaboost", "decision_tree", "extra_trees", "gradient_boosting", "knn", "logistic_regression", "linear_svm", "rbf_svm", "xgboost")]
    [string[]]$Models = @("all"),

    [ValidateSet("all", "baseline", "high_confidence", "gaze_baseline", "gaze_high_confidence")]
    [string[]]$Scenarios = @("all"),

    [int]$NJobs = 1
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

function Test-InSelection {
    param(
        [string]$Value,
        [string[]]$Selection
    )

    return $Selection -contains "all" -or $Selection -contains $Value
}

$jobsByModel = @{
    random_forest = 1
    adaboost = $NJobs
    decision_tree = $NJobs
    extra_trees = $NJobs
    gradient_boosting = $NJobs
    knn = $NJobs
    logistic_regression = $NJobs
    linear_svm = $NJobs
    rbf_svm = $NJobs
    xgboost = $NJobs
}

$runs = @(
    @{ Model = "random_forest"; Scenario = "baseline";             Module = "scripts.models.random_forest.train_rf_baseline" },
    @{ Model = "random_forest"; Scenario = "high_confidence";      Module = "scripts.models.random_forest.train_rf_high_confidence" },
    @{ Model = "random_forest"; Scenario = "gaze_baseline";        Module = "scripts.models.random_forest.train_rf_gaze_baseline" },
    @{ Model = "random_forest"; Scenario = "gaze_high_confidence"; Module = "scripts.models.random_forest.train_rf_gaze_high_confidence" },

    @{ Model = "adaboost"; Scenario = "baseline";             Module = "scripts.models.classical_models.adaboost.run_baseline" },
    @{ Model = "adaboost"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.adaboost.run_high_confidence" },
    @{ Model = "adaboost"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.adaboost.run_gaze_baseline" },
    @{ Model = "adaboost"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.adaboost.run_gaze_high_confidence" },

    @{ Model = "decision_tree"; Scenario = "baseline";             Module = "scripts.models.classical_models.decision_tree.run_baseline" },
    @{ Model = "decision_tree"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.decision_tree.run_high_confidence" },
    @{ Model = "decision_tree"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.decision_tree.run_gaze_baseline" },
    @{ Model = "decision_tree"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.decision_tree.run_gaze_high_confidence" },

    @{ Model = "extra_trees"; Scenario = "baseline";             Module = "scripts.models.classical_models.extra_trees.run_baseline" },
    @{ Model = "extra_trees"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.extra_trees.run_high_confidence" },
    @{ Model = "extra_trees"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.extra_trees.run_gaze_baseline" },
    @{ Model = "extra_trees"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.extra_trees.run_gaze_high_confidence" },

    @{ Model = "gradient_boosting"; Scenario = "baseline";             Module = "scripts.models.classical_models.gradient_boosting.run_baseline" },
    @{ Model = "gradient_boosting"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.gradient_boosting.run_high_confidence" },
    @{ Model = "gradient_boosting"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.gradient_boosting.run_gaze_baseline" },
    @{ Model = "gradient_boosting"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.gradient_boosting.run_gaze_high_confidence" },

    @{ Model = "knn"; Scenario = "baseline";             Module = "scripts.models.classical_models.knn.run_baseline" },
    @{ Model = "knn"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.knn.run_high_confidence" },
    @{ Model = "knn"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.knn.run_gaze_baseline" },
    @{ Model = "knn"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.knn.run_gaze_high_confidence" },

    @{ Model = "logistic_regression"; Scenario = "baseline";             Module = "scripts.models.classical_models.logistic_regression.run_baseline" },
    @{ Model = "logistic_regression"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.logistic_regression.run_high_confidence" },
    @{ Model = "logistic_regression"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.logistic_regression.run_gaze_baseline" },
    @{ Model = "logistic_regression"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.logistic_regression.run_gaze_high_confidence" },

    @{ Model = "linear_svm"; Scenario = "baseline";             Module = "scripts.models.classical_models.linear_svm.run_baseline" },
    @{ Model = "linear_svm"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.linear_svm.run_high_confidence" },
    @{ Model = "linear_svm"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.linear_svm.run_gaze_baseline" },
    @{ Model = "linear_svm"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.linear_svm.run_gaze_high_confidence" },

    @{ Model = "rbf_svm"; Scenario = "baseline";             Module = "scripts.models.classical_models.rbf_svm.run_baseline" },
    @{ Model = "rbf_svm"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.rbf_svm.run_high_confidence" },
    @{ Model = "rbf_svm"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.rbf_svm.run_gaze_baseline" },
    @{ Model = "rbf_svm"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.rbf_svm.run_gaze_high_confidence" },

    @{ Model = "xgboost"; Scenario = "baseline";             Module = "scripts.models.classical_models.xgboost.run_baseline" },
    @{ Model = "xgboost"; Scenario = "high_confidence";      Module = "scripts.models.classical_models.xgboost.run_high_confidence" },
    @{ Model = "xgboost"; Scenario = "gaze_baseline";        Module = "scripts.models.classical_models.xgboost.run_gaze_baseline" },
    @{ Model = "xgboost"; Scenario = "gaze_high_confidence"; Module = "scripts.models.classical_models.xgboost.run_gaze_high_confidence" }
)

$selectedRuns = $runs | Where-Object {
    (Test-InSelection -Value $_.Model -Selection $Models) -and
    (Test-InSelection -Value $_.Scenario -Selection $Scenarios)
}

if (-not $selectedRuns) {
    throw "No runs matched the selected model/scenario filters."
}

Write-Host "Repository root: $repoRoot"
Write-Host "Python: $pythonExe"
Write-Host "Selected runs: $($selectedRuns.Count)"

foreach ($run in $selectedRuns) {
    $modelJobs = $jobsByModel[$run.Model]
    if (-not $modelJobs) {
        $modelJobs = 1
    }

    Write-Host ""
    Write-Host "=== $($run.Model) / $($run.Scenario) ==="
    Write-Host "Module: $($run.Module)"

    $commandArgs = @("-m", $run.Module)
    if ($run.Model -ne "random_forest") {
        $commandArgs += @("--n-jobs", $modelJobs)
    }

    & $pythonExe @commandArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $($run.Model) / $($run.Scenario) with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "All selected runs completed successfully."
