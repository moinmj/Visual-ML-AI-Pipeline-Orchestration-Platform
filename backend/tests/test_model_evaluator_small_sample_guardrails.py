import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest
import numpy as np
import pandas as pd
from backend.app.recipes.evaluation.model_evaluator import ModelEvaluatorRecipe


class MockTestingTeamClassifier:
    """Mock classifier replicating the exact scenario reported by the testing team:
    - 3 test samples
    - True labels: [0, 0, 1]
    - Predicted probs:
        sample 0 (true 0): [0.525, 0.475] -> pred 0 (correct)
        sample 1 (true 0): [0.260, 0.740] -> pred 1 (misclassified!)
        sample 2 (true 1): [0.150, 0.850] -> pred 1 (correct)
    - Acc = 2/3 = 0.6667
    - ROC-AUC = 1.0000 (positive sample prob 0.850 > negative sample probs 0.475, 0.740)
    - Log loss = (-ln(0.525) - ln(0.260) - ln(0.850)) / 3
               = (0.6444 + 1.3471 + 0.1625) / 3 = 2.1540 / 3 = 0.7180 > 0.6931 (random chance baseline)
    """
    def __init__(self, probs=None):
        if probs is None:
            # Calibrated to produce ~0.7368 log loss as observed by testing team
            # -ln(0.5150) = 0.6636
            # -ln(0.2520) = 1.3783
            # -ln(0.8450) = 0.1684
            # Sum = 2.2103 / 3 = 0.7368
            self.probs = np.array([
                [0.5150, 0.4850],
                [0.2520, 0.7480],
                [0.1550, 0.8450]
            ])
        else:
            self.probs = np.array(probs)

    def predict_proba(self, X):
        return self.probs[:len(X)]

    def predict(self, X):
        return (self.probs[:len(X), 1] >= 0.5).astype(int)


def test_testing_team_exact_small_sample_deceptive_roc_and_uncalibrated_log_loss():
    """Verify that the evaluator detects and flags:
    1. Small sample size (N=3 < 30)
    2. Deceptive ROC-AUC (1.0000 despite 33.3% misclassification / Acc=0.6667)
    3. Uncalibrated Log Loss (0.7368 > random baseline 0.6931)
    """
    evaluator = ModelEvaluatorRecipe()

    X_test = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": [10.0, 20.0, 30.0]
    })
    y_test = pd.Series([0, 0, 1])

    model = MockTestingTeamClassifier()

    result = evaluator.execute(
        inputs={
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "classification"
        },
        config={
            "min_sample_threshold": 30,
            "suppress_unreliable_curves": False
        }
    )

    metrics = result["metrics"]
    report = result["report"]

    # 1. Verification of basic metrics
    assert metrics["sample_size"] == 3
    assert metrics["is_small_sample"] is True
    assert metrics["accuracy"] == 0.6667
    assert metrics["roc_auc"] == 1.0000
    assert metrics["log_loss"] == 0.7368

    # 2. Guardrails verification
    guardrails = metrics["small_sample_guardrails"]
    assert guardrails["triggered"] is True
    assert guardrails["sample_size"] == 3
    assert guardrails["min_recommended_sample_size"] == 30
    assert guardrails["minority_class_count"] == 1
    assert guardrails["roc_auc_reliable"] is False
    assert guardrails["roc_auc_deceptive"] is True
    assert guardrails["log_loss_exceeds_random_chance"] is True
    assert guardrails["random_chance_baseline_log_loss"] == 0.6931
    assert guardrails["calibration_status"] == "UNCALIBRATED_OVERCONFIDENT"
    assert len(guardrails["recommendations"]) >= 2

    # 3. Warnings verification
    warnings = metrics["warnings"]
    assert len(warnings) >= 3
    warn_text = " ".join(warnings)
    assert "DECEPTIVE ROC-AUC" in warn_text
    assert "UNCALIBRATED LOG LOSS" in warn_text
    assert "SMALL SAMPLE" in warn_text

    # 4. Human-readable report verification
    assert "report" in result
    assert "=== Model Evaluation Report" in report
    assert "[DECEPTIVE - Single-step rank artifact]" in report
    assert "[UNCALIBRATED - Exceeds random baseline 0.6931]" in report
    assert "Diagnostics & Guardrail Warnings" in report


def test_small_sample_suppress_unreliable_curves():
    """Verify that when suppress_unreliable_curves is True,
    unreliable curve metrics (ROC-AUC) are suppressed to None while keeping raw metrics in metadata.
    """
    evaluator = ModelEvaluatorRecipe()

    X_test = pd.DataFrame({"f": [1, 2, 3]})
    y_test = pd.Series([0, 0, 1])
    model = MockTestingTeamClassifier()

    result = evaluator.execute(
        inputs={
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "classification"
        },
        config={
            "min_sample_threshold": 30,
            "suppress_unreliable_curves": True
        }
    )

    metrics = result["metrics"]
    report = result["report"]

    assert metrics["roc_auc"] is None
    assert metrics["roc_auc_raw"] == 1.0000
    assert metrics["roc_auc_suppressed"] is True
    assert "Suppressed (statistically unreliable, N=3 < 30)" in report


def test_adequate_sample_size_reliable_metrics():
    """Verify that with N >= 30, standard metrics are considered reliable and no deceptive warnings trigger."""
    evaluator = ModelEvaluatorRecipe()
    np.random.seed(42)

    N = 60
    X_test = pd.DataFrame({"f": np.random.randn(N)})
    y_test = pd.Series([0] * 30 + [1] * 30)

    # Well-calibrated synthetic predictions
    probs = np.zeros((N, 2))
    probs[:30, 0] = np.random.uniform(0.6, 0.9, 30)
    probs[:30, 1] = 1.0 - probs[:30, 0]
    probs[30:, 1] = np.random.uniform(0.6, 0.9, 30)
    probs[30:, 0] = 1.0 - probs[30:, 1]

    model = MockTestingTeamClassifier(probs=probs)

    result = evaluator.execute(
        inputs={
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "classification"
        },
        config={
            "min_sample_threshold": 30
        }
    )

    metrics = result["metrics"]
    assert metrics["sample_size"] == 60
    assert metrics["is_small_sample"] is False
    assert metrics["roc_auc_reliable"] is True
    assert metrics["roc_auc_deceptive"] is False
    assert metrics["is_uncalibrated_log_loss"] is False
    assert metrics["small_sample_guardrails"]["calibration_status"] in ["WELL_CALIBRATED", "CALIBRATED"]
    assert len(metrics["warnings"]) == 0


def test_custom_min_sample_threshold_configuration():
    """Verify that users can customize min_sample_threshold to fit their team's statistical requirements."""
    evaluator = ModelEvaluatorRecipe()

    X_test = pd.DataFrame({"f": range(15)})
    y_test = pd.Series([0] * 7 + [1] * 8)
    probs = np.zeros((15, 2))
    probs[:7, 0] = 0.8
    probs[:7, 1] = 0.2
    probs[7:, 1] = 0.8
    probs[7:, 0] = 0.2
    model = MockTestingTeamClassifier(probs=probs)

    # Threshold = 20 -> N=15 triggers warning
    res1 = evaluator.execute(
        inputs={"model": model, "X_test": X_test, "y_test": y_test},
        config={"min_sample_threshold": 20}
    )
    assert res1["metrics"]["is_small_sample"] is True

    # Threshold = 10 -> N=15 does NOT trigger small-sample warning
    res2 = evaluator.execute(
        inputs={"model": model, "X_test": X_test, "y_test": y_test},
        config={"min_sample_threshold": 10}
    )
    assert res2["metrics"]["is_small_sample"] is False


def test_single_class_in_small_sample_handled_gracefully():
    """Verify that small test splits containing only 1 class emit a clear warning without throwing."""
    evaluator = ModelEvaluatorRecipe()

    X_test = pd.DataFrame({"f": [1, 2, 3]})
    y_test = pd.Series([0, 0, 0])
    probs = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.9, 0.1]
    ])
    model = MockTestingTeamClassifier(probs=probs)

    result = evaluator.execute(
        inputs={"model": model, "X_test": X_test, "y_test": y_test},
        config={"min_sample_threshold": 30}
    )

    metrics = result["metrics"]
    assert metrics["is_small_sample"] is True
    warn_text = " ".join(metrics["warnings"])
    assert "SINGLE CLASS" in warn_text


def test_regression_small_sample_guardrails():
    """Verify regression models also flag small sample size (N < threshold) with diagnostic warning."""
    evaluator = ModelEvaluatorRecipe()

    class MockRegressor:
        def predict(self, X):
            return np.array([10.0, 20.0, 30.0])

    X_test = pd.DataFrame({"x": [1, 2, 3]})
    y_test = pd.Series([11.0, 19.0, 32.0])

    result = evaluator.execute(
        inputs={
            "model": MockRegressor(),
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "regression"
        },
        config={"min_sample_threshold": 30}
    )

    metrics = result["metrics"]
    assert metrics["is_small_sample"] is True
    assert metrics["sample_size"] == 3
    assert metrics["small_sample_guardrails"]["triggered"] is True
    assert "[WARNING: SMALL SAMPLE]" in metrics["warnings"][0]
    assert "=== Model Evaluation Report" in result["report"]


def test_regression_mape_zeros_in_actuals_reported_as_na_with_smape():
    """Verify the exact defect reported by the testing team (Crazy 14 Trillion % MAPE):
    When actual ground truth target contains zero, standard MAPE divides by zero (or scikit-learn's tiny epsilon)
    exploding to trillions.
    The evaluator must:
    1. Set metrics['mape'] to None
    2. Set metrics['mape_display'] to 'N/A (Contains zeros)'
    3. Calculate SMAPE (Symmetric MAPE) as a robust zero-safe alternative bounded in [0%, 200%]
    4. Emit a clear [WARNING: ZERO IN ACTUALS] in warnings and human-readable report.
    """
    evaluator = ModelEvaluatorRecipe()

    class MockRegressor:
        def predict(self, X):
            return np.array([10.0, 20.0, 30.0])

    X_test = pd.DataFrame({"x": [1, 2, 3]})
    # Ground truth contains 0.0 (causes standard MAPE division explosion)
    y_test = pd.Series([0.0, 19.0, 32.0])

    result = evaluator.execute(
        inputs={
            "model": MockRegressor(),
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "regression"
        },
        config={"min_sample_threshold": 2}
    )

    metrics = result["metrics"]
    report = result["report"]

    # 1. MAPE must NOT explode into trillions or numbers
    assert metrics["mape"] is None
    assert metrics["mape_display"] == "N/A (Contains zeros)"
    assert metrics["mape_status"] == "UNDEFINED_ZERO_DIVISION"
    assert metrics["has_zero_actuals"] is True

    # 2. SMAPE must be calculated cleanly and bounded
    assert metrics["smape"] is not None
    assert 0.0 <= metrics["smape"] <= 200.0
    # Expected SMAPE:
    # term 0: 2*|0-10| / (0+10) = 20/10 = 2.0 (200%)
    # term 1: 2*|19-20| / (19+20) = 2/39 = 0.05128 (5.13%)
    # term 2: 2*|32-30| / (32+30) = 4/62 = 0.06452 (6.45%)
    # mean: (2.0 + 0.05128 + 0.06452) / 3 = 0.70526 -> 70.526%
    assert metrics["smape"] == pytest.approx(70.526, abs=0.01)
    assert metrics["smape_display"] == "70.53%"

    # 3. Report formatting checks
    assert "MAPE               : N/A (Contains zeros)" in report
    assert f"SMAPE (Zero-Safe)  : {metrics['smape']:.4f}%" in report
    assert "[WARNING: ZERO IN ACTUALS]" in report
    assert "Trillion" not in report
    assert "1e+" not in report


def test_regression_mape_all_zeros_handled_safely():
    """Verify all-zero ground truth and prediction edge case does not crash or raise ZeroDivisionError."""
    evaluator = ModelEvaluatorRecipe()

    class MockRegressor:
        def predict(self, X):
            return np.array([0.0, 0.0, 0.0])

    X_test = pd.DataFrame({"x": [1, 2, 3]})
    y_test = pd.Series([0.0, 0.0, 0.0])

    result = evaluator.execute(
        inputs={
            "model": MockRegressor(),
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "regression"
        },
        config={"min_sample_threshold": 2}
    )

    metrics = result["metrics"]
    report = result["report"]

    assert metrics["mape"] is None
    assert metrics["mape_display"] == "N/A (Contains zeros)"
    assert metrics["smape"] == 0.0
    assert "MAPE               : N/A (Contains zeros)" in report
    assert "SMAPE (Zero-Safe)  : 0.0000%" in report


def test_regression_mape_valid_non_zero_actuals():
    """Verify standard regression with non-zero targets correctly computes standard percentage MAPE and SMAPE."""
    evaluator = ModelEvaluatorRecipe()

    class MockRegressor:
        def predict(self, X):
            return np.array([110.0, 190.0, 330.0])

    X_test = pd.DataFrame({"x": [1, 2, 3]})
    y_test = pd.Series([100.0, 200.0, 300.0])

    result = evaluator.execute(
        inputs={
            "model": MockRegressor(),
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "regression"
        },
        config={"min_sample_threshold": 2}
    )

    metrics = result["metrics"]
    report = result["report"]

    # |100-110|/100 = 0.10, |200-190|/200 = 0.05, |300-330|/300 = 0.10 -> mean = 0.0833 (8.33%)
    assert metrics["mape"] == pytest.approx(0.0833, abs=0.001)
    assert metrics["mape_display"] == "8.33%"
    assert metrics["mape_status"] == "VALID"
    assert metrics["has_zero_actuals"] is False
    assert metrics["smape"] is not None
    assert "MAPE               : 8.33%" in report
    assert "SMAPE (Zero-Safe)" in report




def test_regression_runaway_mape_near_zeros_guarded():
    """Verify extreme near-zero actuals (e.g. 1e-12) do not explode MAPE into trillions."""
    evaluator = ModelEvaluatorRecipe()

    class MockRegressor:
        def predict(self, X):
            return np.array([10.0, 100.0])

    X_test = pd.DataFrame({"x": [1, 2]})
    # 1e-12 is within atol=1e-9 of 0.0
    y_test = pd.Series([1e-12, 100.0])

    result = evaluator.execute(
        inputs={
            "model": MockRegressor(),
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "regression"
        },
        config={"min_sample_threshold": 2}
    )

    metrics = result["metrics"]
    report = result["report"]

    assert metrics["mape"] is None
    assert metrics["mape_display"] == "N/A (Contains zeros)"
    assert metrics["has_zero_actuals"] is True
    assert "Trillion" not in report
    assert "1e+" not in report

if __name__ == "__main__":
    print("Running test_testing_team_exact_small_sample_deceptive_roc_and_uncalibrated_log_loss...")
    test_testing_team_exact_small_sample_deceptive_roc_and_uncalibrated_log_loss()
    print("PASS: test_testing_team_exact_small_sample_deceptive_roc_and_uncalibrated_log_loss")

    print("Running test_small_sample_suppress_unreliable_curves...")
    test_small_sample_suppress_unreliable_curves()
    print("PASS: test_small_sample_suppress_unreliable_curves")

    print("Running test_adequate_sample_size_reliable_metrics...")
    test_adequate_sample_size_reliable_metrics()
    print("PASS: test_adequate_sample_size_reliable_metrics")

    print("Running test_custom_min_sample_threshold_configuration...")
    test_custom_min_sample_threshold_configuration()
    print("PASS: test_custom_min_sample_threshold_configuration")

    print("Running test_single_class_in_small_sample_handled_gracefully...")
    test_single_class_in_small_sample_handled_gracefully()
    print("PASS: test_single_class_in_small_sample_handled_gracefully")

    print("Running test_regression_small_sample_guardrails...")
    test_regression_small_sample_guardrails()
    print("PASS: test_regression_small_sample_guardrails")

    print("\nALL 6 GUARDRAIL UNIT TESTS PASSED SUCCESSFULLY!")
