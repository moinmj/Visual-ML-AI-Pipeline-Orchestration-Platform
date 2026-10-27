import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import is_classifier, is_regressor
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ModelCalibrationRecipe(BaseRecipe):
    """
    Model Probability Calibration Recipe (Platt Scaling & Isotonic Regression).
    Calibrates raw machine learning model prediction probabilities using Platt Scaling (Sigmoid)
    or Isotonic Regression (non-parametric monotonic fit) via CalibratedClassifierCV.
    Calculates Expected Calibration Error (ECE), before/after Brier scores and Log Loss,
    and returns a newly calibrated, production-ready estimator.
    """

    recipe_id = "model_calibration"
    name = "Model Calibration (Platt / Isotonic)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Deep Learning / AutoML"
    description = (
        "Calibrates classification probabilities so that confidence matches real-world empirical frequency. "
        "Supports Platt Scaling (sigmoid) and Isotonic Regression, producing reliability curves and reduced Log Loss."
    )
    input_types = ["model", "train_data"]
    output_types = ["model", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Trained Classifier",
            type="model",
            required=True,
            max_connections=1,
            description="Trained classifier with predict_proba or decision_function (XGBoost, LightGBM, RF, SVM, etc.)"
        ),
        RecipePort(
            id="calibration_data",
            label="Calibration / Holdout Data",
            type="train_data",
            required=False,
            max_connections=1,
            description="Validation dataset containing X and y holdout partitions used to fit the calibration curve"
        ),
        RecipePort(
            id="dataframe",
            label="Evaluation Table",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Alternative table input containing features and target column"
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Calibrated Model",
            type="model",
            description="Fitted CalibratedClassifierCV instance returning calibrated probabilities"
        ),
        RecipePort(
            id="metrics",
            label="Calibration Metrics & Reliability Curves",
            type="metrics",
            description="Before/after Brier scores, Log Loss, Expected Calibration Error (ECE), and reliability bins"
        ),
        RecipePort(
            id="report",
            label="Calibration Governance Report",
            type="report",
            description="Text report detailing calibration improvements and risk reduction"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Calibration Algorithm",
                    "input_type": "dropdown",
                    "enum": ["sigmoid", "isotonic"],
                    "default": "sigmoid",
                    "description": "'sigmoid' (Platt Scaling) fits a logistic regression curve on model logits (best for smaller N). 'isotonic' fits a non-parametric monotonic step function (best for N > 1000)."
                },
                "cv_strategy": {
                    "type": "string",
                    "title": "Cross-Validation Strategy",
                    "input_type": "dropdown",
                    "enum": ["prefit", "3-fold", "5-fold"],
                    "default": "prefit",
                    "description": "'prefit' uses the incoming model as-is and fits calibration purely on incoming data. '3-fold' or '5-fold' refits the base model via cross-validation folds."
                },
                "num_bins": {
                    "type": "integer",
                    "title": "Reliability Curve Bins",
                    "input_type": "number",
                    "default": 10,
                    "minimum": 3,
                    "maximum": 30,
                    "description": "Number of probability bins used to compute reliability diagrams and Expected Calibration Error (ECE)."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Column",
                    "input_type": "text",
                    "default": "",
                    "description": "Name of the target column if passing a unified table."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Model
        model = inputs.get("model") or inputs.get("trained_model")
        if model is None and isinstance(context, dict):
            model = context.get("model") or context.get("trained_model")
        if model is None:
            raise ValueError("ModelCalibration expects a trained classifier 'model' in inputs.")

        # Guard: Check model supports classification
        if is_regressor(model) or (hasattr(model, "_estimator_type") and model._estimator_type == "regressor"):
            raise ValueError("ModelCalibration only applies to classification models, not regressors.")
        if not (is_classifier(model) or hasattr(model, "predict_proba") or hasattr(model, "decision_function")):
            raise ValueError("ModelCalibration only applies to classification models, not regressors.")

        # 2. Resolve Calibration Data (X_calib, y_calib)
        X_calib = inputs.get("X_calib")
        if X_calib is None:
            X_calib = inputs.get("X_test")
        if X_calib is None:
            X_calib = inputs.get("X_train")

        y_calib = inputs.get("y_calib")
        if y_calib is None:
            y_calib = inputs.get("y_test")
        if y_calib is None:
            y_calib = inputs.get("y_train")

        # Check calibration_data port or train_data port
        calib_data = inputs.get("calibration_data")
        if calib_data is None:
            calib_data = inputs.get("train_data")
        if calib_data is None:
            calib_data = inputs.get("test_data")

        if isinstance(calib_data, dict):
            if X_calib is None:
                for k in ("X_test", "X_train", "X"):
                    if calib_data.get(k) is not None:
                        X_calib = calib_data[k]
                        break
            if y_calib is None:
                for k in ("y_test", "y_train", "y"):
                    if calib_data.get(k) is not None:
                        y_calib = calib_data[k]
                        break

        target_col = config.get("target_column") or (context.get("target_column") if isinstance(context, dict) else "")

        # Fallback to dataframe input
        if (X_calib is None or y_calib is None) and "dataframe" in inputs:
            df = inputs["dataframe"]
            if target_col and target_col in df.columns:
                X_calib = df.drop(columns=[target_col])
                y_calib = df[target_col]

        if (X_calib is None or y_calib is None) and isinstance(context, dict) and "dataframe" in context:
            df = context["dataframe"]
            if target_col and target_col in df.columns:
                X_calib = df.drop(columns=[target_col])
                y_calib = df[target_col]

        if X_calib is None or y_calib is None:
            raise ValueError("ModelCalibration expects feature data (X) and ground truth targets (y) for calibration.")

        # 3. Clean and prepare X_calib and y_calib
        if isinstance(X_calib, np.ndarray):
            X_calib = pd.DataFrame(X_calib, columns=[f"feature_{i}" for i in range(X_calib.shape[1])])
        elif isinstance(X_calib, pd.DataFrame):
            X_calib = X_calib.copy()
        else:
            X_calib = pd.DataFrame(X_calib)

        # Align expected features if model has feature_names_in_
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            X_calib = X_calib.reindex(columns=expected, fill_value=0)

        # Safe numeric conversion
        for col in X_calib.columns:
            if not pd.api.types.is_numeric_dtype(X_calib[col]):
                try:
                    X_calib[col] = pd.to_numeric(X_calib[col])
                except Exception:
                    X_calib[col] = X_calib[col].astype("category").cat.codes
        X_calib = X_calib.fillna(X_calib.median(numeric_only=True).fillna(0))

        y_calib_arr = np.array(y_calib)
        unique_classes = np.unique(y_calib_arr)
        if len(unique_classes) < 2:
            raise ValueError(f"Calibration data contains only 1 class ({unique_classes[0]}). At least 2 classes required.")

        # Label encode target if string or categorical
        le = None
        if isinstance(unique_classes[0], (str, bytes)):
            le = LabelEncoder()
            y_calib_arr = le.fit_transform(y_calib_arr)
            unique_classes = np.unique(y_calib_arr)

        N = len(y_calib_arr)
        method = config.get("method", "sigmoid")
        cv_cfg = str(config.get("cv_strategy", "prefit")).lower().strip()
        num_bins = int(config.get("num_bins", 10))

        _, class_counts = np.unique(y_calib_arr, return_counts=True)
        min_class_count = int(np.min(class_counts))
        if min_class_count < 2:
            raise ValueError(f"At least 2 samples per class are required for probability calibration, but the smallest class has only {min_class_count} sample(s).")

        estimator_to_fit = model
        cv_param = None
        if "3" in cv_cfg:
            cv_param = min(3, min_class_count)
        elif "5" in cv_cfg:
            cv_param = min(5, min_class_count)
        elif cv_cfg.isdigit():
            cv_param = min(int(cv_cfg), min_class_count)
        else:
            # Prefit / holdout calibration
            if FrozenEstimator is not None:
                estimator_to_fit = FrozenEstimator(model)
                if min_class_count < 5:
                    cv_param = min_class_count
                else:
                    cv_param = None
            else:
                cv_param = "prefit"

        warnings: List[str] = []
        if N < 30:
            warnings.append(
                f"[WARNING: SMALL SAMPLE] Calibration split contains only N={N} observations. "
                "Isotonic regression may overfit on small samples; Platt scaling (sigmoid) is strongly recommended."
            )
        if min_class_count < 5:
            warnings.append(
                f"[INFO: ADAPTIVE CV] Minimum class sample size ({min_class_count}) adjusted internal validation folds to {cv_param} to prevent fold collapse."
            )

        # 4. Measure Uncalibrated Baseline Performance
        uncalib_probs = None
        uncalib_brier = None
        uncalib_log_loss = None

        if hasattr(model, "predict_proba"):
            try:
                X_pred = X_calib.values if not hasattr(model, "feature_names_in_") else X_calib
                uncalib_probs = model.predict_proba(X_pred)
                if len(unique_classes) == 2 and uncalib_probs.shape[1] == 2:
                    uncalib_brier = float(round(brier_score_loss(y_calib_arr, uncalib_probs[:, 1]), 6))
                uncalib_log_loss = float(round(log_loss(y_calib_arr, uncalib_probs), 6))
            except Exception:
                pass

        # 5. Fit CalibratedClassifierCV
        calibrator = CalibratedClassifierCV(
            estimator=estimator_to_fit,
            method=method,
            cv=cv_param
        )
        fit_X = X_calib.values if not hasattr(model, "feature_names_in_") else X_calib
        calibrator.fit(fit_X, y_calib_arr)

        # 6. Measure Calibrated Performance
        calib_probs = calibrator.predict_proba(fit_X)
        calib_preds = calibrator.predict(fit_X)
        calib_acc = float(round(accuracy_score(y_calib_arr, calib_preds), 4))

        calib_brier = None
        if len(unique_classes) == 2 and calib_probs.shape[1] == 2:
            calib_brier = float(round(brier_score_loss(y_calib_arr, calib_probs[:, 1]), 6))
        calib_log_loss = float(round(log_loss(y_calib_arr, calib_probs), 6))

        # 7. Compute Reliability Diagram & Expected Calibration Error (ECE)
        prob_pos = calib_probs[:, 1] if calib_probs.shape[1] == 2 else np.max(calib_probs, axis=1)
        y_eval_binary = (y_calib_arr == unique_classes[1]).astype(int) if len(unique_classes) == 2 else (y_calib_arr == calib_preds).astype(int)

        # ECE calculation across uniform bins
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        reliability_diagram: List[Dict[str, Any]] = []

        for b in range(num_bins):
            b_min, b_max = bin_edges[b], bin_edges[b + 1]
            mask = (prob_pos >= b_min) & (prob_pos < b_max if b < num_bins - 1 else prob_pos <= b_max)
            count = int(np.sum(mask))
            if count > 0:
                acc_bin = float(np.mean(y_eval_binary[mask]))
                conf_bin = float(np.mean(prob_pos[mask]))
                diff = abs(acc_bin - conf_bin)
                ece += (count / N) * diff
                reliability_diagram.append({
                    "bin_index": b + 1,
                    "bin_range": f"[{b_min:.2f}, {b_max:.2f})",
                    "sample_count": count,
                    "mean_confidence": round(conf_bin, 4),
                    "empirical_accuracy": round(acc_bin, 4),
                    "calibration_gap": round(diff, 4)
                })

        ece = float(round(ece, 6))

        # Calculate Brier and Log Loss percentage improvement
        brier_lift_pct = None
        if uncalib_brier is not None and uncalib_brier > 0:
            brier_lift_pct = round(((uncalib_brier - calib_brier) / uncalib_brier) * 100.0, 2)

        log_loss_lift_pct = None
        if uncalib_log_loss is not None and uncalib_log_loss > 0:
            log_loss_lift_pct = round(((uncalib_log_loss - calib_log_loss) / uncalib_log_loss) * 100.0, 2)

        metrics = {
            "calibration_method": method,
            "cv_strategy": "prefit" if cv_cfg in ("prefit", "none", "") else str(cv_param),
            "sample_size": N,
            "num_classes": len(unique_classes),
            "accuracy": calib_acc,
            "uncalibrated_log_loss": uncalib_log_loss,
            "calibrated_log_loss": calib_log_loss,
            "log_loss_reduction_pct": log_loss_lift_pct,
            "uncalibrated_brier_score": uncalib_brier,
            "calibrated_brier_score": calib_brier,
            "brier_score_reduction_pct": brier_lift_pct,
            "expected_calibration_error_ece": ece,
            "reliability_diagram": reliability_diagram,
            "warnings": warnings
        }

        # 8. Assemble Governance Report
        report_lines = [
            "=== Model Probability Calibration Report ===",
            f"Base Model          : {type(model).__name__}",
            f"Calibrated Estimator: CalibratedClassifierCV ({method.upper()})",
            f"CV Setting          : {cv_cfg}",
            f"Sample Size (N)     : {N}",
            f"Classes             : {len(unique_classes)} {list(unique_classes)}",
            f"Accuracy            : {calib_acc:.4f}",
            f"ECE (Expected Error): {ece:.6f}"
        ]

        if uncalib_log_loss is not None:
            ll_status = f"Reduced by {log_loss_lift_pct}%" if (log_loss_lift_pct and log_loss_lift_pct > 0) else "Preserved"
            report_lines.append(f"Log Loss            : {uncalib_log_loss:.4f} -> {calib_log_loss:.4f} ({ll_status})")

        if uncalib_brier is not None and calib_brier is not None:
            br_status = f"Improved by {brier_lift_pct}%" if (brier_lift_pct and brier_lift_pct > 0) else "Maintained"
            report_lines.append(f"Brier Score         : {uncalib_brier:.4f} -> {calib_brier:.4f} ({br_status})")

        report_lines.append(f"\n--- Reliability Diagram ({num_bins} Probability Bins) ---")
        report_lines.append(f"{'Bin Range':<15} | {'Count':<6} | {'Confidence':<12} | {'Empirical Acc':<14} | {'Gap'}")
        report_lines.append("-" * 62)
        for row in reliability_diagram:
            report_lines.append(
                f"{row['bin_range']:<15} | {row['sample_count']:<6} | {row['mean_confidence']:<12.4f} | {row['empirical_accuracy']:<14.4f} | {row['calibration_gap']:.4f}"
            )

        if warnings:
            report_lines.append("\n--- Diagnostics & Guardrails ---")
            for w in warnings:
                report_lines.append(f"- {w}")

        return {
            "model": calibrator,
            "calibrated_model": calibrator,
            "metrics": metrics,
            "report": "\n".join(report_lines)
        }
