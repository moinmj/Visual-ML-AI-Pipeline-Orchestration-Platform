import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import (
    safe_prepare_training_data,
    extract_train_test_data,
    pass_through_metadata
)


class NeuralNetworkTrainerRecipe(BaseRecipe):
    """
    Neural Network Trainer Recipe (Deep Multi-Layer Perceptron / MLP).
    Trains deep feed-forward artificial neural networks with backpropagation,
    Adam/SGD optimizers, automated input standardization pipeline, early stopping,
    and convergence loss trajectory tracking.
    """

    recipe_id = "neural_network_trainer"
    name = "Neural Network Trainer (Feed-Forward / MLP)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Deep Learning / AutoML"
    description = (
        "Trains a Multi-Layer Perceptron (MLP) deep feed-forward neural network for tabular classification "
        "or regression with Adam optimizer and loss curve tracking."
    )
    input_types = ["train_data"]
    output_types = ["model", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="train_data",
            label="Train/Test Split Data",
            type="train_data",
            required=True,
            max_connections=1,
            description="Training dataset containing X_train, y_train and optional X_test, y_test."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Trained Neural Network Pipeline",
            type="model",
            description="Trained Pipeline containing StandardScaler and MLPClassifier/MLPRegressor."
        ),
        RecipePort(
            id="metrics",
            label="Training Metrics",
            type="metrics",
            description="Epochs trained, loss convergence, accuracy/F1 or R2/RMSE, and loss trajectory."
        ),
        RecipePort(
            id="report",
            label="Executive Summary Report",
            type="report",
            description="Markdown report containing layer architecture, loss curve progression, and holdout performance."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "title": "ML Task Type",
                    "input_type": "dropdown",
                    "enum": ["auto", "classification", "regression"],
                    "default": "auto",
                    "description": "Auto-detects classification vs. regression based on target variable."
                },
                "architecture_preset": {
                    "type": "string",
                    "title": "Network Architecture Preset",
                    "input_type": "dropdown",
                    "enum": [
                        "two_layer_medium",
                        "single_layer",
                        "three_layer_deep",
                        "wide_shallow",
                        "custom"
                    ],
                    "default": "two_layer_medium",
                    "description": "Layer dimensions (e.g., 'two_layer_medium' uses (64, 32) hidden nodes)."
                },
                "hidden_layers": {
                    "type": "string",
                    "title": "Custom Hidden Layer Sizes",
                    "input_type": "text",
                    "default": "64, 32",
                    "description": "Comma-separated integers specifying neurons per layer (used if custom preset is selected)."
                },
                "activation": {
                    "type": "string",
                    "title": "Activation Function",
                    "input_type": "dropdown",
                    "enum": ["relu", "tanh", "logistic"],
                    "default": "relu",
                    "description": "Non-linear activation function between layers (ReLU is standard for deep learning)."
                },
                "optimizer": {
                    "type": "string",
                    "title": "Optimization Algorithm",
                    "input_type": "dropdown",
                    "enum": ["adam", "sgd", "lbfgs"],
                    "default": "adam",
                    "description": "Gradient descent solver. Adam handles large datasets with adaptive momentum."
                },
                "learning_rate_init": {
                    "type": "number",
                    "title": "Initial Learning Rate",
                    "input_type": "number",
                    "default": 0.001,
                    "description": "Initial step size for weight parameter updates."
                },
                "max_iter": {
                    "type": "integer",
                    "title": "Max Epochs (Iterations)",
                    "input_type": "slider",
                    "default": 200,
                    "minimum": 50,
                    "maximum": 1000
                },
                "alpha": {
                    "type": "number",
                    "title": "L2 Weight Regularization (Alpha)",
                    "input_type": "number",
                    "default": 0.0001,
                    "description": "L2 penalty weight decay parameter to prevent overfitting."
                },
                "early_stopping": {
                    "type": "boolean",
                    "title": "Early Stopping Guard",
                    "input_type": "toggle",
                    "default": True,
                    "description": "Terminates training early when validation loss plateaus to guard against memorization."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "input_type": "number",
                    "default": 42
                }
            },
            "required": []
        }

    def _infer_task_type(self, y: Any) -> str:
        y_arr = np.array(y)
        if y_arr.dtype.kind in ("U", "S", "O", "b"):
            return "classification"
        unique_vals = np.unique(y_arr[~pd.isna(y_arr)])
        if len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int)):
            return "classification"
        return "regression"

    def _parse_hidden_layers(self, preset: str, custom_str: str) -> Tuple[int, ...]:
        preset_map = {
            "single_layer": (64,),
            "two_layer_medium": (64, 32),
            "three_layer_deep": (128, 64, 32),
            "wide_shallow": (128,)
        }
        if preset in preset_map:
            return preset_map[preset]

        try:
            parts = [int(p.strip()) for p in custom_str.split(",") if p.strip().isdigit()]
            if parts and all(p > 0 for p in parts):
                return tuple(parts)
        except Exception:
            pass

        return (64, 32)

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Extract Training Data
        X_train, y_train, X_test, y_test = extract_train_test_data(inputs)
        if (X_train is None or y_train is None) and isinstance(context, dict):
            X_train, y_train, X_test, y_test = extract_train_test_data(context)

        # Fallback to direct dataframe input
        if (X_train is None or y_train is None) and "dataframe" in inputs:
            df = inputs["dataframe"]
            target_col = config.get("target_column") or (context.get("target_column") if isinstance(context, dict) else "")
            if target_col and target_col in df.columns:
                X_train = df.drop(columns=[target_col])
                y_train = df[target_col]

        if X_train is None or y_train is None:
            raise ValueError(
                "NeuralNetworkTrainer expects 'X_train' and 'y_train' in inputs. "
                "Please connect a Train/Test Split node before this trainer."
            )

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(X_train.shape[1])])
        elif not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=X_train.columns)
        elif X_test is not None and not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        # Clean non-numeric features safely
        X_train, X_test = safe_prepare_training_data(X_train, X_test)
        feature_names = list(X_train.columns)

        # 2. Determine Task Type
        configured_task = str(config.get("task_type", "auto")).lower()
        if configured_task in ("classification", "regression"):
            task_type = configured_task
        else:
            task_type = self._infer_task_type(y_train)

        # 3. Target Variable Guardrails & Encoding
        y_train_arr = np.array(y_train)
        target_encoder = None
        target_classes: Optional[List[str]] = None

        if task_type == "classification":
            unique_classes, counts = np.unique(y_train_arr, return_counts=True)
            if len(unique_classes) < 2:
                raise ValueError(
                    f"Classification target variable contains only 1 class ({unique_classes[0]}). "
                    "At least 2 classes are required to train a neural network classifier."
                )

            # String or categorical label encoding
            if isinstance(unique_classes[0], (str, bytes)):
                target_encoder = LabelEncoder()
                y_train_arr = target_encoder.fit_transform(y_train_arr)
                target_classes = [str(c) for c in target_encoder.classes_]
            else:
                target_classes = [str(c) for c in unique_classes]

            if y_test is not None and target_encoder is not None:
                y_test_clean = target_encoder.transform(np.array(y_test))
            elif y_test is not None:
                y_test_clean = np.array(y_test)
            else:
                y_test_clean = None
        else:
            try:
                y_train_arr = pd.to_numeric(pd.Series(y_train_arr)).to_numpy()
            except Exception:
                raise ValueError("Regression target variable must be numeric.")
            if y_test is not None:
                y_test_clean = pd.to_numeric(pd.Series(np.array(y_test))).to_numpy()
            else:
                y_test_clean = None

        # 4. Neural Network Hyperparameter Configuration
        preset = config.get("architecture_preset", "two_layer_medium")
        custom_str = str(config.get("hidden_layers", "64, 32"))
        hidden_layer_sizes = self._parse_hidden_layers(preset, custom_str)

        activation = str(config.get("activation", "relu"))
        solver = str(config.get("optimizer", "adam"))
        lr_init = float(config.get("learning_rate_init", 0.001))
        max_iter = int(config.get("max_iter", 200))
        alpha = float(config.get("alpha", 0.0001))
        early_stopping = bool(config.get("early_stopping", True))
        random_state = int(config.get("random_state", 42))

        warnings: List[str] = []
        n_samples = len(y_train_arr)

        # For very small sample sizes, disable early stopping validation split to avoid fold collapse
        effective_early_stopping = early_stopping
        if n_samples < 30 and early_stopping:
            effective_early_stopping = False
            warnings.append(
                f"[INFO: ADAPTIVE REGULARIZATION] Sample size (N={n_samples}) is small. Early stopping was "
                "temporarily disabled to ensure all observations contribute to weight updates."
            )

        # 5. Build Standardized Neural Network Pipeline
        if task_type == "classification":
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                learning_rate_init=lr_init,
                max_iter=max_iter,
                alpha=alpha,
                early_stopping=effective_early_stopping,
                random_state=random_state
            )
        else:
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                learning_rate_init=lr_init,
                max_iter=max_iter,
                alpha=alpha,
                early_stopping=effective_early_stopping,
                random_state=random_state
            )

        # Pipeline standardizes inputs automatically
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", mlp)
        ])

        # 6. Train Neural Network
        pipeline.fit(X_train, y_train_arr)
        fitted_mlp = pipeline.named_steps["mlp"]

        # 7. Extract Loss Trajectory & Performance Metrics
        epochs_trained = getattr(fitted_mlp, "n_iter_", max_iter)
        raw_loss = getattr(fitted_mlp, "best_loss_", None)
        if raw_loss is None:
            raw_loss = getattr(fitted_mlp, "loss_", 0.0)
        best_loss = float(round(raw_loss, 6)) if raw_loss is not None else 0.0
        loss_curve = getattr(fitted_mlp, "loss_curve_", [])
        loss_history = [float(round(val, 6)) for val in loss_curve]

        # In-sample evaluation
        train_preds = pipeline.predict(X_train)
        if task_type == "classification":
            train_acc = float(round(accuracy_score(y_train_arr, train_preds), 4))
            train_f1 = float(round(f1_score(y_train_arr, train_preds, average="macro", zero_division=0), 4))
            performance_metrics = {
                "train_accuracy": train_acc,
                "train_f1_macro": train_f1
            }
        else:
            train_r2 = float(round(r2_score(y_train_arr, train_preds), 4))
            train_rmse = float(round(math.sqrt(mean_squared_error(y_train_arr, train_preds)), 4))
            train_mae = float(round(mean_absolute_error(y_train_arr, train_preds), 4))
            performance_metrics = {
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "train_mae": train_mae
            }

        # Holdout test set evaluation (if provided)
        test_metrics: Dict[str, Any] = {}
        if X_test is not None and y_test_clean is not None:
            test_preds = pipeline.predict(X_test)
            if task_type == "classification":
                test_metrics["test_accuracy"] = float(round(accuracy_score(y_test_clean, test_preds), 4))
                test_metrics["test_f1_macro"] = float(round(f1_score(y_test_clean, test_preds, average="macro", zero_division=0), 4))
            else:
                test_metrics["test_r2"] = float(round(r2_score(y_test_clean, test_preds), 4))
                test_metrics["test_rmse"] = float(round(math.sqrt(mean_squared_error(y_test_clean, test_preds)), 4))
                test_metrics["test_mae"] = float(round(mean_absolute_error(y_test_clean, test_preds), 4))

        metrics = {
            "task_type": task_type,
            "architecture": list(hidden_layer_sizes),
            "total_layers": len(hidden_layer_sizes) + 2,  # input + hidden + output
            "activation": activation,
            "optimizer": solver,
            "epochs_trained": epochs_trained,
            "best_loss": best_loss,
            "early_stopping_triggered": (epochs_trained < max_iter) if effective_early_stopping else False,
            "train_metrics": performance_metrics,
            "test_metrics": test_metrics,
            "loss_curve_sample": loss_history[:10] + loss_history[-5:] if len(loss_history) > 15 else loss_history,
            "warnings": warnings
        }

        # 8. Format Executive Report
        report_lines = [
            f"# === Deep Neural Network (MLP) Training Report ({task_type.capitalize()}) ===",
            f"- **Network Architecture**: Input({len(feature_names)}) -> {' -> '.join(str(n) for n in hidden_layer_sizes)} -> Output",
            f"- **Activation Function**: {activation.upper()}",
            f"- **Optimizer**: {solver.upper()} (Initial LR: {lr_init})",
            f"- **Convergence**: Stopped at Epoch **{epochs_trained} / {max_iter}** (Final Loss: **{best_loss:.6f}**)",
            "",
            "## Model Accuracy & Generalization:"
        ]

        if task_type == "classification":
            report_lines.append(f"- **Train Accuracy**: **{train_acc * 100:.2f}%** | **Macro F1**: **{train_f1:.4f}**")
            if test_metrics:
                report_lines.append(f"- **Holdout Test Accuracy**: **{test_metrics['test_accuracy'] * 100:.2f}%** | **Test Macro F1**: **{test_metrics['test_f1_macro']:.4f}**")
        else:
            report_lines.append(f"- **Train R² Score**: **{train_r2:.4f}** | **Train RMSE**: **{train_rmse:.4f}**")
            if test_metrics:
                report_lines.append(f"- **Holdout Test R² Score**: **{test_metrics['test_r2']:.4f}** | **Test RMSE**: **{test_metrics['test_rmse']:.4f}**")

        if loss_history:
            report_lines.append("")
            report_lines.append("## Training Loss Progression:")
            report_lines.append(f"- Initial Loss (Epoch 1): `{loss_history[0]:.6f}`")
            report_lines.append(f"- Mid-point Loss: `{loss_history[len(loss_history)//2]:.6f}`")
            report_lines.append(f"- Converged Loss: `{loss_history[-1]:.6f}`")

        if warnings:
            report_lines.append("")
            report_lines.append("## Guardrail Alerts:")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "model": pipeline,
            "pipeline": pipeline,
            "task_type": task_type,
            "metrics": metrics,
            "report": report_markdown,
            "architecture": list(hidden_layer_sizes),
            "feature_names": feature_names
        }

        if target_classes:
            output["target_classes"] = target_classes
        if target_encoder:
            output["target_encoder"] = target_encoder

        if X_test is not None:
            output["X_test"] = X_test
        if y_test is not None:
            output["y_test"] = y_test

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        task_type = config.get("task_type", "classification")
        layers = config.get("hidden_layers", "64, 32")
        act = config.get("activation", "relu")
        cls_name = "MLPClassifier" if task_type == "classification" else "MLPRegressor"
        return (
            f"from sklearn.neural_network import {cls_name}\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n\n"
            f"mlp = {cls_name}(hidden_layer_sizes=({layers}), activation='{act}', max_iter=200, random_state=42)\n"
            "pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', mlp)])\n"
            "pipeline.fit(X_train, y_train)"
        )
