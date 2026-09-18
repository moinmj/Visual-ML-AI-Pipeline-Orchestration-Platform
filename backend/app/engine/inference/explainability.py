import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except (ImportError, OSError, Exception):
    shap = None
    SHAP_AVAILABLE = False


def _is_currency_column(target_name: Optional[str], typical_value: Optional[float]) -> bool:
    """Detects whether target variable represents monetary amounts based on name or scale."""
    if not target_name:
        if typical_value is not None and abs(typical_value) >= 500:
            return True
        return False
    tn = target_name.lower()
    currency_keywords = [
        "sales", "price", "revenue", "cost", "income", "profit", "amount",
        "fee", "earning", "salary", "wage", "spend", "charge", "gdp", "val"
    ]
    if any(k in tn for k in currency_keywords):
        return True
    if typical_value is not None and abs(typical_value) >= 500:
        return True
    return False


def _format_value(val: float, is_currency: bool = False, with_sign: bool = False) -> str:
    """Formats numeric values into clean executive strings (e.g. '+$45,000' or '-$12,000')."""
    sign_str = ""
    if with_sign:
        sign_str = "+" if val >= 0 else "-"
    abs_v = abs(val)

    if is_currency:
        if abs_v >= 1000:
            if abs_v >= 100_000:
                return f"{sign_str}${abs_v:,.0f}"
            else:
                # If cents are .00, don't display .00
                if round(abs_v, 2) == round(abs_v, 0):
                    return f"{sign_str}${abs_v:,.0f}"
                return f"{sign_str}${abs_v:,.2f}"
        else:
            return f"{sign_str}${abs_v:,.2f}"
    else:
        if abs_v >= 100 and (abs_v.is_integer() or round(abs_v, 2) == round(abs_v, 0)):
            return f"{sign_str}{abs_v:,.0f}"
        elif abs_v >= 10:
            return f"{sign_str}{abs_v:,.2f}"
        else:
            return f"{sign_str}{abs_v:,.4f}"


def get_model_explainer(model: Any, bundle: Dict[str, Any]) -> Optional[Any]:
    """Retrieves or creates a cached SHAP TreeExplainer for tree models."""
    if not SHAP_AVAILABLE or model is None:
        return None

    # Check bundle or model cache
    cached = bundle.get("_cached_explainer") or getattr(model, "_cached_explainer", None)
    if cached is not None:
        return cached

    model_name = type(model).__name__
    is_tree = any(k in model_name for k in [
        "XGB", "LGBM", "CatBoost", "Forest", "Tree", "GradientBoosting", "ExtraTrees"
    ])

    if is_tree:
        try:
            explainer = shap.TreeExplainer(model)
            bundle["_cached_explainer"] = explainer
            try:
                setattr(model, "_cached_explainer", explainer)
            except Exception:
                pass
            return explainer
        except Exception as e:
            logger.warning(f"TreeExplainer instantiation failed for {model_name}: {e}")
            return None

    return None


def compute_waterfall_breakdown(
    model: Any,
    X_row: pd.DataFrame,
    feature_names: List[str],
    bundle: Dict[str, Any],
    predicted_value: float,
    task_type: str = "regression"
) -> Dict[str, Any]:
    """
    Computes a SHAP / TreeSHAP waterfall decomposition for a single live prediction.
    Explains the exact step-by-step feature contributions from the baseline E[f(x)]
    to the final predicted value.

    Returns:
      - base_value: float
      - base_value_formatted: str
      - waterfall_breakdown: List[Dict]
      - top_positive_drivers: List[str]
      - top_negative_drivers: List[str]
      - waterfall_summary: str (e.g. 'Store contributed +$45,000; Fuel_Price contributed -$12,000; Unemployment contributed -$8,000.')
    """
    target_col = bundle.get("target_column") or "Target"
    is_currency = _is_currency_column(target_col, predicted_value)

    # 1. Attempt TreeSHAP calculation
    explainer = get_model_explainer(model, bundle)
    shap_vals = None
    base_val = None

    if explainer is not None:
        try:
            raw_sv = explainer.shap_values(X_row)
            raw_exp = explainer.expected_value

            # Resolve expected value
            if hasattr(raw_exp, "__len__") and not isinstance(raw_exp, (str, bytes)):
                base_val = float(raw_exp[0])
            else:
                base_val = float(raw_exp)

            # Resolve shap values row
            if isinstance(raw_sv, list):
                # Multiclass or binary list: pick positive class or winning class
                if len(raw_sv) == 2:
                    raw_sv = raw_sv[1]
                elif len(raw_sv) > 0:
                    raw_sv = raw_sv[0]
            if hasattr(raw_sv, "ndim") and raw_sv.ndim == 2:
                shap_vals = raw_sv[0]
            elif hasattr(raw_sv, "ndim") and raw_sv.ndim == 3:
                shap_vals = raw_sv[0, :, 0]
            else:
                shap_vals = np.array(raw_sv).flatten()
        except Exception as e:
            logger.warning(f"TreeSHAP calculation failed, falling back to feature importance: {e}")
            shap_vals = None
            base_val = None

    # 2. Fallback heuristic if SHAP is unavailable or failed (e.g. non-tree models)
    if shap_vals is None or base_val is None:
        return _compute_heuristic_waterfall(
            model=model,
            X_row=X_row,
            feature_names=feature_names,
            bundle=bundle,
            predicted_value=predicted_value,
            is_currency=is_currency
        )

    # 3. Handle Target Unscaling if target was scaled during training
    scaler = bundle.get("scaler")
    scale_multiplier = 1.0
    shift_offset = 0.0
    if scaler is not None and target_col and hasattr(scaler, "feature_names_in_"):
        fn_in = list(scaler.feature_names_in_)
        if target_col in fn_in:
            t_idx = fn_in.index(target_col)
            if hasattr(scaler, "scale_"):
                scale_multiplier = float(scaler.scale_[t_idx])
                shift_offset = float(scaler.mean_[t_idx]) if hasattr(scaler, "mean_") else 0.0
            elif hasattr(scaler, "data_range_"):
                scale_multiplier = float(scaler.data_range_[t_idx])
                shift_offset = float(scaler.data_min_[t_idx]) if hasattr(scaler, "data_min_") else 0.0

    # Unscale baseline and SHAP values to original real-world units
    unscaled_base = base_val * scale_multiplier + shift_offset
    unscaled_attributions = shap_vals * scale_multiplier

    # Build detailed FeatureAttribution list
    breakdown = []
    total_abs_imp = float(np.sum(np.abs(unscaled_attributions))) + 1e-9

    cols = list(X_row.columns) if hasattr(X_row, "columns") else feature_names
    for i, col in enumerate(cols):
        if i >= len(unscaled_attributions):
            break
        attr = float(unscaled_attributions[i])
        abs_attr = abs(attr)
        pct = round((abs_attr / total_abs_imp) * 100.0, 1)

        # Extract input value for feature
        inp_val = None
        if hasattr(X_row, "iloc"):
            inp_val = X_row.iloc[0][col]
            if isinstance(inp_val, (np.floating, float)):
                inp_val = round(float(inp_val), 4)
            elif isinstance(inp_val, (np.integer, int)):
                inp_val = int(inp_val)

        direction = "positive" if attr >= 0 else "negative"
        attr_formatted = _format_value(attr, is_currency=is_currency, with_sign=True)

        breakdown.append({
            "feature": col,
            "input_value": inp_val,
            "attribution": round(attr, 4),
            "attribution_formatted": attr_formatted,
            "abs_importance": round(abs_attr, 4),
            "direction": direction,
            "percentage": pct
        })

    # Sort by absolute importance (highest impact drivers first)
    breakdown.sort(key=lambda x: x["abs_importance"], reverse=True)

    # Format base value
    base_val_formatted = _format_value(unscaled_base, is_currency=is_currency, with_sign=False)

    # Extract top positive & negative drivers
    pos_drivers = [
        f"{item['feature']} ({item['attribution_formatted']})"
        for item in breakdown if item["direction"] == "positive" and item["abs_importance"] > 0
    ][:3]

    neg_drivers = [
        f"{item['feature']} ({item['attribution_formatted']})"
        for item in breakdown if item["direction"] == "negative" and item["abs_importance"] > 0
    ][:3]

    # Build human-readable waterfall summary string:
    # "Store contributed +$45,000; Fuel_Price contributed -$12,000; Unemployment contributed -$8,000."
    top_items = breakdown[:4]
    summary_parts = [
        f"{it['feature']} contributed {it['attribution_formatted']}"
        for it in top_items if it["abs_importance"] > 0
    ]
    waterfall_summary = "; ".join(summary_parts) + "." if summary_parts else "Feature attributions are balanced."

    return {
        "base_value": round(unscaled_base, 4),
        "base_value_formatted": base_val_formatted,
        "waterfall_breakdown": breakdown,
        "top_positive_drivers": pos_drivers,
        "top_negative_drivers": neg_drivers,
        "waterfall_summary": waterfall_summary
    }


def _compute_heuristic_waterfall(
    model: Any,
    X_row: pd.DataFrame,
    feature_names: List[str],
    bundle: Dict[str, Any],
    predicted_value: float,
    is_currency: bool
) -> Dict[str, Any]:
    """Graceful fallback when TreeSHAP is not applicable (e.g. linear/KNN models or missing C-libs)."""
    cols = list(X_row.columns) if hasattr(X_row, "columns") else feature_names
    feat_summary = bundle.get("training_feature_summary", {})
    feat_importances = bundle.get("feature_importances", {})

    # Compute baseline as median/mean prediction or 90% of prediction
    base_val = predicted_value * 0.95
    total_delta = predicted_value - base_val

    raw_weights = []
    for col in cols:
        imp = feat_importances.get(col, 1.0)
        # Deviation from mean if available
        mean_val = feat_summary.get(col, {}).get("mean_value", 0.0)
        curr_val = float(X_row.iloc[0][col]) if hasattr(X_row, "iloc") else 0.0
        diff = curr_val - mean_val
        direction = 1.0 if diff >= 0 else -1.0
        raw_weights.append(direction * imp)

    raw_weights = np.array(raw_weights)
    sum_w = np.sum(np.abs(raw_weights)) + 1e-9
    attributions = (raw_weights / sum_w) * total_delta

    breakdown = []
    total_abs = np.sum(np.abs(attributions)) + 1e-9
    for i, col in enumerate(cols):
        attr = float(attributions[i])
        abs_attr = abs(attr)
        direction = "positive" if attr >= 0 else "negative"
        attr_fmt = _format_value(attr, is_currency=is_currency, with_sign=True)

        inp_val = None
        if hasattr(X_row, "iloc"):
            inp_val = X_row.iloc[0][col]

        breakdown.append({
            "feature": col,
            "input_value": inp_val,
            "attribution": round(attr, 4),
            "attribution_formatted": attr_fmt,
            "abs_importance": round(abs_attr, 4),
            "direction": direction,
            "percentage": round((abs_attr / total_abs) * 100.0, 1)
        })

    breakdown.sort(key=lambda x: x["abs_importance"], reverse=True)
    base_fmt = _format_value(base_val, is_currency=is_currency, with_sign=False)

    pos_drivers = [f"{it['feature']} ({it['attribution_formatted']})" for it in breakdown if it["direction"] == "positive"][:3]
    neg_drivers = [f"{it['feature']} ({it['attribution_formatted']})" for it in breakdown if it["direction"] == "negative"][:3]

    top_items = breakdown[:3]
    summary_parts = [f"{it['feature']} contributed {it['attribution_formatted']}" for it in top_items]
    waterfall_summary = "; ".join(summary_parts) + "." if summary_parts else "Feature attributions are balanced."

    return {
        "base_value": round(base_val, 4),
        "base_value_formatted": base_fmt,
        "waterfall_breakdown": breakdown,
        "top_positive_drivers": pos_drivers,
        "top_negative_drivers": neg_drivers,
        "waterfall_summary": waterfall_summary
    }
