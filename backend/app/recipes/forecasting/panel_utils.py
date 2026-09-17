import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

PANEL_CONFIG_PROPERTIES = {
    "group_by_column": {
        "type": "string",
        "title": "Entity / Group Column",
        "description": "Optional column identifying distinct entities (e.g. 'Store', 'Region', 'Product') in panel datasets."
    },
    "panel_strategy": {
        "type": "string",
        "title": "Panel Data Handling Strategy",
        "enum": ["auto", "aggregate_sum", "aggregate_mean", "filter_entity"],
        "default": "auto",
        "description": "Strategy for datasets where multiple rows share the same date. 'auto' / 'aggregate_sum' sums across entities, 'aggregate_mean' averages, and 'filter_entity' isolates a single entity."
    },
    "entity_value": {
        "type": "string",
        "title": "Entity Filter Value",
        "description": "Specific entity identifier to forecast (e.g. '1' for Store 1) when 'filter_entity' is selected."
    }
}


def prepare_univariate_panel_series(
    df: pd.DataFrame,
    valid_ds: pd.Series,
    target_col: str,
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transforms arbitrary tabular or multi-entity panel data into a single,
    unambiguous univariate time series (ds, y) suitable for Prophet / ARIMA.

    Detects duplicate timestamps (multiple entities per date) and resolves them via:
    1. Entity filtering (e.g., isolating Store == 1)
    2. Network aggregation (e.g., sum or mean of Weekly_Sales across all stores per date)
    """
    _log = log or logger

    group_col = config.get("group_by_column")
    if group_col and str(group_col).strip() in df.columns:
        group_col = str(group_col).strip()
    else:
        # Auto-detect potential entity identifier columns if duplicates occur
        group_col = None
        cand_names = ["store", "store_id", "region", "product", "product_id", "item", "item_id", "entity", "account", "sensor", "city", "country"]
        for c in df.columns:
            if c != target_col and c.lower() in cand_names:
                group_col = c
                break

    base_dict = {
        "ds": valid_ds,
        "y": pd.to_numeric(df[target_col], errors="coerce")
    }
    if group_col and group_col in df.columns:
        base_dict["_entity"] = df[group_col].astype(str)

    raw_df = pd.DataFrame(base_dict).dropna(subset=["ds", "y"]).sort_values(by="ds").reset_index(drop=True)

    dup_count = int(raw_df["ds"].duplicated().sum())
    panel_detected = dup_count > 0
    strategy = str(config.get("panel_strategy", "auto")).strip().lower()
    entity_val = config.get("entity_value")

    if panel_detected:
        unique_dates = raw_df["ds"].nunique()
        total_rows = len(raw_df)
        avg_entities = round(total_rows / max(1, unique_dates), 1)

        # Strategy 1: Filter to specific single entity
        if (strategy == "filter_entity" or (entity_val is not None and str(entity_val).strip())) and "_entity" in raw_df.columns:
            val_str = str(entity_val).strip() if entity_val is not None else ""
            if not val_str:
                # Default to top entity with most historical records
                val_str = str(raw_df["_entity"].value_counts().index[0])
            
            filtered = raw_df[raw_df["_entity"] == val_str].drop(columns=["_entity"]).reset_index(drop=True)
            if len(filtered) >= 5:
                # Ensure no remaining duplicates within that single entity
                if filtered["ds"].duplicated().any():
                    filtered = filtered.groupby("ds", as_index=False)["y"].mean()

                summary_info = (
                    f"Multi-entity panel data detected (~{avg_entities} entities per timestamp across {unique_dates} unique dates). "
                    f"Filtered series exclusively to entity '{group_col} = {val_str}' ({len(filtered)} observations)."
                )
                _log.info(summary_info)
                return filtered, {
                    "panel_data_detected": True,
                    "duplicate_timestamps": dup_count,
                    "panel_strategy_applied": "filter_entity",
                    "group_by_column": group_col,
                    "entity_value": val_str,
                    "panel_summary_info": summary_info
                }
            else:
                _log.warning(
                    f"Entity filter '{val_str}' for column '{group_col}' yielded fewer than 5 rows ({len(filtered)}). "
                    "Falling back to network aggregation."
                )

        # Strategy 2: Aggregate across all entities per timestamp
        agg_func = "mean" if strategy in ["aggregate_mean", "mean"] else "sum"
        ts_df = raw_df.groupby("ds", as_index=False)["y"].agg(agg_func)
        applied_strategy = f"aggregate_{agg_func}"

        summary_info = (
            f"Multi-entity panel data detected ({total_rows} records across {unique_dates} unique dates, "
            f"~{avg_entities} entities per date). "
            f"Auto-aggregated metric across all entities using '{agg_func}' to ensure valid univariate time-series modeling."
        )
        _log.info(summary_info)
        return ts_df, {
            "panel_data_detected": True,
            "duplicate_timestamps": dup_count,
            "panel_strategy_applied": applied_strategy,
            "group_by_column": group_col,
            "entity_value": None,
            "panel_summary_info": summary_info
        }

    else:
        # Clean 1-point-per-date series
        if "_entity" in raw_df.columns:
            raw_df = raw_df.drop(columns=["_entity"])
        return raw_df, {
            "panel_data_detected": False,
            "duplicate_timestamps": 0,
            "panel_strategy_applied": "single_series",
            "group_by_column": group_col,
            "entity_value": None,
            "panel_summary_info": "Dataset contains a single clean time-series (1 observation per timestamp)."
        }
