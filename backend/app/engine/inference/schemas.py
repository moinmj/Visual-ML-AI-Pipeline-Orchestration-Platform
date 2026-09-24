from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union


class PredictionRequest(BaseModel):
    """
    Standard request payload for live model predictions.
    Supports single records, batch records, time-series forecast specifications,
    future period projections, or natural language AI queries.
    """
    inputs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None,
        description="Single feature dictionary or list of feature dictionaries for tabular ML / Anomaly scoring."
    )
    forecast_horizon: Optional[int] = Field(
        default=None,
        description="Number of future intervals to forecast for time-series models."
    )
    future_periods: Optional[int] = Field(
        default=None,
        description="Number of future intervals or steps to project forward for tabular models."
    )
    target_year: Optional[int] = Field(
        default=None,
        description="Specific target future year to project towards (e.g., 2025)."
    )
    future_feature_overrides: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Explicit 'what-if' feature values for specific future periods, keyed by year as a string, e.g. {\"2014\": {\"CPI\": 210.0}, \"2015\": {\"CPI\": 215.0}}. For a given year+feature, this overrides both the frozen base value and the auto-drifted trend value. Years/features not mentioned continue to use normal historical drift."
    )
    natural_language_query: Optional[str] = Field(
        default=None,
        description="Natural language prompt for AI-driven prediction (e.g. 'Predict for 2025 with high humidity')."
    )
    requested_metric: Optional[str] = Field(
        default=None,
        description="Requested aggregate metric (e.g. 'average', 'max', 'min', 'end')."
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Optional start date for custom forecasting range (YYYY-MM-DD)."
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Optional end date for custom forecasting range (YYYY-MM-DD)."
    )
    freq: Optional[str] = Field(
        default=None,
        description="Frequency code for forecasting (e.g., 'D', 'W', 'M', 'H', 'Y')."
    )
    text_input: Optional[str] = Field(
        default=None,
        description="Raw string input for NLP text classification pipelines."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "inputs": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            }
        }
    }


class FeatureAttribution(BaseModel):
    """
    Individual feature contribution in a SHAP / TreeSHAP waterfall decomposition.
    """
    feature: str = Field(description="Name of the input feature")
    input_value: Optional[Any] = Field(default=None, description="Input value used in prediction")
    attribution: float = Field(description="Signed SHAP value contribution (+ or -)")
    attribution_formatted: str = Field(description="Human-formatted contribution string (e.g. '+$45,000' or '-$12,000')")
    abs_importance: float = Field(description="Absolute magnitude of contribution")
    direction: str = Field(description="'positive' (boosts target) or 'negative' (reduces target)")
    percentage: Optional[float] = Field(default=None, description="Percentage share of total attribution")


class PredictionResponse(BaseModel):
    """
    Standardized response payload for model predictions across all task families.
    """
    status: str = Field(default="SUCCESS", description="SUCCESS or FAILED")
    task_type: str = Field(description="classification, regression, time_series_forecasting, or anomaly_detection")
    execution_id: str = Field(description="Pipeline execution ID that trained this model")
    
    # Classification & Regression Outputs
    target_column: Optional[str] = Field(default=None, description="Name of the predicted target variable")
    prediction: Optional[Any] = Field(default=None, description="Decoded class label or continuous numeric target")
    prediction_label: Optional[str] = Field(default=None, description="Descriptive label for the prediction (e.g. 'Forecasted Average: Tlog')")
    prediction_raw: Optional[Any] = Field(default=None, description="Raw model output before class decoding")
    confidence: Optional[float] = Field(default=None, description="Confidence score percentage (0-100%) for classification")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Per-class probability distribution")
    
    # SHAP / TreeSHAP Waterfall Interpretability Outputs
    base_value: Optional[float] = Field(default=None, description="SHAP baseline / expected value E[f(x)] before feature contributions")
    base_value_formatted: Optional[str] = Field(default=None, description="Human-formatted baseline value string (e.g. '$915,554.58')")
    waterfall_breakdown: Optional[List[FeatureAttribution]] = Field(default=None, description="Ordered waterfall list of SHAP contributions from baseline to final prediction")
    top_positive_drivers: Optional[List[str]] = Field(default=None, description="Summary strings of top positive drivers (e.g. ['Store (+45,000)'])")
    top_negative_drivers: Optional[List[str]] = Field(default=None, description="Summary strings of top negative drivers (e.g. ['Fuel_Price (-12,000)'])")
    waterfall_summary: Optional[str] = Field(default=None, description="Plain English summary of waterfall breakdown, e.g. 'Store contributed +$45,000; Fuel_Price contributed -$12,000; Unemployment contributed -$8,000.'")

    # AI Natural Language Query Outputs
    ai_explanation: Optional[str] = Field(default=None, description="Natural language conversational explanation synthesized by LLM")
    inferred_inputs: Optional[Dict[str, Any]] = Field(default=None, description="Feature inputs extracted/inferred by AI from prompt")
    unrecognized_features: Optional[List[str]] = Field(default=None, description="Features mentioned in query prompt that do not exist in the dataset schema")

    # Anomaly Detection Outputs
    is_anomaly: Optional[int] = Field(default=None, description="1 if anomalous, 0 if normal")
    anomaly_score: Optional[float] = Field(default=None, description="Continuous outlier score (0.0 to 1.0)")
    verdict: Optional[str] = Field(default=None, description="Human-friendly risk verdict")
    risk_level: Optional[str] = Field(default=None, description="LOW, MEDIUM, HIGH, or CRITICAL")
    
    # Time-Series & Future Projection Outputs
    forecast_horizon: Optional[int] = Field(default=None, description="Number of forecasted intervals")
    forecast_records: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of {ds, yhat, yhat_lower, yhat_upper} records")
    trajectory: Optional[List[Dict[str, Any]]] = Field(default=None, description="Chronological trajectory records for line charts")
    projected_end_value: Optional[float] = Field(default=None, description="Projected final trajectory value")
    projected_change_pct: Optional[float] = Field(default=None, description="Projected percentage growth or decline")
    trend: Optional[str] = Field(default=None, description="Upward, Downward, or Neutral")
    series_summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary statistics of the projected time series (average, peak, trough, range)")
    is_trend_extrapolated: Optional[bool] = Field(default=None, description="True only when no feature carried a usable historical drift, so the target itself was extrapolated using its own historical annual trend as a last resort. False when every future point is a genuine model prediction driven by projected feature values.")
    feature_trend_basis: Optional[Dict[str, Any]] = Field(default=None, description="Per-feature drift assumptions actually applied when constructing future rows (slope per year, source: 'projected_from_history' | 'frozen_at_input').")
    dataset_preview: Optional[List[Dict[str, Any]]] = Field(default=None, description="This prediction reshaped into the original dataset's own column layout — one row per trajectory point (or a single row for a manual, non-forecast prediction) — with the target column and any feature that was drifted from history labeled '<column> (Predicted)'. Columns that were explicitly typed by the caller or held constant (e.g. Holiday_Flag) are left unlabeled since they are not model-generated values.")
    
    # Batch Outputs
    batch_predictions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Scored records for batch CSV requests")
    
    # Performance Telemetry
    inference_latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    features_used: List[str] = Field(default_factory=list, description="List of input features fed to the model")
    inputs_used: Optional[Dict[str, Any]] = Field(default=None, description="Inputs used for inference")
    error_message: Optional[str] = Field(default=None, description="Details if inference failed")


class FeatureSchemaItem(BaseModel):
    name: str
    data_type: str  # numeric, categorical, datetime, text
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median_value: Optional[float] = None
    mean_value: Optional[float] = None
    default_value: Optional[Any] = None
    allowed_categories: Optional[List[str]] = None


class InferenceSchemaResponse(BaseModel):
    """
    Schema describing required inputs, types, valid ranges, temporal metadata, and sample payload for a trained pipeline.
    """
    execution_id: str
    task_type: str
    target_column: Optional[str] = None
    target_classes: List[str] = Field(default_factory=list)
    features: List[FeatureSchemaItem] = Field(default_factory=list)
    sample_payload: Dict[str, Any] = Field(default_factory=dict)
    last_historical_row: Optional[Dict[str, Any]] = Field(default=None, description="Actual most recent real historical record (not a synthetic median) — the anchor used for pure future forecasts with no caller-supplied inputs.")
    time_series_meta: Optional[Dict[str, Any]] = None
    has_temporal_feature: bool = Field(default=False, description="True if dataset contains temporal/date/year features.")
    temporal_column: Optional[str] = Field(default=None, description="Name of the detected date/time/year column.")
    min_year: Optional[int] = Field(default=None, description="Minimum year found in training data.")
    max_year: Optional[int] = Field(default=None, description="Maximum year found in training data.")
    annual_trend_pct: Optional[float] = Field(default=None, description="Empirical historical annual trend/growth rate percentage for the TARGET column. Informational only — future projections now drive predictions by projecting individual feature values, not by applying this rate to the output.")
    feature_trends: Optional[Dict[str, Dict[str, float]]] = Field(default=None, description="Per-feature empirical drift vs. the temporal column (slope_per_unit_time, pct_per_unit_time), computed from training data. Features absent from this map are held constant at whatever value the caller supplies for future projections.")
    entity_value_sets: Optional[Dict[str, List[Any]]] = Field(default=None, description="Distinct historical values captured for low-cardinality entity/grouping columns (e.g. Store IDs [1..45]).")
    seasonal_profile: Optional[Dict[str, Dict[str, float]]] = Field(default=None, description="Historical feature averages grouped by date subcomponents (week, month).")


class NativeCadenceDatasetRequest(BaseModel):
    """
    Request parameters for generating a full row-by-row synthetic future dataset
    at the original dataset's native temporal cadence (e.g. weekly per store).
    """
    horizon_years: Optional[int] = Field(default=3, description="Number of future years to generate (default: 3).")
    periods: Optional[int] = Field(default=None, description="Explicit number of step periods to generate (e.g., 156 weeks).")
    step_unit: Optional[str] = Field(default="auto", description="Temporal step unit: 'week', 'month', 'year', or 'auto'.")
    feature_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom feature overrides for the generated dataset.")


class NativeCadenceDatasetResponse(BaseModel):
    """
    Response containing the generated row-by-row synthetic future dataset
    formatted in the exact column schema of the original dataset.
    """
    status: str = Field(default="SUCCESS")
    execution_id: str
    target_column: Optional[str] = None
    step_column: Optional[str] = Field(default=None, description="The temporal subcomponent column used for step pacing (e.g. Date_week).")
    cadence: str = Field(default="weekly", description="Detected cadence (weekly, monthly, annual).")
    total_rows: int = Field(default=0, description="Total number of generated rows across all entities.")
    columns: List[str] = Field(default_factory=list, description="Original dataset column headers.")
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list, description="Top 10 generated rows formatted as dataset preview.")
    is_trend_extrapolated: Optional[bool] = Field(default=None, description="True only when no feature carried a usable historical drift, so the target itself was extrapolated using its own historical trend as a last resort — the same honesty flag the annual trajectory path already surfaces.")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="All generated records.")


class CombinedDatasetRequest(BaseModel):
    """
    Request parameters for generating a combined dataset containing both
    historical records and synthetic predicted future records.
    """
    horizon_years: Optional[int] = Field(default=3, description="Number of future years to generate (default: 3).")
    periods: Optional[int] = Field(default=None, description="Explicit number of step periods to generate.")
    step_unit: Optional[str] = Field(default="auto", description="Temporal step unit: 'week', 'month', 'year', or 'auto'.")
    feature_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom feature overrides for future periods.")
    max_historical_rows: Optional[int] = Field(default=1000, description="Max historical rows to include in concatenation (default: 1000).")


class CombinedDatasetResponse(BaseModel):
    """
    Response containing the combined historical and predicted dataset
    formatted with original CSV column headers and a RECORD_TYPE column.
    """
    status: str = Field(default="SUCCESS")
    execution_id: str
    target_column: Optional[str] = None
    cadence: str = Field(default="weekly", description="Detected cadence (weekly, monthly, annual).")
    historical_rows_count: int = Field(default=0, description="Number of historical records included.")
    future_rows_count: int = Field(default=0, description="Number of predicted future records included.")
    total_rows_count: int = Field(default=0, description="Total number of combined records.")
    columns: List[str] = Field(default_factory=list, description="Original dataset column headers plus RECORD_TYPE.")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="All combined records (historical + predicted).")

