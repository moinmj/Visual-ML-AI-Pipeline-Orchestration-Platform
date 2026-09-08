from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union


class PredictionRequest(BaseModel):
    """
    Standard request payload for live model predictions.
    Supports single records, batch records, or time-series forecast specifications.
    """
    inputs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None,
        description="Single feature dictionary or list of feature dictionaries for tabular ML / Anomaly scoring."
    )
    forecast_horizon: Optional[int] = Field(
        default=None,
        description="Number of future intervals to forecast for time-series models."
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
        description="Frequency code for forecasting (e.g., 'D', 'W', 'M', 'H')."
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


class PredictionResponse(BaseModel):
    """
    Standardized response payload for model predictions across all task families.
    """
    status: str = Field(default="SUCCESS", description="SUCCESS or FAILED")
    task_type: str = Field(description="classification, regression, time_series_forecasting, or anomaly_detection")
    execution_id: str = Field(description="Pipeline execution ID that trained this model")
    
    # Classification & Regression Outputs
    prediction: Optional[Any] = Field(default=None, description="Decoded class label or continuous numeric target")
    prediction_raw: Optional[Any] = Field(default=None, description="Raw model output before class decoding")
    confidence: Optional[float] = Field(default=None, description="Confidence score percentage (0-100%) for classification")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Per-class probability distribution")
    
    # Anomaly Detection Outputs
    is_anomaly: Optional[int] = Field(default=None, description="1 if anomalous, 0 if normal")
    anomaly_score: Optional[float] = Field(default=None, description="Continuous outlier score (0.0 to 1.0)")
    verdict: Optional[str] = Field(default=None, description="Human-friendly risk verdict")
    risk_level: Optional[str] = Field(default=None, description="LOW, MEDIUM, HIGH, or CRITICAL")
    
    # Time-Series Forecasting Outputs
    forecast_horizon: Optional[int] = Field(default=None, description="Number of forecasted intervals")
    forecast_records: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of {ds, yhat, yhat_lower, yhat_upper} records")
    projected_end_value: Optional[float] = Field(default=None, description="Projected final trajectory value")
    projected_change_pct: Optional[float] = Field(default=None, description="Projected percentage growth or decline")
    trend: Optional[str] = Field(default=None, description="Upward, Downward, or Neutral")
    
    # Batch Outputs
    batch_predictions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Scored records for batch CSV requests")
    
    # Performance Telemetry
    inference_latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    features_used: List[str] = Field(default_factory=list, description="List of input features fed to the model")
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
    Schema describing required inputs, types, valid ranges, and sample payload for a trained pipeline.
    """
    execution_id: str
    task_type: str
    target_column: Optional[str] = None
    target_classes: List[str] = Field(default_factory=list)
    features: List[FeatureSchemaItem] = Field(default_factory=list)
    sample_payload: Dict[str, Any] = Field(default_factory=dict)
    time_series_meta: Optional[Dict[str, Any]] = None
