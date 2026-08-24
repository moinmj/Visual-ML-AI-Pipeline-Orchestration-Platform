from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class RecipeMetadata(BaseModel):
    recipe_id: str
    name: str
    version: str = "1.0.0"
    category: str  # ingestion, preprocessing, feature_engineering, splitting, training, evaluation, anomaly, forecasting, governance
    description: str
    input_types: List[str]  # e.g. ["dataframe"], ["model", "dataframe"]
    output_types: List[str]  # e.g. ["dataframe"], ["model"], ["metrics"]
    parameters_schema: Dict[str, Any]


class BaseRecipe(ABC):
    """
    Abstract base class for all Pipeline Recipes.
    Every transformation, ML model, or loader must implement this interface.
    """

    recipe_id: str
    name: str
    version: str = "1.0.0"
    category: str
    description: str
    input_types: List[str]
    output_types: List[str]

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Returns JSON Schema for the recipe's configuration parameters.
        The frontend will use this schema to render dynamic UI forms.
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validates provided configuration against requirements.
        Returns a list of error strings if invalid, empty list if valid.
        """
        return []

    @abstractmethod
    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Executes the recipe logic deterministically.
        :param inputs: Dict of input artifacts (e.g. {"df": DataFrame} or {"model": Model, "test_df": DataFrame})
        :param config: User configuration matching get_schema()
        :param context: Execution context (storage manager, logger, MLflow, etc.)
        :return: Dict of outputs (e.g. {"df": transformed_df} or {"model": trained_model})
        """
        pass

    def to_code(self, config: Dict[str, Any]) -> str:
        """
        Renders the recipe configuration as a clean Python code snippet for the Node Code-View (Section 9).
        """
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in config.items())
        return f"# Recipe: {self.name}\n# ID: {self.recipe_id}\nresult = {self.recipe_id}({args_str})"

    def to_metadata(self) -> RecipeMetadata:
        return RecipeMetadata(
            recipe_id=self.recipe_id,
            name=self.name,
            version=self.version,
            category=self.category,
            description=self.description,
            input_types=self.input_types,
            output_types=self.output_types,
            parameters_schema=self.get_schema()
        )
