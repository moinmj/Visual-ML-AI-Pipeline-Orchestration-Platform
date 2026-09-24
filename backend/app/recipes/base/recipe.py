from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class RecipePort(BaseModel):
    id: str
    label: str
    type: str = "dataframe"
    required: bool = True
    max_connections: Optional[int] = 1
    description: Optional[str] = None


class RecipeMetadata(BaseModel):
    recipe_id: str
    name: str
    version: str = "1.0.0"
    category: str
    group: Optional[str] = None
    subgroup: Optional[str] = None
    description: str
    input_types: List[str]
    output_types: List[str]
    inputs: List[RecipePort] = []
    outputs: List[RecipePort] = []
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
    inputs: Optional[List[RecipePort]] = None
    outputs: Optional[List[RecipePort]] = None

    def get_inputs(self) -> List[RecipePort]:
        """
        Returns structured input port definitions for canvas handle rendering.
        Defaults to generating ports from input_types for backward compatibility.
        """
        if getattr(self, "inputs", None):
            return self.inputs
        ports: List[RecipePort] = []
        for i, t in enumerate(self.input_types or []):
            p_id = "input" if len(self.input_types) == 1 else f"input_{i+1}"
            label = "Input Table" if t == "dataframe" else f"Input ({t})"
            ports.append(RecipePort(id=p_id, label=label, type=t, required=True, max_connections=1))
        return ports

    def get_outputs(self) -> List[RecipePort]:
        """
        Returns structured output port definitions for canvas handle rendering.
        Defaults to generating ports from output_types for backward compatibility.
        """
        if getattr(self, "outputs", None):
            return self.outputs
        ports: List[RecipePort] = []
        for i, t in enumerate(self.output_types or []):
            p_id = "output" if len(self.output_types) == 1 else f"output_{i+1}"
            label = "Output Table" if t == "dataframe" else f"Output ({t})"
            ports.append(RecipePort(id=p_id, label=label, type=t, description="Primary output" if t != "dataframe" else "Primary dataset"))
        return ports

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
            inputs=self.get_inputs(),
            outputs=self.get_outputs(),
            parameters_schema=self.get_schema()
        )
