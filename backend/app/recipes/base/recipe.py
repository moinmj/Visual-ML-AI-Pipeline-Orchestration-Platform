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
    # Kind of unstructured file the recipe works on: "image", "audio" or "video".
    # None for ordinary table recipes. The frontend uses it to filter the file picker.
    media_type: Optional[str] = None
    description: str
    input_types: List[str]
    output_types: List[str]
    inputs: List[RecipePort] = []
    outputs: List[RecipePort] = []
    parameters_schema: Dict[str, Any]
    # True = implemented and usable. False = planned / "coming soon": listed so the
    # frontend can show it, but it cannot be run, tested or validated yet.
    enabled: bool = True

class BaseRecipe(ABC):
    """
    Abstract base class for all Pipeline Recipes.
    Every transformation, ML model, or loader must implement this interface.
    """

    recipe_id: str
    name: str
    version: str = "1.0.0"
    category: str
    media_type: Optional[str] = None  # "image" | "audio" | "video" for media recipes, else None
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

    @classmethod
    def enrich_schema_input_types(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures all properties in the schema have an explicit 'input_type' matching
        frontend Inspector widgets and scrubs empty string defaults on column/metric/target fields.
        """
        if not isinstance(schema, dict):
            return schema

        props = schema.get("properties")
        if not isinstance(props, dict):
            return schema

        for prop_name, prop_data in props.items():
            if not isinstance(prop_data, dict):
                continue

            # Recursively enrich nested properties (e.g. array items with object properties)
            if "properties" in prop_data:
                cls.enrich_schema_input_types(prop_data)
            if "items" in prop_data and isinstance(prop_data["items"], dict) and "properties" in prop_data["items"]:
                cls.enrich_schema_input_types(prop_data["items"])

            low_name = prop_name.lower()

            # Rule: Keep parameter schemas free of empty string defaults on keys containing column, metric, or target
            if any(k in low_name for k in ("column", "metric", "target")):
                if prop_data.get("default") == "":
                    prop_data.pop("default", None)

            # If input_type is already set, normalize it if necessary
            current_input_type = prop_data.get("input_type")
            if current_input_type:
                if current_input_type == "select":
                    prop_data["input_type"] = "dropdown"
                continue

            prop_type = prop_data.get("type", "")
            has_enum = "enum" in prop_data

            if has_enum or prop_type in ("select", "radio"):
                if prop_type == "array" or ("items" in prop_data and isinstance(prop_data["items"], dict) and "enum" in prop_data["items"]):
                    prop_data["input_type"] = "multi_select"
                else:
                    prop_data["input_type"] = "dropdown"
            elif prop_type == "boolean":
                prop_data["input_type"] = "checkbox"
            elif prop_type in ("number", "integer", "float"):
                if "minimum" in prop_data and "maximum" in prop_data:
                    prop_data["input_type"] = "slider"
                else:
                    prop_data["input_type"] = "number"
            elif prop_type == "textarea" or any(kw in low_name for kw in ("query", "code", "sql", "formula", "prompt", "script", "template", "message_body")):
                prop_data["input_type"] = "textarea"
            elif prop_type == "object" or any(kw in low_name for kw in ("mapping", "conversions", "dictionary", "key_value")):
                prop_data["input_type"] = "key_value"
            elif prop_type == "password" or any(kw in low_name for kw in ("password", "secret")):
                prop_data["input_type"] = "password"
            elif any(kw in low_name for kw in ("dataset_id", "file_id", "image_files", "input_file")):
                prop_data["input_type"] = "file_select"
            elif prop_type == "array":
                if any(kw in low_name for kw in ("column", "feature", "key", "group", "subset", "order", "drop", "select", "exclude", "include", "target")):
                    prop_data["input_type"] = "multi_column_select"
                else:
                    prop_data["input_type"] = "tags"
            else:
                is_excluded_col = any(low_name.endswith(sfx) for sfx in ("_format", "_name", "_prefix", "_suffix", "_unit", "_type", "_mode", "_action", "_style", "_strategy", "_method", "_provider", "_delim", "_delimiter", "_val", "_value"))
                if not is_excluded_col and (
                    low_name in ("column", "target", "label", "date_column", "time_column", "timestamp_column", "target_column")
                    or low_name.endswith(("_column", "_col", "_field"))
                    or ("column" in low_name and not any(kw in low_name for kw in ("columns", "order", "mapping", "prefix", "suffix", "count")))
                    or ("target" in low_name and not any(kw in low_name for kw in ("targets", "target_currencies", "metric", "directory", "size", "length", "ratio", "count")))
                ):
                    prop_data["input_type"] = "column_select"
                else:
                    prop_data["input_type"] = "text"

        return schema

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "get_schema" in cls.__dict__:
            orig_get_schema = cls.__dict__["get_schema"]
            def wrapped_get_schema(self, *args, **kw):
                raw_schema = orig_get_schema(self, *args, **kw)
                return BaseRecipe.enrich_schema_input_types(raw_schema)
            # Preserve original docstring and function metadata
            wrapped_get_schema.__name__ = orig_get_schema.__name__
            wrapped_get_schema.__doc__ = orig_get_schema.__doc__
            cls.get_schema = wrapped_get_schema

    def to_code(self, config: Dict[str, Any]) -> str:
        """
        Renders the recipe configuration as a clean Python code snippet for the Node Code-View (Section 9).
        """
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in config.items())
        return f"# Recipe: {self.name}\n# ID: {self.recipe_id}\nresult = {self.recipe_id}({args_str})"

    def to_metadata(self) -> RecipeMetadata:
        schema = self.get_schema()
        schema = self.enrich_schema_input_types(schema)
        return RecipeMetadata(
            recipe_id=self.recipe_id,
            name=self.name,
            version=self.version,
            category=self.category,
            group=getattr(self, "group", None),
            subgroup=getattr(self, "subgroup", None),
            media_type=self.media_type,
            description=self.description,
            input_types=self.input_types,
            output_types=self.output_types,
            inputs=self.get_inputs(),
            outputs=self.get_outputs(),
            parameters_schema=schema
        )