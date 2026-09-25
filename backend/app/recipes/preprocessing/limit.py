import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class LimitRecipe(BaseRecipe):
    recipe_id = "limit"
    name = "Limit"
    version = "1.0.0"
    category = "preprocessing"
    description = "Restricts row count (head/tail/random sample)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rows": {"type": "integer", "title": "Row Count", "default": 1000, "minimum": 1},
                "mode": {"type": "string", "title": "Mode", "enum": ["head", "tail", "random"], "default": "head"},
                "random_state": {"type": "integer", "title": "Random Seed", "default": 42}
            },
            "required": ["rows"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        return ["Row count must be positive."] if int(config.get("rows", 0)) <= 0 else []

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("LimitRecipe expects 'dataframe' in inputs.")
        n = int(config.get("rows", 1000))
        mode = config.get("mode", "head")
        if mode == "tail":
            out = df.tail(n)
        elif mode == "random":
            out = df.sample(n=min(n, len(df)), random_state=int(config.get("random_state", 42)))
        else:
            out = df.head(n)
        out = out.reset_index(drop=True)
        return {"dataframe": out, "feature_names": list(out.columns), "output_summary": {"row_count": len(out)}}

    def to_code(self, config: Dict[str, Any]) -> str:
        n = config.get("rows", 1000)
        mode = config.get("mode", "head")
        return f"df = df.{mode}({n})" if mode != "random" else f"df = df.sample(n={n}, random_state={config.get('random_state', 42)})"