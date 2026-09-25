import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class CombineRecipe(BaseRecipe):
    recipe_id = "combine"
    name = "Combine"
    version = "1.0.0"
    category = "preprocessing"
    description = "Stacks (unions) rows from two tables with matching/overlapping columns."
    input_types = ["dataframe"]
    output_types = ["dataframe"]
    inputs = [
        RecipePort(id="left", label="Table A", type="dataframe", required=True, max_connections=1),
        RecipePort(id="right", label="Table B", type="dataframe", required=True, max_connections=1)
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "title": "Mode", "enum": ["union", "union_distinct", "intersect", "except"], "default": "union"}
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        left: pd.DataFrame = inputs.get("left") or inputs.get("dataframe")
        right: pd.DataFrame = inputs.get("right")
        if left is None or right is None:
            raise ValueError("CombineRecipe expects 'left' and 'right' dataframes in inputs.")
        mode = config.get("mode", "union")
        common = [c for c in left.columns if c in right.columns]
        left_c, right_c = left[common], right[common]
        if mode == "union":
            out = pd.concat([left_c, right_c], ignore_index=True)
        elif mode == "union_distinct":
            out = pd.concat([left_c, right_c], ignore_index=True).drop_duplicates().reset_index(drop=True)
        elif mode == "intersect":
            out = left_c.merge(right_c.drop_duplicates(), how="inner")
        else:  # except
            out = left_c.merge(right_c.drop_duplicates(), how="left", indicator=True)
            out = out[out["_merge"] == "left_only"].drop(columns=["_merge"])
        return {"dataframe": out.reset_index(drop=True), "feature_names": list(out.columns), "output_summary": {"row_count": len(out)}}

    def to_code(self, config: Dict[str, Any]) -> str:
        return f"df = pd.concat([left, right], ignore_index=True)  # mode={config.get('mode', 'union')}"