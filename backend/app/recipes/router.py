import json
from fastapi import APIRouter, Depends, HTTPException, Query, status
from backend.app.core.security import TokenData, get_current_user
from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.app.recipes.base.recipe import RecipeMetadata
from backend.app.recipes.base.registry import recipe_registry
import backend.app.recipes  # Trigger registration

# Requires a valid Bearer token (no tenant data here, but no anonymous access either).
router = APIRouter(prefix="/recipes", tags=["Recipes & Components"], dependencies=[Depends(get_current_user)])

CurrentUser = Annotated[TokenData, Depends(get_current_user)]


@router.get("/", response_model=List[RecipeMetadata])
async def list_recipes(
    category: Optional[str] = Query(None, description="Filter by category"),
    enabled: Optional[bool] = Query(None, description="true = only usable recipes, false = only upcoming ones; omit for all"),
):
    """
    Retrieve the entire recipe catalog. Implemented recipes have "enabled": true;
    planned / coming-soon recipes are included with "enabled": false (show them
    in the list, but they cannot be run yet).
    """
    return recipe_registry.list_all(category=category, enabled=enabled)


@router.get("/categories", response_model=List[str])
async def get_recipe_categories():
    """
    Get all unique categories of recipes.
    """
    return recipe_registry.get_categories()


@router.get("/{recipe_id}", response_model=RecipeMetadata)
async def get_recipe(recipe_id: str):
    """
    Get recipe metadata by ID (works for enabled and planned recipes).
    """
    return recipe_registry.get_metadata(recipe_id)


@router.get("/{recipe_id}/schema", response_model=Dict[str, Any])
async def get_recipe_schema(recipe_id: str):
    """
    Retrieve JSON Schema for dynamic form rendering on the frontend.
    """
    meta = recipe_registry.get_metadata(recipe_id)
    return {
        "recipe_id": meta.recipe_id,
        "name": meta.name,
        "parameters_schema": meta.parameters_schema,
        "enabled": meta.enabled,
    }


# ---------------------------------------------------------------------------
# Recipe Run endpoint
# ---------------------------------------------------------------------------

class RecipeRunRequest(BaseModel):
    config: Dict[str, Any] = Field(
        ...,
        description=(
            "Recipe configuration matching the recipe's JSON schema. "
            "Example for **object_detection** (images from your unstructured files): "
            '{"image_files": ["unstructured/commute.jpg"], "model": "yolov8n.pt", '
            '"confidence_threshold": 0.25, "save_annotated_images": true}'
        ),
        examples=[{
            "image_files": ["unstructured/commute.jpg"],
            "model": "yolov8n.pt",
            "confidence_threshold": 0.25,
            "save_annotated_images": True,
        }],
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional pre-loaded inputs for the recipe (e.g. a serialised dataframe). "
            "For object_detection with image_files, leave this empty: the images are "
            "read from your Ozone folder using the paths in config."
        ),
    )


class RecipeRunResponse(BaseModel):
    recipe_id: str
    status: str = "success"
    metrics: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    preview_items: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optional previews, only if the recipe provides them."
    )
    rows: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Output rows (dataframe serialised as list of dicts). Up to 500 rows shown.",
    )
    columns: Optional[List[str]] = None
    total_rows: Optional[int] = None
    errors: Optional[List[str]] = None


def _plain(obj: Any) -> Any:
    """Turns NumPy numbers, timestamps etc. into plain JSON types so the response can always be serialised."""
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


@router.post(
    "/{recipe_id}/run",
    response_model=RecipeRunResponse,
    status_code=status.HTTP_200_OK,
)
async def run_recipe(
    recipe_id: str,
    body: RecipeRunRequest,
    user: CurrentUser,
):
    """
    **Run a single recipe and get its result back immediately.**

    A quick way to test one recipe on its own. For real pipelines (several
    steps, connected tables), use the workflow engine instead.

    - `config`: the recipe's settings, as described by `GET /recipes/{recipe_id}/schema`.
    - `inputs`: optional pre-loaded inputs. Recipes that need an upstream table
      won't work here, because the request body is not converted into a DataFrame.
    - The response has `metrics`, plus `rows`, `columns` and `total_rows` when the
      recipe returns a table (the first 500 rows).
    - Your `tenant_id` is added from your login token, so the recipe can only
      read and write your own Ozone folder.

    ### Example: Object Detection on uploaded images

    **Step 1 - Upload an image** with `POST /datasets/upload`, `data_type=unstructured`
    (note the `path` in the response, e.g. `unstructured/commute.jpg`).

    **Step 2 - Run detection** with `recipe_id = object_detection`:
    ```json
    {
      "config": {
        "image_files": ["unstructured/commute.jpg"],
        "model": "yolov8n.pt",
        "confidence_threshold": 0.25,
        "save_annotated_images": true
      },
      "inputs": {}
    }
    ```

    Annotated images (if requested) are saved in your Ozone folder under
    `outputs/object_detection/<model>_<time>/`, one new folder per run.

    ### Example: load a dataset
    `recipe_id = csv_loader`, with `{"config": {"dataset_id": "<uuid>"}, "inputs": {}}`
    """
    from starlette.concurrency import run_in_threadpool

    recipe = recipe_registry.get(recipe_id)

    # Inject tenant context - recipes need it to scope Ozone reads/writes.
    # It is added AFTER the body config, so a caller cannot override it.
    config = {**body.config, "tenant_id": user.tenant_id}

    # Validate config before executing
    errors = recipe.validate_config(config)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Recipe configuration is invalid.", "errors": errors},
        )

    try:
        result = await run_in_threadpool(recipe.execute, body.inputs, config)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Recipe execution failed.", "error": str(e)},
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"message": "Recipe dependency missing.", "error": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Recipe execution error.", "error": str(e)},
        )

    # Serialise the output dataframe (if any) to plain JSON rows.
    df = result.get("dataframe")
    rows: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    total_rows: Optional[int] = None
    if df is not None:
        total_rows = len(df)
        columns = [str(c) for c in df.columns]
        # to_json handles NumPy ints/bools, NaN (-> null) and timestamps
        rows = json.loads(df.head(500).to_json(orient="records", double_precision=15))

    metrics = _plain(result.get("metrics") or result.get("output_summary"))
    output_summary = _plain(result.get("output_summary"))
    preview_items = (metrics or {}).get("preview_items") if metrics else None

    return RecipeRunResponse(
        recipe_id=recipe_id,
        status="success",
        metrics=metrics,
        output_summary=output_summary,
        preview_items=preview_items,
        rows=rows,
        columns=columns,
        total_rows=total_rows,
    )