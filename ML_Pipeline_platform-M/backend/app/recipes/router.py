from fastapi import APIRouter, Query
from typing import List, Optional, Dict, Any
from backend.app.recipes.base.recipe import RecipeMetadata
from backend.app.recipes.base.registry import recipe_registry
import backend.app.recipes  # Trigger registration

router = APIRouter(prefix="/recipes", tags=["Recipes & Components"])


@router.get("/", response_model=List[RecipeMetadata])
async def list_recipes(category: Optional[str] = Query(None, description="Filter by category")):
    """
    Retrieve the entire catalog of registered AI/ML and data-processing recipes.
    """
    return recipe_registry.list_all(category=category)


@router.get("/categories", response_model=List[str])
async def get_recipe_categories():
    """
    Get all unique categories of recipes.
    """
    return recipe_registry.get_categories()


@router.get("/{recipe_id}", response_model=RecipeMetadata)
async def get_recipe(recipe_id: str):
    """
    Get recipe metadata by ID.
    """
    recipe = recipe_registry.get(recipe_id)
    return recipe.to_metadata()


@router.get("/{recipe_id}/schema", response_model=Dict[str, Any])
async def get_recipe_schema(recipe_id: str):
    """
    Retrieve JSON Schema for dynamic form rendering on the frontend.
    """
    recipe = recipe_registry.get(recipe_id)
    return {
        "recipe_id": recipe.recipe_id,
        "name": recipe.name,
        "parameters_schema": recipe.get_schema()
    }
