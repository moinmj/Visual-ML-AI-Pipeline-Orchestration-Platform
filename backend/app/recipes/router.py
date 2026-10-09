from fastapi import APIRouter, Depends, Query
from backend.app.core.security import get_current_user
from typing import List, Optional, Dict, Any
from backend.app.recipes.base.recipe import RecipeMetadata
from backend.app.recipes.base.registry import recipe_registry
import backend.app.recipes  # Trigger registration

# Requires a valid Bearer token (no tenant data here, but no anonymous access either).
router = APIRouter(prefix="/recipes", tags=["Recipes & Components"], dependencies=[Depends(get_current_user)])


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