from typing import Dict, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipeMetadata
from backend.app.core.exceptions import NotFoundException
from backend.app.core.logging import logger


class RecipeRegistry:
    def __init__(self):
        self._recipes: Dict[str, BaseRecipe] = {}

    def register(self, recipe: BaseRecipe):
        self._recipes[recipe.recipe_id] = recipe
        logger.debug(f"Registered recipe: {recipe.recipe_id} ({recipe.name})")

    def get(self, recipe_id: str) -> BaseRecipe:
        if recipe_id not in self._recipes:
            raise NotFoundException("Recipe", recipe_id)
        return self._recipes[recipe_id]

    def list_all(self, category: Optional[str] = None) -> List[RecipeMetadata]:
        recipes = list(self._recipes.values())
        if category:
            recipes = [r for r in recipes if r.category.lower() == category.lower()]
        return [r.to_metadata() for r in recipes]

    def get_categories(self) -> List[str]:
        return sorted(list(set(r.category for r in self._recipes.values())))


recipe_registry = RecipeRegistry()
