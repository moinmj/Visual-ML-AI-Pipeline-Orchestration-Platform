from typing import Dict, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipeMetadata
from backend.app.core.exceptions import NotFoundException
from backend.app.core.logging import logger


class RecipeRegistry:
    def __init__(self):
        self._recipes: Dict[str, BaseRecipe] = {}
        self._aliases: Dict[str, str] = {
            "classification_evaluator": "model_evaluator",
            "regression_evaluator": "model_evaluator",
            "model_governance_card": "mlflow_tracker",
            "governance_card": "mlflow_tracker",
        }

    def register(self, recipe: BaseRecipe):
        self._recipes[recipe.recipe_id] = recipe
        logger.debug(f"Registered recipe: {recipe.recipe_id} ({recipe.name})")

    def register_alias(self, alias: str, target_id: str):
        self._aliases[alias] = target_id
        logger.debug(f"Registered recipe alias: '{alias}' -> '{target_id}'")

    def get(self, recipe_id: str) -> BaseRecipe:
        resolved_id = self._aliases.get(recipe_id, recipe_id)
        if resolved_id not in self._recipes:
            raise NotFoundException("Recipe", recipe_id)
        return self._recipes[resolved_id]

    def has(self, recipe_id: str) -> bool:
        resolved_id = self._aliases.get(recipe_id, recipe_id)
        return resolved_id in self._recipes

    def list_all(self, category: Optional[str] = None) -> List[RecipeMetadata]:
        recipes = list(self._recipes.values())
        if category:
            recipes = [r for r in recipes if r.category.lower() == category.lower()]
        return [r.to_metadata() for r in recipes]

    def get_categories(self) -> List[str]:
        return sorted(list(set(r.category for r in self._recipes.values())))


recipe_registry = RecipeRegistry()
