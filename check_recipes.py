import os
# dummy Ozone values: only needed so the config loads, nothing is uploaded
for k, v in {"OZONE_ENDPOINT": "http://localhost:1", "OZONE_ACCESS_KEY": "x",
             "OZONE_SECRET_KEY": "x", "OZONE_BUCKET": "x"}.items():
    os.environ.setdefault(k, v)

import backend.app.recipes
from backend.app.recipes.base.registry import recipe_registry
from backend.app.core.exceptions import ValidationException

allr = recipe_registry.list_all()
enabled = [r for r in allr if r.enabled]
planned = [r for r in allr if not r.enabled]
print("Total:", len(allr), "| enabled:", len(enabled), "| planned:", len(planned))   # 217 | 53 | 164

print("\nExample planned recipe JSON:")
print(recipe_registry.get_metadata("postgresql_connector").model_dump_json(indent=2))

try:
    recipe_registry.get("postgresql_connector")
except ValidationException as e:
    print("\nRunning a planned recipe is blocked:", e.message)