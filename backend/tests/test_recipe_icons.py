# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
import pytest
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.base.icons import OFFICIAL_RECIPE_ICONS, CATEGORY_FALLBACK_ICONS, resolve_recipe_icon
import backend.app.recipes  # Ensure registration


def test_all_224_recipes_have_valid_official_icons():
    """
    Ensures that every registered recipe has an official icon mapping,
    points to an existing SVG file on disk, and contains valid XML.
    """
    recipes = recipe_registry.list_all()
    assert len(recipes) >= 224, f"Expected at least 224 recipes, found {len(recipes)}"

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    for recipe in recipes:
        assert recipe.icon is not None and recipe.icon != "", f"Recipe {recipe.recipe_id} missing icon slug"
        assert recipe.icon_url is not None and recipe.icon_url.startswith("/static/icons/recipes/"), (
            f"Recipe {recipe.recipe_id} invalid icon_url: {recipe.icon_url}"
        )

        rel_path = recipe.icon_url.replace("/static/", "", 1).strip("/")
        full_path = os.path.normpath(os.path.join(root_dir, "backend", "app", "static", *rel_path.split("/")))
        assert os.path.exists(full_path), f"Recipe {recipe.recipe_id} icon file not found: {full_path}"

        # Verify XML validity
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            tree = ET.fromstring(content)
            assert tree.tag.endswith("svg"), f"Root tag of {full_path} is not svg"
        except Exception as e:
            pytest.fail(f"Invalid SVG XML for recipe {recipe.recipe_id} at {full_path}: {e}")


def test_postgresql_and_output_icons_authenticity():
    """
    Verifies that postgresql, notifications, and export_writer have authentic,
    technology-accurate iconography (not speech bubbles or generic csv).
    """
    pg_meta = recipe_registry.get_metadata("postgresql_connector")
    assert pg_meta.icon == "postgresql"
    assert pg_meta.icon_url == "/static/icons/recipes/postgresql.svg"

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pg_path = os.path.join(root_dir, "backend", "app", "static", "icons", "recipes", "postgresql.svg")
    with open(pg_path, "r", encoding="utf-8") as f:
        pg_svg = f.read()

    # Verify elephant characteristics and absence of old chat bubble path
    assert "M12 4c-4.4 0-8 3.1-8 7" not in pg_svg, "PostgreSQL SVG must not be a speech bubble"
    assert "Trunk" in pg_svg or "Slonik" in pg_svg or "336791" in pg_svg, "PostgreSQL SVG must have Slonik blue/features"

    # Notifications must have a bell, not csv.svg
    notif_meta = recipe_registry.get_metadata("notifications")
    assert notif_meta.icon == "bell"
    assert notif_meta.icon_url == "/static/icons/recipes/notifications.svg"

    # Export Writer must have export-writer.svg, not csv.svg
    export_meta = recipe_registry.get_metadata("export_writer")
    assert export_meta.icon == "export"
    assert export_meta.icon_url == "/static/icons/recipes/export-writer.svg"


def test_category_fallback_ingestion_not_csv():
    """
    Ensures ingestion category fallback is database.svg, not csv.svg.
    """
    fallback = CATEGORY_FALLBACK_ICONS.get("ingestion")
    assert fallback is not None
    assert fallback[1] != "csv.svg", "Ingestion category fallback must not be csv.svg"
    assert fallback[1] == "database.svg"
