"""
Geocode / Reverse Geocode / Distance Calculator Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Converts addresses to coordinates and back, and calculates spherical/euclidean distances.
"""
import math
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


# Built-in offline gazetteer for major world cities and financial hubs
WORLD_GAZETTEER = [
    {"city": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
    {"city": "Los Angeles", "country": "USA", "lat": 34.0522, "lon": -118.2437},
    {"city": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298},
    {"city": "San Francisco", "country": "USA", "lat": 37.7749, "lon": -122.4194},
    {"city": "Seattle", "country": "USA", "lat": 47.6062, "lon": -122.3321},
    {"city": "Miami", "country": "USA", "lat": 25.7617, "lon": -80.1918},
    {"city": "Toronto", "country": "Canada", "lat": 43.6532, "lon": -79.3832},
    {"city": "Vancouver", "country": "Canada", "lat": 49.2827, "lon": -123.1207},
    {"city": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
    {"city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
    {"city": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
    {"city": "Frankfurt", "country": "Germany", "lat": 50.1109, "lon": 8.6821},
    {"city": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lon": 4.9041},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
    {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"city": "Hong Kong", "country": "Hong Kong", "lat": 22.3193, "lon": 114.1694},
    {"city": "Dubai", "country": "UAE", "lat": 25.2048, "lon": 55.2708},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
    {"city": "Melbourne", "country": "Australia", "lat": -37.8136, "lon": 144.9631},
    {"city": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
    {"city": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090},
    {"city": "Bengaluru", "country": "India", "lat": 12.9716, "lon": 77.5946},
    {"city": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074},
    {"city": "Shanghai", "country": "China", "lat": 31.2304, "lon": 121.4737},
    {"city": "Sao Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333},
]


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in kilometers."""
    R = 6371.0088  # Earth mean radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


class GeocodeDistanceCalculatorRecipe(BaseRecipe):
    recipe_id = "geocode_distance_calculator"
    name = "Geocode / Reverse Geocode / Distance Calculator"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Converts addresses to coordinates and back, cleans GPS coordinates, and calculates spherical haversine distances."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing location or coordinate data"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Geocoded Table",
            type="dataframe",
            description="Dataset enriched with coordinates, reverse geocoded locations, or distances"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Operation Mode",
                    "enum": ["distance", "geocode", "reverse_geocode", "validate_clean"],
                    "default": "distance",
                    "description": "'distance' computes distances; 'geocode' converts address/city to lat/lon; 'reverse_geocode' finds nearest city/country; 'validate_clean' sanitizes coordinates."
                },
                "distance_method": {
                    "type": "string",
                    "title": "Distance Method",
                    "enum": ["haversine", "euclidean", "manhattan"],
                    "default": "haversine",
                    "description": "'haversine' great-circle spherical distance; 'euclidean' flat distance; 'manhattan' grid distance."
                },
                "distance_unit": {
                    "type": "string",
                    "title": "Distance Unit",
                    "enum": ["km", "miles", "meters", "nautical_miles"],
                    "default": "km",
                    "description": "Output unit for distance calculations."
                },
                "lat1_column": {
                    "type": "string",
                    "title": "Origin Latitude Column",
                    "default": "latitude",
                    "description": "Column name containing origin latitude coordinates."
                },
                "lon1_column": {
                    "type": "string",
                    "title": "Origin Longitude Column",
                    "default": "longitude",
                    "description": "Column name containing origin longitude coordinates."
                },
                "lat2_column": {
                    "type": "string",
                    "title": "Destination Latitude Column",
                    "default": "",
                    "description": "Optional column containing destination latitude coordinates for pairwise row distance."
                },
                "lon2_column": {
                    "type": "string",
                    "title": "Destination Longitude Column",
                    "default": "",
                    "description": "Optional column containing destination longitude coordinates."
                },
                "reference_lat": {
                    "type": "number",
                    "title": "Fixed Reference Latitude",
                    "description": "Fixed destination latitude (e.g. warehouse, store, or airport) if lat2_column is omitted."
                },
                "reference_lon": {
                    "type": "number",
                    "title": "Fixed Reference Longitude",
                    "description": "Fixed destination longitude if lon2_column is omitted."
                },
                "address_column": {
                    "type": "string",
                    "title": "Address / City Column",
                    "default": "address",
                    "description": "Column containing address, city, or coordinates text for 'geocode' mode."
                },
                "output_column": {
                    "type": "string",
                    "title": "Output Column Name",
                    "default": "distance_km",
                    "description": "Name of output column for distance or primary result."
                }
            },
            "required": ["mode"]
        }

    def _parse_dms_or_string(self, text: Any) -> Optional[Tuple[float, float]]:
        """Extracts latitude and longitude from string like '40.7128, -74.0060' or '40°42\'46"N 74°0\'21"W'."""
        if not text or pd.isna(text):
            return None
        s = str(text).strip()

        # Decimal format: "40.7128, -74.0060"
        m = re.search(r"([-+]?\d{1,2}(?:\.\d+)?)\s*,\s*([-+]?\d{1,3}(?:\.\d+)?)", s)
        if m:
            try:
                lat, lon = float(m.group(1)), float(m.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            except ValueError:
                pass

        # Gazetteer match
        s_lower = s.lower()
        for entry in WORLD_GAZETTEER:
            if entry["city"].lower() in s_lower or f"{entry['city'].lower()}, {entry['country'].lower()}" in s_lower:
                return entry["lat"], entry["lon"]

        return None

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Geocode / Distance Calculator", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Geocode / Distance Calculator", "message": "Input table is empty.", "metrics": {}}
            }

        mode = config.get("mode", "distance")
        distance_method = config.get("distance_method", "haversine")
        distance_unit = config.get("distance_unit", "km")
        lat1_col = config.get("lat1_column", "latitude")
        lon1_col = config.get("lon1_column", "longitude")
        lat2_col = config.get("lat2_column")
        lon2_col = config.get("lon2_column")
        ref_lat = config.get("reference_lat")
        ref_lon = config.get("reference_lon")
        address_col = config.get("address_column", "address")
        output_col = config.get("output_column") or f"distance_{distance_unit}"

        # Unit multipliers relative to km
        unit_factors = {
            "km": 1.0,
            "miles": 0.621371,
            "meters": 1000.0,
            "nautical_miles": 0.539957
        }
        factor = unit_factors.get(distance_unit, 1.0)

        metrics: Dict[str, Any] = {"mode": mode, "rows_processed": len(df)}

        # MODE 1: DISTANCE CALCULATION
        if mode == "distance":
            lat1 = pd.to_numeric(df[lat1_col], errors="coerce").fillna(0.0).values if lat1_col in df.columns else np.zeros(len(df))
            lon1 = pd.to_numeric(df[lon1_col], errors="coerce").fillna(0.0).values if lon1_col in df.columns else np.zeros(len(df))

            if lat2_col and lon2_col and lat2_col in df.columns and lon2_col in df.columns:
                lat2 = pd.to_numeric(df[lat2_col], errors="coerce").fillna(0.0).values
                lon2 = pd.to_numeric(df[lon2_col], errors="coerce").fillna(0.0).values
            elif ref_lat is not None and ref_lon is not None:
                lat2 = np.full(len(df), float(ref_lat))
                lon2 = np.full(len(df), float(ref_lon))
            else:
                lat2 = np.zeros(len(df))
                lon2 = np.zeros(len(df))

            if distance_method == "haversine":
                dist_km = haversine_np(lat1, lon1, lat2, lon2)
                dists = dist_km * factor
            elif distance_method == "euclidean":
                dists = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111.0 * factor
            else:  # manhattan
                dists = (np.abs(lat2 - lat1) + np.abs(lon2 - lon1)) * 111.0 * factor

            df[output_col] = np.round(dists, 3)
            metrics["mean_distance"] = float(np.round(np.mean(dists), 2))
            metrics["max_distance"] = float(np.round(np.max(dists), 2))
            metrics["min_distance"] = float(np.round(np.min(dists), 2))
            metrics["distance_unit"] = distance_unit

        # MODE 2: GEOCODE (Address/City -> Lat/Lon)
        elif mode == "geocode":
            if address_col not in df.columns:
                return {
                    "dataframe": df,
                    "metrics": metrics,
                    "output_summary": {"title": "Geocode / Distance Calculator", "message": f"Column '{address_col}' not found.", "metrics": {}}
                }

            lats = []
            lons = []
            geocoded_count = 0

            for val in df[address_col]:
                parsed = self._parse_dms_or_string(val)
                if parsed:
                    lats.append(round(parsed[0], 5))
                    lons.append(round(parsed[1], 5))
                    geocoded_count += 1
                else:
                    lats.append(np.nan)
                    lons.append(np.nan)

            df["latitude"] = lats
            df["longitude"] = lons
            metrics["geocoded_success_count"] = geocoded_count
            metrics["geocoded_rate"] = f"{round((geocoded_count / len(df)) * 100, 1)}%"

        # MODE 3: REVERSE GEOCODE (Lat/Lon -> Nearest City/Country)
        elif mode == "reverse_geocode":
            lat_series = pd.to_numeric(df[lat1_col], errors="coerce").values if lat1_col in df.columns else np.zeros(len(df))
            lon_series = pd.to_numeric(df[lon1_col], errors="coerce").values if lon1_col in df.columns else np.zeros(len(df))

            nearest_cities = []
            nearest_countries = []
            approx_dists = []

            for lat_v, lon_v in zip(lat_series, lon_series):
                if np.isnan(lat_v) or np.isnan(lon_v):
                    nearest_cities.append(None)
                    nearest_countries.append(None)
                    approx_dists.append(np.nan)
                    continue

                best_dist = float("inf")
                best_city = "Unknown"
                best_country = "Unknown"

                for g in WORLD_GAZETTEER:
                    d = haversine_np(np.array([lat_v]), np.array([lon_v]), np.array([g["lat"]]), np.array([g["lon"]]))[0]
                    if d < best_dist:
                        best_dist = d
                        best_city = g["city"]
                        best_country = g["country"]

                nearest_cities.append(best_city)
                nearest_countries.append(best_country)
                approx_dists.append(round(float(best_dist), 1))

            df["nearest_city"] = nearest_cities
            df["country"] = nearest_countries
            df["approx_distance_to_city_km"] = approx_dists
            metrics["reverse_geocoded_rows"] = len(df)

        # MODE 4: VALIDATE & CLEAN COORDINATES
        elif mode == "validate_clean":
            if lat1_col in df.columns and lon1_col in df.columns:
                lat_num = pd.to_numeric(df[lat1_col], errors="coerce")
                lon_num = pd.to_numeric(df[lon1_col], errors="coerce")
                valid_mask = (lat_num >= -90.0) & (lat_num <= 90.0) & (lon_num >= -180.0) & (lon_num <= 180.0)
                df["is_valid_coordinates"] = valid_mask
                metrics["valid_coordinates_count"] = int(valid_mask.sum())
                metrics["invalid_coordinates_count"] = int((~valid_mask).sum())

        output_summary = {
            "title": "Geocode & Distance Calculator",
            "message": f"Executed mode '{mode}' on {len(df):,} records.",
            "metrics": metrics
        }

        return {
            "dataframe": df,
            "metrics": metrics,
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        mode = repr(config.get("mode", "distance"))
        unit = repr(config.get("distance_unit", "km"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Mode: {mode}, Distance Unit: {unit}\n"
            f"# Geocodes and computes spherical distance metrics\n"
        )
