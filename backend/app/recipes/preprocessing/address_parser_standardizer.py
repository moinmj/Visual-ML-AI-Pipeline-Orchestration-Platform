"""
Address Parser / Standardizer Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Splits freeform addresses into structured components (street, unit, city, state, postal code, country)
and applies standard postal formatting abbreviations (USPS / Universal).
"""
import re
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


STREET_ABBREVIATIONS = {
    r"\bstreet\b": "St",
    r"\bavenue\b": "Ave",
    r"\bboulevard\b": "Blvd",
    r"\bdrive\b": "Dr",
    r"\broad\b": "Rd",
    r"\blane\b": "Ln",
    r"\bcourt\b": "Ct",
    r"\bcircle\b": "Cir",
    r"\bparkway\b": "Pkwy",
    r"\bhighway\b": "Hwy",
    r"\bterrace\b": "Ter",
    r"\bplace\b": "Pl",
    r"\bway\b": "Way",
    r"\bsuite\b": "Ste",
    r"\bapartment\b": "Apt",
    r"\bunit\b": "Unit",
    r"\bfloor\b": "Fl",
    r"\bbuilding\b": "Bldg",
    r"\bnorth\b": "N",
    r"\bsouth\b": "S",
    r"\beast\b": "E",
    r"\bwest\b": "W",
    r"\bnortheast\b": "NE",
    r"\bnorthwest\b": "NW",
    r"\bsoutheast\b": "SE",
    r"\bsouthwest\b": "SW",
}

US_STATE_MAP = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC"
}


class AddressParserStandardizerRecipe(BaseRecipe):
    recipe_id = "address_parser_standardizer"
    name = "Address Parser / Standardizer"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Splits freeform addresses into components (number, street, unit, city, state, postal code) and standardizes formatting."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing raw address text"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Standardized Table",
            type="dataframe",
            description="Dataset enriched with parsed and standardized address fields"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "address_column": {
                    "type": "string",
                    "title": "Address Column",
                    "default": "address",
                    "description": "Column containing the freeform address strings to parse."
                },
                "column_prefix": {
                    "type": "string",
                    "title": "Output Column Prefix",
                    "default": "addr_",
                    "description": "Prefix for parsed component columns (e.g. 'addr_city', 'addr_state', 'addr_zip')."
                },
                "standardize_case": {
                    "type": "string",
                    "title": "Address Casing",
                    "enum": ["title", "upper", "preserve"],
                    "default": "title",
                    "description": "Casing convention for street names and cities ('title' or 'upper')."
                },
                "generate_single_line_standardized": {
                    "type": "boolean",
                    "title": "Generate Standardized Address String",
                    "default": True,
                    "description": "If true, creates a clean unified standardized address string column."
                }
            },
            "required": ["address_column"]
        }

    def _parse_address(self, raw_addr: Any, case_mode: str) -> Dict[str, Optional[str]]:
        if not raw_addr or pd.isna(raw_addr):
            return {"number": None, "street": None, "unit": None, "city": None, "state": None, "zip": None, "country": None}

        s = str(raw_addr).strip()
        # Apply standard street abbreviations (case-insensitive regex)
        for pat, rep in STREET_ABBREVIATIONS.items():
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)

        # Extract Zip/Postal Code (US 5-digit / ZIP+4 or Canada / UK / generic 5-6 digit)
        zip_match = re.search(r"\b(\d{5}(?:-\d{4})?|[A-Z]\d[A-Z]\s*\d[A-Z]\d)\b", s)
        postal_code = None
        if zip_match:
            postal_code = zip_match.group(1).upper()
            s = s[:zip_match.start()] + s[zip_match.end():]

        # Extract State/Province
        state = None
        # Check standard 2-letter uppercase codes
        state_match = re.search(r",?\s*\b([A-Z]{2})\b(?:\s*,|\s*$)", s)
        if state_match:
            cand = state_match.group(1)
            if cand in US_STATE_MAP.values():
                state = cand
                s = s[:state_match.start()] + s[state_match.end():]
        if not state:
            # Check full state names
            for full_state, code in US_STATE_MAP.items():
                pattern = rf"\b{full_state}\b"
                if re.search(pattern, s, re.IGNORECASE):
                    state = code
                    s = re.sub(pattern, "", s, flags=re.IGNORECASE)
                    break

        # Extract Unit / Suite / Apt
        unit = None
        unit_match = re.search(r"\b(Ste|Apt|Unit|Fl|Bldg|#)\s*([A-Za-z0-9-]+)\b", s, re.IGNORECASE)
        if unit_match:
            unit = f"{unit_match.group(1).capitalize()} {unit_match.group(2)}"
            s = s[:unit_match.start()] + s[unit_match.end():]

        # Split remaining by comma
        parts = [p.strip() for p in s.split(",") if p.strip()]

        city = None
        street_part = ""
        if len(parts) >= 2:
            street_part = parts[0]
            city = parts[1]
        elif len(parts) == 1:
            street_part = parts[0]

        # Extract street number and street name
        street_num = None
        street_name = None
        num_match = re.match(r"^(\d+[-\w]*)\s+(.*)$", street_part)
        if num_match:
            street_num = num_match.group(1)
            street_name = num_match.group(2).strip()
        else:
            street_name = street_part if street_part else None

        # Clean city name if it contains leftover spaces
        if city:
            city = re.sub(r"\s+", " ", city).strip()

        # Format casing
        def apply_case(text):
            if not text:
                return text
            if case_mode == "upper":
                return text.upper()
            elif case_mode == "title":
                return text.title()
            return text

        return {
            "number": street_num,
            "street": apply_case(street_name),
            "unit": unit,
            "city": apply_case(city),
            "state": state,
            "zip": postal_code,
            "country": "USA" if (state in US_STATE_MAP.values() and not postal_code) or (postal_code and re.match(r"^\d{5}", postal_code)) else None
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Address Parser & Standardizer", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Address Parser & Standardizer", "message": "Input table is empty.", "metrics": {}}
            }

        addr_col = config.get("address_column", "address")
        prefix = config.get("column_prefix", "addr_")
        case_mode = config.get("standardize_case", "title")
        gen_single_line = config.get("generate_single_line_standardized", True)

        if addr_col not in df.columns:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": len(df)},
                "output_summary": {"title": "Address Parser & Standardizer", "message": f"Column '{addr_col}' not found.", "metrics": {}}
            }

        numbers, streets, units, cities, states, zips, countries = [], [], [], [], [], [], []
        single_lines = []

        for val in df[addr_col]:
            p = self._parse_address(val, case_mode)
            numbers.append(p["number"])
            streets.append(p["street"])
            units.append(p["unit"])
            cities.append(p["city"])
            states.append(p["state"])
            zips.append(p["zip"])
            countries.append(p["country"])

            if gen_single_line:
                line_parts = []
                if p["number"] and p["street"]:
                    line_parts.append(f"{p['number']} {p['street']}")
                elif p["street"]:
                    line_parts.append(p["street"])
                if p["unit"]:
                    line_parts.append(p["unit"])
                if p["city"]:
                    line_parts.append(p["city"])
                if p["state"] and p["zip"]:
                    line_parts.append(f"{p['state']} {p['zip']}")
                elif p["state"]:
                    line_parts.append(p["state"])
                elif p["zip"]:
                    line_parts.append(p["zip"])
                single_lines.append(", ".join(line_parts) if line_parts else None)

        df[f"{prefix}street_number"] = numbers
        df[f"{prefix}street_name"] = streets
        df[f"{prefix}unit"] = units
        df[f"{prefix}city"] = cities
        df[f"{prefix}state"] = states
        df[f"{prefix}postal_code"] = zips
        df[f"{prefix}country"] = countries

        if gen_single_line:
            df[f"{prefix}standardized"] = single_lines

        output_summary = {
            "title": "Address Parser & Standardizer",
            "message": f"Parsed and standardized {len(df):,} addresses from column '{addr_col}'.",
            "metrics": {
                "addresses_processed": len(df),
                "postal_codes_found": sum(1 for z in zips if z is not None),
                "states_found": sum(1 for s in states if s is not None),
                "cities_found": sum(1 for c in cities if c is not None)
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        addr = repr(config.get("address_column", "address"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Column: {addr}\n"
            f"# Parses address into street, city, state, postal code\n"
        )
