"""
Business-Day Calculator / Holiday Calendar Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Calculates business days, SLA deadlines, and flags regional holidays and weekends.
"""
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


# Standard fixed/floating holiday definitions across multiple regions
def get_regional_holidays(year: int, calendar_code: str) -> Dict[str, str]:
    """Returns mapping of 'YYYY-MM-DD' -> 'Holiday Name' for given year and calendar."""
    holidays: Dict[str, str] = {}
    code = calendar_code.upper()

    # Universal / Global
    holidays[f"{year}-01-01"] = "New Year's Day"
    holidays[f"{year}-12-25"] = "Christmas Day"

    if code in ["US", "GLOBAL", "ALL"]:
        holidays[f"{year}-07-04"] = "Independence Day"
        holidays[f"{year}-06-19"] = "Juneteenth"
        holidays[f"{year}-11-11"] = "Veterans Day"

        # US Floating holidays
        # MLK Day: 3rd Monday of Jan
        first_jan = date(year, 1, 1)
        mlk_offset = (0 - first_jan.weekday()) % 7 + 14
        holidays[(first_jan + timedelta(days=mlk_offset)).isoformat()] = "Martin Luther King Jr. Day"

        # Memorial Day: Last Monday of May
        last_may = date(year, 5, 31)
        mem_offset = (last_may.weekday() - 0) % 7
        holidays[(last_may - timedelta(days=mem_offset)).isoformat()] = "Memorial Day"

        # Labor Day: 1st Monday of Sep
        first_sep = date(year, 9, 1)
        labor_offset = (0 - first_sep.weekday()) % 7
        holidays[(first_sep + timedelta(days=labor_offset)).isoformat()] = "Labor Day"

        # Thanksgiving: 4th Thursday of Nov
        first_nov = date(year, 11, 1)
        tg_offset = (3 - first_nov.weekday()) % 7 + 21
        holidays[(first_nov + timedelta(days=tg_offset)).isoformat()] = "Thanksgiving Day"

    if code in ["UK", "EU", "ALL"]:
        holidays[f"{year}-12-26"] = "Boxing Day"
        holidays[f"{year}-05-01"] = "Early May Bank Holiday"

    if code in ["IN", "INDIA", "ALL"]:
        holidays[f"{year}-01-26"] = "Republic Day"
        holidays[f"{year}-08-15"] = "Independence Day"
        holidays[f"{year}-10-02"] = "Gandhi Jayanti"

    return holidays


class BusinessDayHolidayCalculatorRecipe(BaseRecipe):
    recipe_id = "business_day_holiday_calculator"
    name = "Business-Day Calculator / Holiday Calendar"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Calculates business days between dates, determines SLA completion dates, and flags holidays and weekends using regional calendars."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing date columns to calculate business days on"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Calculated Table",
            type="dataframe",
            description="Dataset enriched with business day counts, due dates, or holiday flags"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Operation Mode",
                    "enum": ["business_days_between", "add_business_days", "flag_calendar"],
                    "default": "business_days_between",
                    "description": "'business_days_between' counts working days; 'add_business_days' adds N working days for SLA deadlines; 'flag_calendar' marks holidays & weekends."
                },
                "start_date_column": {
                    "type": "string",
                    "title": "Start Date Column",
                    "default": "start_date",
                    "description": "Column containing start date or primary date."
                },
                "end_date_column": {
                    "type": "string",
                    "title": "End Date Column (for between mode)",
                    "default": "end_date",
                    "description": "Column containing end date when calculating business days between."
                },
                "calendar": {
                    "type": "string",
                    "title": "Regional Holiday Calendar",
                    "enum": ["US", "UK", "EU", "IN", "Global", "None"],
                    "default": "US",
                    "description": "Pre-configured country or regional holiday calendar."
                },
                "custom_holidays": {
                    "type": "array",
                    "title": "Custom Holiday Dates (YYYY-MM-DD)",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Optional list of additional specific holiday dates to exclude."
                },
                "n_business_days": {
                    "type": "integer",
                    "title": "Number of Business Days to Add",
                    "default": 5,
                    "description": "Fixed number of business days to add in 'add_business_days' mode."
                },
                "n_days_column": {
                    "type": "string",
                    "title": "Days Column (Variable SLA)",
                    "default": "",
                    "description": "Optional column name containing the dynamic number of days to add per row."
                },
                "output_column": {
                    "type": "string",
                    "title": "Output Column Name",
                    "default": "business_days",
                    "description": "Column name to store the calculated business days or target date."
                }
            },
            "required": ["mode", "start_date_column"]
        }

    def _get_holidays_set(self, start_years: Set[int], calendar: str, custom_holidays: List[str]) -> Tuple[Set[str], Dict[str, str]]:
        holidays_map: Dict[str, str] = {}
        for y in start_years:
            holidays_map.update(get_regional_holidays(y, calendar))
        for ch in custom_holidays:
            if ch:
                holidays_map[str(ch).strip()] = "Custom Holiday"
        return set(holidays_map.keys()), holidays_map

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Business Day Calculator", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Business Day Calculator", "message": "Input table is empty.", "metrics": {}}
            }

        mode = config.get("mode", "business_days_between")
        start_col = config.get("start_date_column", "start_date")
        end_col = config.get("end_date_column", "end_date")
        calendar = config.get("calendar", "US")
        custom_holidays = config.get("custom_holidays", [])
        n_days_fixed = config.get("n_business_days", 5)
        n_days_col = config.get("n_days_column")
        output_col = config.get("output_column") or ("business_days" if mode == "business_days_between" else "due_date")

        if start_col not in df.columns:
            # Fallback to first datetime column
            for c in df.columns:
                if "date" in str(c).lower():
                    start_col = c
                    break

        if start_col not in df.columns:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": len(df)},
                "output_summary": {"title": "Business Day Calculator", "message": f"Start date column '{start_col}' not found.", "metrics": {}}
            }

        start_dt = pd.to_datetime(df[start_col], errors="coerce")

        # Collect years span
        valid_dates = start_dt.dropna()
        years = set(valid_dates.dt.year.unique()) if len(valid_dates) > 0 else {datetime.now().year}
        if mode == "business_days_between" and end_col in df.columns:
            end_dt = pd.to_datetime(df[end_col], errors="coerce")
            valid_ends = end_dt.dropna()
            if len(valid_ends) > 0:
                years.update(valid_ends.dt.year.unique())

        # Include next year for adding days
        years.update({y + 1 for y in list(years)})
        holiday_dates, holiday_names = self._get_holidays_set(years, calendar, custom_holidays)

        metrics: Dict[str, Any] = {"mode": mode, "calendar": calendar, "rows_processed": len(df)}

        # MODE 1: BUSINESS DAYS BETWEEN
        if mode == "business_days_between":
            if end_col not in df.columns:
                return {
                    "dataframe": df,
                    "metrics": metrics,
                    "output_summary": {"title": "Business Day Calculator", "message": f"End date column '{end_col}' not found.", "metrics": {}}
                }

            end_dt = pd.to_datetime(df[end_col], errors="coerce")
            bus_counts = []

            for s_val, e_val in zip(start_dt, end_dt):
                if pd.isna(s_val) or pd.isna(e_val):
                    bus_counts.append(np.nan)
                    continue

                d1 = s_val.date()
                d2 = e_val.date()
                if d1 > d2:
                    d1, d2 = d2, d1
                    sign = -1
                else:
                    sign = 1

                # Generate date range excluding weekends & holidays
                curr = d1
                count = 0
                while curr < d2:
                    curr += timedelta(days=1)
                    if curr.weekday() < 5 and curr.isoformat() not in holiday_dates:
                        count += 1
                bus_counts.append(count * sign)

            df[output_col] = bus_counts
            metrics["mean_business_days"] = float(round(pd.Series(bus_counts).mean(), 2)) if bus_counts else 0.0

        # MODE 2: ADD BUSINESS DAYS (SLA DEADLINES)
        elif mode == "add_business_days":
            due_dates = []
            n_days_series = df[n_days_col] if (n_days_col and n_days_col in df.columns) else pd.Series([n_days_fixed] * len(df))

            for s_val, n_days in zip(start_dt, n_days_series):
                if pd.isna(s_val) or pd.isna(n_days):
                    due_dates.append(pd.NaT)
                    continue

                days_to_add = int(n_days)
                curr = s_val.date()
                added = 0
                step = 1 if days_to_add >= 0 else -1
                target = abs(days_to_add)

                while added < target:
                    curr += timedelta(days=step)
                    if curr.weekday() < 5 and curr.isoformat() not in holiday_dates:
                        added += 1

                due_dates.append(pd.Timestamp(curr))

            df[output_col] = due_dates

        # MODE 3: FLAG CALENDAR (WEEKENDS & REGIONAL HOLIDAYS)
        elif mode == "flag_calendar":
            is_weekend = []
            is_holiday = []
            holiday_col = []
            is_business_day = []

            for s_val in start_dt:
                if pd.isna(s_val):
                    is_weekend.append(False)
                    is_holiday.append(False)
                    holiday_col.append(None)
                    is_business_day.append(False)
                    continue

                d_iso = s_val.date().isoformat()
                wknd = s_val.weekday() >= 5
                hday = d_iso in holiday_dates
                h_name = holiday_names.get(d_iso)

                is_weekend.append(bool(wknd))
                is_holiday.append(bool(hday))
                holiday_col.append(h_name)
                is_business_day.append(bool(not wknd and not hday))

            df[f"{start_col}_is_weekend"] = is_weekend
            df[f"{start_col}_is_holiday"] = is_holiday
            df[f"{start_col}_holiday_name"] = holiday_col
            df[f"{start_col}_is_business_day"] = is_business_day
            metrics["holidays_detected"] = sum(is_holiday)
            metrics["business_days_detected"] = sum(is_business_day)

        output_summary = {
            "title": "Business Day & Holiday Calculator",
            "message": f"Processed business day calculations in mode '{mode}' with calendar '{calendar}'.",
            "metrics": metrics
        }

        return {
            "dataframe": df,
            "metrics": metrics,
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        mode = repr(config.get("mode", "business_days_between"))
        cal = repr(config.get("calendar", "US"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Mode: {mode}, Calendar: {cal}\n"
            f"# Calculates business days and checks regional holidays\n"
        )
