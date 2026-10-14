import re
import hashlib
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Set
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort

# ----------------------------------------------------------------------
# ZERO-DEPENDENCY PII & SENSITIVE ENTITY DETECTION & MASKING ENGINE
# Provides fast, deterministic, offline entity detection and redaction
# ----------------------------------------------------------------------

# Common honorific prefixes
HONORIFICS = r"(?:Mr\.|Mrs\.|Ms\.|Miss|Dr\.|Prof\.|Sir|Madam|Lord|Lady|Hon\.|Rev\.)"

# Curated high-frequency first and last names for named entity recognition
POPULAR_NAMES: Set[str] = {
    # First names (Male & Female)
    "james", "john", "robert", "michael", "william", "david", "richard", "joseph", "thomas", "charles",
    "christopher", "daniel", "matthew", "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
    "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", "jessica", "sarah", "karen",
    "lisa", "nancy", "betty", "margaret", "sandra", "ashley", "kimberly", "emily", "donna", "michelle",
    "alexander", "alex", "alice", "brian", "kevin", "george", "edward", "brian", "ronald", "timothy",
    "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric", "jonathan", "stephen", "larry",
    "justin", "scott", "brandon", "benjamin", "samuel", "gregory", "frank", "alexander", "raymond", "patrick",
    "jack", "dennis", "jerry", "tyler", "aaron", "jose", "adam", "henry", "nathan", "douglas", "zachary",
    "peter", "kyle", "walter", "ethan", "jeremy", "harold", "keith", "christian", "roger", "noah", "gerald",
    "carl", "terry", "sean", "austin", "arthur", "lawrence", "jesse", "dylan", "bryan", "joe", "jordan",
    "billy", "bruce", "albert", "willie", "gabriel", "logan", "alan", "juan", "wayne", "roy", "ralph",
    "rachel", "amanda", "laura", "sarah", "stephanie", "rebecca", "sharon", "cynthia", "kathleen", "amy",
    "shirley", "angela", "helen", "anna", "brenda", "pamela", "nicole", "samantha", "katherine", "emma",
    "christine", "debra", "rachel", "carolyn", "janet", "catherine", "maria", "heather", "diane", "olivia",
    "julie", "joyce", "victoria", "ruth", "virginia", "lauren", "kelly", "christina", "joan", "evelyn",
    "judith", "andrea", "megan", "cheryl", "hannah", "jacqueline", "martha", "gloria", "teresa", "ann",
    "sara", "madison", "frances", "kathryn", "janice", "jean", "abigail", "alice", "julia", "judy",
    "sophia", "grace", "denise", "amber", "doris", "marilyn", "danielle", "beverly", "diana", "brittany",
    "theresa", "charlotte", "marie", "kayla", "alexis", "lori", "ahmed", "mohamed", "ali", "fatima",
    "priya", "rahul", "arjun", "sneha", "vikram", "rohit", "ananya", "amit", "suresh", "rajesh",
    "chen", "wei", "li", "zhang", "wang", "liu", "yang", "huang", "wu", "zhao",

    # Common last names
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis", "rodriguez", "martinez",
    "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas", "taylor", "moore", "jackson", "martin",
    "lee", "perez", "thompson", "white", "harris", "sanchez", "clark", "ramirez", "lewis", "robinson",
    "walker", "young", "allen", "king", "wright", "scott", "torres", "nguyen", "hill", "flores", "green",
    "adams", "nelson", "baker", "hall", "rivera", "campbell", "mitchell", "carter", "roberts", "gomez",
    "phillips", "evans", "turner", "diaz", "parker", "cruz", "edwards", "collins", "reyes", "stewart",
    "morris", "morales", "murphy", "cook", "rogers", "gutierrez", "ortiz", "morgan", "cooper", "peterson",
    "bailey", "reed", "kelly", "howard", "ramos", "kim", "cox", "ward", "richardson", "watson", "brooks",
    "chavez", "wood", "james", "bennett", "gray", "mendoza", "ruiz", "hughes", "price", "alvarez",
    "castillo", "sanders", "patel", "myers", "long", "ross", "foster", "jimenez", "powell", "jenkins",
    "perry", "russell", "sullivan", "bell", "coleman", "butler", "henderson", "barnes", "gonzales", "fisher",
    "sharma", "gupta", "verma", "kumar", "singh", "khan", "shaikh", "reddy", "rao", "nair"
}

# Compiled regex patterns
REGEX_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Phone patterns supporting +1-800-555-0199, (555) 123-4567, 555-123-4567, +91 98765 43210, etc.
REGEX_PHONE = re.compile(
    r"(?:\+?\d{1,4}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b"
)

# Credit Card pattern (13 to 19 digits formatted with dashes, spaces, or raw)
REGEX_CARD_CANDIDATE = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b|\b\d{13,19}\b"
)

# US SSN pattern
REGEX_SSN = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}\b"
)

# IPv4 and IPv6 patterns
REGEX_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
REGEX_IPV6 = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
)

# Honorific person name pattern (e.g. Mr. John Doe, Dr. Alice Smith)
REGEX_HONORIFIC_NAME = re.compile(
    rf"\b{HONORIFICS}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)

# Explicit labeled name pattern (e.g. Name: John Doe, Patient: Jane Doe)
REGEX_LABELED_NAME = re.compile(
    r"(?i)\b(?:name|patient|customer|client|caller|employee|user)\s*[:=-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

# Date of Birth pattern with explicit context prefix
REGEX_DOB = re.compile(
    r"(?i)\b(?:dob|d\.o\.b\.|date of birth|born|birth date)\s*[:=-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)

# IBAN and Account Number patterns
REGEX_IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b"
)
REGEX_ACCOUNT_NUM = re.compile(
    r"(?i)\b(?:acct|account|acc)\s*#?\s*[:=-]?\s*(\d{6,17})\b"
)


def is_luhn_valid(card_str: str) -> bool:
    """Validates credit/debit card numbers using the standard Luhn algorithm."""
    digits = [int(c) for c in card_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def detect_names_in_text(text: str) -> List[Tuple[int, int, str]]:
    """
    Detects person names in text using honorifics, explicit label cues,
    and two-word sequences matching known popular names.
    Returns list of (start, end, matched_text).
    """
    spans: List[Tuple[int, int, str]] = []

    # 1. Honorific + Names: Mr. John Doe
    for m in REGEX_HONORIFIC_NAME.finditer(text):
        spans.append((m.start(), m.end(), m.group()))

    # 2. Labeled Names: Name: Alice Smith
    for m in REGEX_LABELED_NAME.finditer(text):
        name_group = m.group(1)
        start_idx = m.start(1)
        end_idx = m.end(1)
        spans.append((start_idx, end_idx, name_group))

    # 3. Two-word capitalized pairs matching popular first & last names
    words = list(re.finditer(r"\b[A-Z][a-z]{1,20}\b", text))
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        # Ensure words are adjacent separated only by a single whitespace
        if w1.end() + 1 == w2.start() and text[w1.end()] == " ":
            token1 = w1.group().lower()
            token2 = w2.group().lower()
            # If at least one word is in our curated names dictionary
            if (token1 in POPULAR_NAMES or token2 in POPULAR_NAMES) and (len(token1) > 2 and len(token2) > 2):
                span_start = w1.start()
                span_end = w2.end()
                spans.append((span_start, span_end, text[span_start:span_end]))

    return spans


def format_masked_token(
    entity_type: str,
    raw_val: str,
    mask_style: str,
    custom_token: str = "[REDACTED]"
) -> str:
    """Formats the replacement string according to the selected masking style."""
    raw_clean = raw_val.strip()

    if mask_style == "Full Redaction":
        return custom_token or "[REDACTED]"

    elif mask_style == "Asterisks Obfuscation":
        # Preserve spaces and hyphens if present, mask characters with *
        masked_chars = [c if c in {"-", " ", ".", "@", "(", ")", "/"} else "*" for c in raw_clean]
        return "".join(masked_chars)

    elif mask_style == "Pseudonymization (Hash)":
        # Consistent deterministic hash token across rows
        token_hash = hashlib.md5(raw_clean.lower().encode("utf-8")).hexdigest()[:8]
        return f"[{entity_type}_{token_hash}]"

    elif mask_style == "Partial Masking (Last 4 Visible)":
        digits_or_chars = [c for c in raw_clean if c.isalnum()]
        if len(digits_or_chars) <= 4:
            return "*" * len(raw_clean)
        # Keep the last 4 alphanumeric characters visible, mask earlier ones
        seen = 0
        res = []
        for c in reversed(raw_clean):
            if c.isalnum():
                seen += 1
                if seen <= 4:
                    res.append(c)
                else:
                    res.append("*")
            else:
                res.append(c)
        return "".join(reversed(res))

    else:
        # Default: "Entity Tag Replacement"
        return f"[{entity_type}]"


def mask_text_pii(
    text: str,
    mask_style: str = "Entity Tag Replacement",
    entity_scope: str = "All Entities",
    custom_token: str = "[REDACTED]",
    custom_regex_str: str = ""
) -> Tuple[str, Dict[str, int]]:
    """
    Scans a single text string, detects sensitive entities, and applies masking.
    Returns (masked_text, count_by_entity_type).
    """
    if not text or not isinstance(text, str):
        return ("" if text is None else str(text), {})

    counts: Dict[str, int] = Counter()
    candidates: List[Tuple[int, int, int, str, str]] = []  # (start, end, priority, entity_type, raw_val)

    # 1. Custom Regex (Priority 1 - highest user priority)
    if custom_regex_str and custom_regex_str.strip():
        try:
            cregex = re.compile(custom_regex_str.strip())
            for m in cregex.finditer(text):
                candidates.append((m.start(), m.end(), 1, "CUSTOM", m.group()))
        except Exception:
            pass

    if entity_scope != "Custom Regex Only":
        include_all = entity_scope == "All Entities"
        include_personal = include_all or "Personal" in entity_scope
        include_financial = include_all or "Financial" in entity_scope
        include_contact = include_all or "Emails" in entity_scope or include_personal

        # 2. Credit Cards (Priority 2)
        if include_all or include_financial:
            for m in REGEX_CARD_CANDIDATE.finditer(text):
                card_num = m.group()
                digit_count = sum(c.isdigit() for c in card_num)
                if 13 <= digit_count <= 19:
                    # Require Luhn check or explicit dashed card format
                    if is_luhn_valid(card_num) or ("-" in card_num and digit_count == 16):
                        candidates.append((m.start(), m.end(), 2, "CREDIT_CARD", card_num))

        # 3. SSN (Priority 3)
        if include_all or include_financial or include_personal:
            for m in REGEX_SSN.finditer(text):
                candidates.append((m.start(), m.end(), 3, "SSN", m.group()))

        # 4. Email (Priority 4)
        if include_all or include_contact:
            for m in REGEX_EMAIL.finditer(text):
                candidates.append((m.start(), m.end(), 4, "EMAIL", m.group()))

        # 5. IP Addresses (Priority 5)
        if include_all:
            for m in REGEX_IPV4.finditer(text):
                candidates.append((m.start(), m.end(), 5, "IP_ADDRESS", m.group()))
            for m in REGEX_IPV6.finditer(text):
                candidates.append((m.start(), m.end(), 5, "IP_ADDRESS", m.group()))

        # 6. Phone Numbers (Priority 6)
        if include_all or include_contact:
            for m in REGEX_PHONE.finditer(text):
                p_text = m.group()
                digits = sum(c.isdigit() for c in p_text)
                # Ensure plausible phone length (7 to 15 digits) and avoids plain numbers like '2024'
                if 7 <= digits <= 15 and ("-" in p_text or "(" in p_text or "+" in p_text or " " in p_text or digits >= 10):
                    candidates.append((m.start(), m.end(), 6, "PHONE", p_text))

        # 7. DOB & Financial (Priority 7)
        if include_all or include_personal:
            for m in REGEX_DOB.finditer(text):
                # Mask the date value itself
                candidates.append((m.start(1), m.end(1), 7, "DOB", m.group(1)))

        if include_all or include_financial:
            for m in REGEX_IBAN.finditer(text):
                candidates.append((m.start(), m.end(), 7, "IBAN", m.group()))
            for m in REGEX_ACCOUNT_NUM.finditer(text):
                candidates.append((m.start(1), m.end(1), 7, "ACCOUNT_NUM", m.group(1)))

        # 8. Person Names (Priority 8)
        if include_all or include_personal:
            for start, end, name_val in detect_names_in_text(text):
                candidates.append((start, end, 8, "NAME", name_val))

    if not candidates:
        return (text, {})

    # Sort candidates by:
    # 1. start index ascending
    # 2. priority ascending (lower number = higher priority)
    # 3. length descending
    candidates.sort(key=lambda x: (x[0], x[2], -(x[1] - x[0])))

    # Resolve overlapping spans by taking the highest-priority non-overlapping spans
    accepted_spans: List[Tuple[int, int, str, str]] = []
    last_end = -1

    for start, end, priority, entity_type, raw_val in candidates:
        if start >= last_end:
            accepted_spans.append((start, end, entity_type, raw_val))
            last_end = end

    # Replace from right-to-left so original start/end indices remain intact
    accepted_spans.sort(key=lambda x: x[0], reverse=True)
    chars = list(text)

    for start, end, entity_type, raw_val in accepted_spans:
        replacement = format_masked_token(entity_type, raw_val, mask_style, custom_token)
        chars[start:end] = list(replacement)
        counts[entity_type] += 1

    masked_text = "".join(chars)
    return (masked_text, dict(counts))


# ----------------------------------------------------------------------
# PII / ENTITY MASKING RECIPE CLASS
# ----------------------------------------------------------------------

class PIIEntityMaskingRecipe(BaseRecipe):
    """
    NLP / Text PII and Entity Masking Recipe.
    Detects personal identifiers (names, emails, phone numbers, cards, SSN, IP addresses)
    and masks them with configurable tags, tokens, asterisks, hashes, or partial redactions.
    """

    recipe_id = "pii_entity_masking"
    name = "PII / Entity Masking"
    version = "1.0.0"
    category = "nlp"
    group = "Data Preparation & Transformation"
    subgroup = "NLP/Text"
    description = "Detects names, emails, phone numbers and other entities in text and masks them."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="dataframe",
            label="Input DataFrame",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing text columns with personal identifiers to mask"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Masked DataFrame",
            type="dataframe",
            description="Dataset with sensitive entities and PII masked"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Target Text Columns",
                    "input_type": "column_select",
                    "description": "Text columns to scan and mask sensitive entities in. If left empty, all text/string columns will be processed."
                },
                "mask_style": {
                    "type": "select",
                    "input_type": "dropdown",
                    "title": "Masking Style",
                    "enum": [
                        "Entity Tag Replacement",
                        "Full Redaction",
                        "Asterisks Obfuscation",
                        "Pseudonymization (Hash)",
                        "Partial Masking (Last 4 Visible)"
                    ],
                    "default": "Entity Tag Replacement",
                    "description": "Strategy used to mask detected sensitive entities."
                },
                "entity_scope": {
                    "type": "select",
                    "input_type": "dropdown",
                    "title": "Entity Scope",
                    "enum": [
                        "All Entities",
                        "Personal Data (Names, Emails, Phones, DOB)",
                        "Financial & Identity (Cards, SSN, IBAN)",
                        "Emails & Phones Only",
                        "Custom Regex Only"
                    ],
                    "default": "All Entities",
                    "description": "Scope of sensitive entities to detect and mask."
                },
                "custom_mask_token": {
                    "type": "text",
                    "input_type": "text",
                    "title": "Custom Redaction Token",
                    "default": "[REDACTED]",
                    "description": "Replacement string used when 'Full Redaction' masking style is selected."
                },
                "custom_regex": {
                    "type": "text",
                    "input_type": "text",
                    "title": "Custom Regex Pattern",
                    "default": "",
                    "description": "Optional custom regular expression pattern to detect and mask specialized domain secrets (e.g. API keys, internal employee IDs)."
                },
                "append_columns": {
                    "type": "boolean",
                    "input_type": "boolean",
                    "title": "Append Output Columns",
                    "default": True,
                    "description": "When true, appends new masked columns alongside existing data. When false, replaces original text columns."
                },
                "output_suffix": {
                    "type": "text",
                    "input_type": "text",
                    "title": "Output Column Suffix",
                    "default": "_masked",
                    "description": "Suffix for newly appended masked columns (e.g. '_masked')."
                }
            }
        }

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None or not isinstance(df, pd.DataFrame):
            df = inputs.get("left_dataframe") or inputs.get("input")
            if df is None or not isinstance(df, pd.DataFrame):
                return {
                    "dataframe": pd.DataFrame(),
                    "output_summary": {"error": "No valid input DataFrame provided for PII / Entity Masking."}
                }

        if df.empty:
            return {
                "dataframe": df.copy(),
                "metrics": {"total_rows": 0, "total_entities_masked": 0, "masked_columns": []},
                "output_summary": {"message": "Input DataFrame is empty. No PII masking performed."}
            }

        out_df = df.copy()

        # Parse configuration
        mask_style = str(config.get("mask_style") or "Entity Tag Replacement")
        entity_scope = str(config.get("entity_scope") or "All Entities")
        custom_token = str(config.get("custom_mask_token") or "[REDACTED]")
        custom_regex = str(config.get("custom_regex") or "").strip()
        append_cols = bool(config.get("append_columns", True))
        suffix = str(config.get("output_suffix", "_masked")).strip() or "_masked"

        # Determine target columns
        configured_cols = config.get("columns")
        if configured_cols and isinstance(configured_cols, list):
            target_cols = [c for c in configured_cols if c in out_df.columns]
        else:
            # Auto-detect object / string columns
            target_cols = [
                c for c in out_df.columns
                if out_df[c].dtype == object or str(out_df[c].dtype).startswith("string")
            ]

        if not target_cols:
            return {
                "dataframe": out_df,
                "metrics": {"total_rows": len(out_df), "total_entities_masked": 0, "masked_columns": []},
                "output_summary": {"message": "No text columns found or specified for PII masking."}
            }

        overall_counts: Counter = Counter()
        masked_columns: List[str] = []
        samples_preview: Dict[str, List[Dict[str, str]]] = {}

        for col in target_cols:
            col_series = out_df[col]
            masked_series = []
            col_samples = []

            for val in col_series:
                if pd.isna(val):
                    masked_series.append(val)
                    continue

                text_val = str(val)
                masked_val, entity_counts = mask_text_pii(
                    text_val,
                    mask_style=mask_style,
                    entity_scope=entity_scope,
                    custom_token=custom_token,
                    custom_regex_str=custom_regex
                )
                masked_series.append(masked_val)

                for ent, cnt in entity_counts.items():
                    overall_counts[ent] += cnt

                if entity_counts and len(col_samples) < 3:
                    col_samples.append({
                        "original": text_val,
                        "masked": masked_val
                    })

            dest_col = f"{col}{suffix}" if append_cols else col
            out_df[dest_col] = masked_series
            masked_columns.append(dest_col)
            if col_samples:
                samples_preview[col] = col_samples

        total_entities = sum(overall_counts.values())

        metrics = {
            "total_rows_processed": int(len(out_df)),
            "total_columns_processed": int(len(target_cols)),
            "total_entities_masked": int(total_entities),
            "entities_by_type": dict(overall_counts),
            "masked_columns": masked_columns
        }

        output_summary = {
            "status": "success",
            "message": f"Masked {total_entities} sensitive entities across {len(target_cols)} column(s).",
            "mask_style": mask_style,
            "entity_scope": entity_scope,
            "entities_detected": dict(overall_counts),
            "sample_preview": samples_preview
        }

        return {
            "dataframe": out_df,
            "metrics": metrics,
            "output_summary": output_summary
        }
