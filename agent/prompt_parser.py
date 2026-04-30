import re


def extract_sku_from_prompt(prompt: str, valid_skus: set[str]) -> str | None:
    """Find the first alphanumeric token in `prompt` that matches a known StockCode."""
    tokens = re.findall(r"[A-Za-z0-9]+", str(prompt).upper())
    valid_upper = {sku.upper(): sku for sku in valid_skus}
    for tok in tokens:
        if tok in valid_upper:
            return valid_upper[tok]
    return None


def extract_horizon_from_prompt(prompt: str, default: int = 12) -> int:
    """Match patterns like 'next 8 weeks', 'week 5', '4w'. Falls back to default."""
    p = str(prompt).lower()
    m = re.search(r"(?:next|in)\s+(\d{1,2})\s*w", p) or re.search(r"\b(\d{1,2})\s*w(?:eek)?s?\b", p)
    if m:
        return int(m.group(1))
    return default
