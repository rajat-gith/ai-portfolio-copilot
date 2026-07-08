"""Small, dependency-free helpers for turning raw CMS payloads into
predictable Python shapes. Shared by every section builder so parsing
logic only lives in one place.
"""
from typing import Any, Dict, List


def normalize_list_response(resp: Any) -> List[Dict]:
    """
    Normalize CMS responses into a list of dicts.

    Supports:
    - [{...}, {...}]
    - {"data": [{...}, {...}]}
    - {"data": {...}}
    - {...}
    """
    if resp is None:
        return []

    if isinstance(resp, list):
        return resp

    if isinstance(resp, dict):
        if "data" in resp:
            data = resp["data"]
            if isinstance(data, list):
                return data
            if data is not None:
                return [data]
            return []
        # Fallback: treat single dict as one item
        return [resp]

    return []


def safe_get(data: Dict, key: str, default: str = "N/A") -> str:
    """Safely extract a value from a dict, always returning a string."""
    value = data.get(key, default)
    return str(value) if value is not None else default


def safe_list(data: Dict, key: str, default: str = "N/A") -> str:
    """Safely extract a list-like value and render it as a comma-separated string."""
    value = data.get(key)
    if value is None:
        return default
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def truncate_text(text: str, limit: int = 1200) -> str:
    """Truncate long free-text fields to keep embedded chunks reasonably sized."""
    if not isinstance(text, str):
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def format_duration(start: str, end_raw: Any, is_ongoing: bool) -> str:
    """Render a start/end pair as 'start - end', collapsing ongoing ranges to 'Present'."""
    end = "Present" if is_ongoing else (str(end_raw) if end_raw else "N/A")
    return f"{start} - {end}"
