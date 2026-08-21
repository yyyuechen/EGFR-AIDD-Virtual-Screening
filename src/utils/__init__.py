"""Shared helpers for the EGFR AIDD project.

The project intentionally keeps helper code minimal.  Anything more complex
than a few small functions should live in the milestone script that uses it.
"""

from __future__ import annotations

import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert NaN/Inf to None so JSON output stays RFC 8259 compliant."""
    if isinstance(value, float) or (
        hasattr(value, "dtype") and hasattr(value, "item")
    ):
        try:
            if not math.isfinite(float(value)):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    return value
