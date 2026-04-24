"""Built-in computational tools: date/time, arithmetic, unit conversion."""
from __future__ import annotations

import datetime
import math
import re


_MATH_SAFE = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "pow": pow,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "ceil": math.ceil,
    "floor": math.floor,
}

_UNIT_TABLE: dict[tuple[str, str], float] = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("kg", "lbs"): 2.20462,
    ("lbs", "kg"): 0.453592,
    ("celsius", "fahrenheit"): None,
    ("fahrenheit", "celsius"): None,
}


class Tools:
    def execute(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ("date", "time", "today", "now", "current day")):
            return self._datetime()
        if any(w in q for w in ("convert", "in miles", "in km", "in lbs", "in kg", "fahrenheit", "celsius")):
            return self._unit_convert(query)
        return self._calculate(query)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _datetime(self) -> str:
        now = datetime.datetime.now()
        return (
            f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
            f"Current time: {now.strftime('%H:%M:%S %Z')}"
        )

    def _calculate(self, query: str) -> str:
        expr = re.sub(r"[^0-9+\-*/().\s]", " ", query).strip()
        expr = re.sub(r"\s+", "", expr)
        if not expr:
            return f"No mathematical expression found in: {query!r}"
        try:
            result = eval(expr, _MATH_SAFE)  # noqa: S307 — sandboxed namespace
            return f"{expr} = {result}"
        except Exception as exc:
            return f"Could not evaluate '{expr}': {exc}"

    def _unit_convert(self, query: str) -> str:
        q = query.lower()
        # Temperature special cases
        m = re.search(r"([\d.]+)\s*°?c(?:elsius)?\s+(?:to|in)\s+f(?:ahrenheit)?", q)
        if m:
            c = float(m.group(1))
            return f"{c}°C = {c * 9/5 + 32:.2f}°F"
        m = re.search(r"([\d.]+)\s*°?f(?:ahrenheit)?\s+(?:to|in)\s+c(?:elsius)?", q)
        if m:
            f = float(m.group(1))
            return f"{f}°F = {(f - 32) * 5/9:.2f}°C"
        # Generic factor-based conversions
        for (src, dst), factor in _UNIT_TABLE.items():
            if factor is None:
                continue
            pattern = rf"([\d.]+)\s*{re.escape(src)}\s+(?:to|in)\s+{re.escape(dst)}"
            m = re.search(pattern, q)
            if m:
                val = float(m.group(1))
                return f"{val} {src} = {val * factor:.4f} {dst}"
        return self._calculate(query)
