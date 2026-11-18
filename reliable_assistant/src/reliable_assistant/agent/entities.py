"""Deterministic identifier auto-repair (§1.2.6): SKU99 -> SKU-099, lot 44 -> LOT-00044."""
import re

_SKU_I = re.compile(r"SKU[\s_-]?(\d{1,6})", re.I)
_LOT_I = re.compile(r"LOT[\s_-]?(\d{1,5})", re.I)
_SKU_EMB = re.compile(r"\bSKU[\s_-]?(\d{1,6})\b")   # embedded in prose: uppercase prefix only
_LOT_EMB = re.compile(r"\bLOT[\s_-]?(\d{1,5})\b")


def _sku(m): return f"SKU-{m.group(1).zfill(3)}"
def _lot(m): return f"LOT-{m.group(1).zfill(5)}"


def repair_str(s: str) -> str:
    """Whole-value identifiers repair case-insensitively; embedded ones only when written in uppercase
    (so prose like 'a lot 5 times' is never rewritten)."""
    t = s.strip()
    for full, fn in ((_SKU_I, _sku), (_LOT_I, _lot)):
        if full.fullmatch(t):
            return full.sub(fn, t)
    return _LOT_EMB.sub(_lot, _SKU_EMB.sub(_sku, s))


def repair_value(obj):
    if isinstance(obj, str):
        return repair_str(obj)
    if isinstance(obj, dict):
        return {k: repair_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [repair_value(v) for v in obj]
    return obj


def extract_entities(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if m := re.search(r"\bSKU[\s_-]?(\d{1,6})\b", text, re.I):
        out["sku"] = _sku(m)
    if m := re.search(r"\bLOT[\s_-]?(\d{1,5})\b", text, re.I):
        out["lot"] = _lot(m)
    return out
