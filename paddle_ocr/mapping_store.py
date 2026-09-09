"""Persist a fixed name→value-offset mapping and apply it to any OCR list."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
MAPPING_FILE = DATA_DIR / "saved_mapping.json"

_AR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ٱ": "ا",
    }
)


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.translate(_AR_MAP)
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_mapping() -> dict[str, Any]:
    ensure_data_dir()
    if not MAPPING_FILE.exists():
        return {"mapping": [], "updated_at": None, "mode": "fixed"}
    try:
        data = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"mapping": [], "updated_at": None, "mode": "fixed"}
        data.setdefault("mapping", [])
        data.setdefault("mode", "fixed")
        return data
    except Exception:  # noqa: BLE001
        return {"mapping": [], "updated_at": None, "mode": "fixed"}


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique_ints(values: list[Any]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for raw in values or []:
        vi = _as_int(raw)
        if vi is None or vi in seen:
            continue
        seen.add(vi)
        out.append(vi)
    return out


def _compute_offsets(
    name_index: int | None,
    value_indices: list[int],
    explicit_offsets: list[Any] | None,
) -> list[int]:
    if explicit_offsets is not None and len(list(explicit_offsets)) > 0:
        return _unique_ints(explicit_offsets)
    if name_index is None:
        return []
    return [vi - int(name_index) for vi in value_indices]


def save_mapping(
    mapping: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a fixed template: field name + relative value offsets."""
    ensure_data_dir()

    cleaned: list[dict[str, Any]] = []
    for row in mapping or []:
        name = str(row.get("name") or "").strip() or None
        name_index = _as_int(row.get("name_index"))
        value_indices = _unique_ints(row.get("value_indices") or [])
        offsets = _compute_offsets(
            name_index,
            value_indices,
            row.get("value_offsets"),
        )

        if not name and name_index is None:
            continue

        cleaned.append(
            {
                # Primary keys for any OCR run:
                "name": name,
                "value_offsets": offsets,
                # Reference snapshot from the document used to teach the template:
                "name_index": name_index,
                "value_indices": value_indices,
                "values": [str(v) for v in (row.get("values") or []) if v is not None],
            }
        )

    payload = {
        "mode": "fixed",
        "mapping": cleaned,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
    }
    MAPPING_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _score_name_match(candidate: str, target: str) -> float:
    cand = normalize_text(candidate)
    tgt = normalize_text(target)
    if not cand or not tgt:
        return 0.0
    if cand == tgt:
        return 100.0
    if cand.startswith(tgt) or tgt.startswith(cand):
        return 80.0 - abs(len(cand) - len(tgt)) * 0.1
    if tgt in cand:
        return 60.0 - (len(cand) - len(tgt)) * 0.05
    if cand in tgt:
        return 50.0 - (len(tgt) - len(cand)) * 0.05
    return 0.0


def _find_name_index(
    ocr_by_index: dict[int, dict[str, Any]],
    row: dict[str, Any],
) -> int | None:
    """Find field name in current OCR by saved name text (index is only a hint)."""
    name = row.get("name")
    if not name:
        return None

    best_idx: int | None = None
    best_score = 0.0
    for idx, item in sorted(ocr_by_index.items()):
        score = _score_name_match(str(item.get("text") or ""), str(name))
        if score > best_score:
            best_score = score
            best_idx = idx

    # Accept only reasonably close label matches.
    if best_score >= 50.0:
        return best_idx
    return None


def _resolve_value_indices(
    ocr_by_index: dict[int, dict[str, Any]],
    row: dict[str, Any],
    resolved_name_index: int | None,
) -> list[int]:
    """Resolve values from fixed offsets relative to the found name."""
    if resolved_name_index is None:
        return []

    offsets = row.get("value_offsets")
    if offsets is None:
        # Backward-compatible: derive from old absolute indices.
        old_name = _as_int(row.get("name_index"))
        old_values = _unique_ints(row.get("value_indices") or [])
        if old_name is not None and old_values:
            offsets = [vi - old_name for vi in old_values]
        else:
            offsets = []

    offsets = _unique_ints(offsets)
    if not offsets:
        return []

    resolved: list[int] = []
    for off in offsets:
        vi = int(resolved_name_index) + int(off)
        if vi in ocr_by_index:
            resolved.append(vi)
    return resolved


def apply_mapping(
    ocr_list: list[dict[str, Any]],
    mapping: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the fixed mapping template onto any OCR list."""
    stored = load_mapping() if mapping is None else {"mapping": mapping, "mode": "fixed"}
    rows = stored.get("mapping") or []

    ocr_by_index: dict[int, dict[str, Any]] = {}
    for item in ocr_list or []:
        idx = _as_int(item.get("index") if isinstance(item, dict) else None)
        if idx is None:
            continue
        ocr_by_index[idx] = item

    results: list[dict[str, Any]] = []
    for row in rows:
        name_index = _find_name_index(ocr_by_index, row)
        name_text = None
        page = None
        if name_index is not None and name_index in ocr_by_index:
            name_text = ocr_by_index[name_index].get("text")
            page = ocr_by_index[name_index].get("page")

        value_indices = _resolve_value_indices(ocr_by_index, row, name_index)
        value_items = []
        for vi in value_indices:
            it = ocr_by_index[vi]
            value_items.append(
                {
                    "value_index": vi,
                    "value": it.get("text"),
                    "page": it.get("page"),
                    "offset": vi - name_index if name_index is not None else None,
                }
            )

        results.append(
            {
                "name": name_text or row.get("name"),
                "template_name": row.get("name"),
                "name_index": name_index,
                "page": page,
                "value_offsets": row.get("value_offsets")
                or [
                    vi - int(row["name_index"])
                    for vi in (row.get("value_indices") or [])
                    if row.get("name_index") is not None
                ],
                "value_indices": value_indices,
                "values": [v["value"] for v in value_items],
                "value_items": value_items,
                "found_name": name_index is not None,
                "has_values": len(value_items) > 0,
            }
        )

    return {
        "mapping_applied": True,
        "mode": "fixed",
        "mapping_count": len(rows),
        "results": results,
        "updated_at": stored.get("updated_at"),
    }
