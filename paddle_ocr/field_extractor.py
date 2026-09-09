"""Extract field values using spatial zoning from OCR bounding boxes.

Zoning (primary):
  For a label box, look for value blocks in nearby zones only:
    1) same horizontal band (left of label — Arabic RTL forms)
    2) same horizontal band (right of label — LTR)
    3) directly below the label
  Never take distant blocks from other parts of the page.

Fallback:
  Text-order between nearby labels / same-next line (if no boxes).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

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

_SEP_CHARS = ":：=–—-|•·*"
_NEAR_LABEL_CHARS = 160
_LOCAL_VALUE_CHARS = 80
_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_\/]{0,24}")
_CODE_LABEL_HINTS = ("كود", "رمز", "رقم", "code", "id", "no", "number")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.translate(_AR_MAP)
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


@dataclass
class FieldHit:
    key: str
    label: str
    value: str
    found: bool
    confidence: float
    source_page: int | None
    message: str
    next_label: str | None = None
    zone: str | None = None
    name_index: int | None = None  # index of field name in ocr_list
    value_index: int | None = None  # index of field value in ocr_list

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    fields: list[FieldHit] = field(default_factory=list)
    found_count: int = 0
    requested_count: int = 0
    ocr_list: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ocr_list": self.ocr_list,
            "fields": [f.to_dict() for f in self.fields],
            "pairs": [
                {
                    "name": f.label,
                    "value": f.value,
                    "name_index": f.name_index,
                    "value_index": f.value_index,
                    "found": f.found,
                }
                for f in self.fields
            ],
            "found_count": self.found_count,
            "requested_count": self.requested_count,
            "valid_count": self.found_count,
            "missing_required": [f.label for f in self.fields if not f.found],
            "all_required_ok": self.found_count == self.requested_count
            and self.requested_count > 0,
        }


@dataclass
class _Block:
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    score: float = 0.0
    index: int = -1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def w(self) -> float:
        return max(self.x1 - self.x0, 1.0)

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 1.0)


class FieldExtractor:
    def __init__(self) -> None:
        self.document_type = "حقول مخصصة (Zoning)"
        self.schema: list[dict[str, Any]] = []

    def extract_from_pages(
        self,
        pages: list[dict[str, Any]],
        labels: list[str] | None = None,
    ) -> ExtractionResult:
        requested = self._normalize_labels(labels or [])
        blocks = self._collect_blocks(pages)
        ocr_list = [
            {
                "index": b.index,
                "text": b.text,
                "page": b.page,
                "bbox": [b.x0, b.y0, b.x1, b.y1],
                "score": b.score,
            }
            for b in blocks
        ]

        if not requested:
            return ExtractionResult(
                fields=[],
                found_count=0,
                requested_count=0,
                ocr_list=ocr_list,
            )

        label_norm = {normalize_text(l): l for l in requested}

        hits: list[FieldHit] = []
        for label in requested:
            hit = self._extract_with_zoning(label, blocks, label_norm)
            if not hit.found:
                fallback = self._extract_text_fallback(label, pages, requested)
                if fallback.found:
                    # Try to resolve indices from ocr_list for fallback values.
                    fallback.name_index = self._find_text_index(blocks, label, prefer="label")
                    fallback.value_index = self._find_text_index(
                        blocks, fallback.value, prefer="value", after_index=fallback.name_index
                    )
                    hit = fallback
            hits.append(hit)

        return ExtractionResult(
            fields=hits,
            found_count=sum(1 for h in hits if h.found),
            requested_count=len(hits),
            ocr_list=ocr_list,
        )

    def _extract_with_zoning(
        self,
        label: str,
        blocks: list[_Block],
        label_norm: dict[str, str],
    ) -> FieldHit:
        key = normalize_text(label)
        label_blocks = [
            b
            for b in blocks
            if self._label_match_quality(b.text, label) > 0
        ]
        if not label_blocks:
            return FieldHit(
                key=key.replace(" ", "_")[:64],
                label=label,
                value="",
                found=False,
                confidence=0.0,
                source_page=None,
                message="لم يظهر اسم الحقل في النص المقروء",
                name_index=None,
                value_index=None,
            )

        # Best label box: strongest match, then top-most / left-most (label then value to its right).
        label_blocks.sort(
            key=lambda b: (
                -self._label_match_quality(b.text, label),
                b.page,
                b.y0,
                b.x0,
            )
        )
        lb = label_blocks[0]

        # Same block: "اسم الفرع: الدقى" / "اسم الفرع الدقى"
        same_glued = self._value_after_label_in_text(lb.text, label)
        if same_glued and not self._looks_like_field_label(same_glued, label_norm, key):
            val = self._finalize_value(label, same_glued)
            if val:
                return FieldHit(
                    key=key.replace(" ", "_")[:64],
                    label=label,
                    value=val,
                    found=True,
                    confidence=0.98,
                    source_page=lb.page,
                    message=f"Zoning: قيمة في نفس صندوق «{label}»",
                    zone="same-block",
                    name_index=lb.index,
                    value_index=lb.index,  # name+value in same OCR item
                )

        other_labels = {normalize_text(v) for k, v in label_norm.items() if k != key}
        candidates: list[tuple[float, str, str, _Block]] = []

        for b in blocks:
            if b.page != lb.page:
                continue
            if self._same_block(lb, b):
                continue
            if self._label_match_quality(b.text, label) > 0:
                continue
            if self._looks_like_field_label(b.text, label_norm, key):
                continue
            if any(self._text_is_label(b.text, ol) for ol in other_labels):
                continue

            zone, zone_score = self._zone_score(lb, b)
            if zone is None:
                continue
            # User rule: value is to the RIGHT of the label — prefer that strongly.
            if zone != "right-of-label" and zone != "below-label":
                continue

            val = self._finalize_value(label, b.text)
            if not val or self._looks_like_field_label(val, label_norm, key):
                continue

            # Prefer nearest block to the right (same row).
            dx = max(0.0, b.x0 - lb.x1) / max(lb.w, 1.0)
            dy = abs(b.cy - lb.cy) / max(lb.h, 1.0)
            score = zone_score - 0.08 * dx - 0.12 * dy + 0.02 * min(b.score, 1.0)
            candidates.append((score, zone, val, b))

        if not candidates:
            return FieldHit(
                key=key.replace(" ", "_")[:64],
                label=label,
                value="",
                found=False,
                confidence=0.0,
                source_page=lb.page,
                message="وُجدت التسمية لكن لا توجد قيمة على يمين الحقل",
                name_index=lb.index,
                value_index=None,
            )

        candidates.sort(key=lambda c: c[0], reverse=True)
        score, zone, value, value_block = candidates[0]
        return FieldHit(
            key=key.replace(" ", "_")[:64],
            label=label,
            value=value,
            found=True,
            confidence=round(min(max(score, 0.5), 0.99), 3),
            source_page=lb.page,
            message=f"Zoning: قيمة يمين «{label}» ({zone})",
            zone=zone,
            name_index=lb.index,
            value_index=value_block.index,
        )

    @staticmethod
    def _same_block(a: _Block, b: _Block) -> bool:
        return (
            a.page == b.page
            and abs(a.x0 - b.x0) < 1
            and abs(a.y0 - b.y0) < 1
            and a.text == b.text
        )

    def _label_match_quality(self, text: str, label: str) -> int:
        tn = normalize_text(text)
        ln = normalize_text(label)
        if not ln or not tn:
            return -1
        if tn == ln:
            return 100
        # Label with separator/value glued: still a label box.
        if tn.startswith(ln) and len(tn) <= len(ln) + 40:
            return 90
        if ln in tn and len(tn) <= len(ln) + 12:
            return 70
        return -1

    def _looks_like_field_label(
        self, text: str, label_norm: dict[str, str], current_key: str
    ) -> bool:
        t = normalize_text(text)
        if not t:
            return True
        for key, label in label_norm.items():
            if key == current_key:
                continue
            ln = normalize_text(label)
            if t == ln or t.startswith(ln) or ln.startswith(t) and len(t) >= 3:
                return True
            if ln in t and len(t) <= len(ln) + 8:
                return True
        # Generic bilingual label-ish tokens that often sit near branch name.
        hints = (
            "branch code",
            "branch name",
            "كود الفرع",
            "اسم الفرع",
            "كود",
            "code",
            "sector",
            "قطاع",
        )
        for h in hints:
            hn = normalize_text(h)
            if t == hn or t.startswith(hn + " ") or hn == t:
                # Allow if this IS the current label's own hint only when equal to current
                if hn == current_key or current_key.startswith(hn) or hn.startswith(current_key):
                    continue
                if hn in t and len(t) <= len(hn) + 10:
                    return True
        return False

    def _zone_score(self, label: _Block, cand: _Block) -> tuple[str | None, float]:
        """Value zones: prefer RIGHT of label (user layout)."""
        same_row = abs(cand.cy - label.cy) <= max(label.h, cand.h) * 1.05

        # Strict-ish right: candidate starts at/after label's right edge.
        right = cand.cx > label.cx and same_row and cand.x0 >= label.x0
        right_near = right and (cand.x0 - label.x1) <= max(label.w * 6.0, 220.0)
        if right_near or (same_row and cand.x0 >= label.x1 - label.w * 0.15 and cand.cx > label.cx):
            return "right-of-label", 0.97

        # Below as secondary (value under label).
        below = cand.y0 >= label.y1 - label.h * 0.35 and cand.cy > label.cy
        below_near = below and (cand.y0 - label.y1) <= label.h * 2.5
        x_close = abs(cand.cx - label.cx) <= label.w * 2.8
        if below_near and x_close:
            return "below-label", 0.7

        return None, 0.0

    def _extract_text_fallback(
        self,
        label: str,
        pages: list[dict[str, Any]],
        requested: list[str],
    ) -> FieldHit:
        """Previous text-order logic as fallback."""
        lines = self._index_lines(pages)
        full_text, page_at = self._build_corpus(lines)
        spans = self._find_label_spans(full_text, label)
        key = normalize_text(label)
        if not spans:
            return FieldHit(
                key=key.replace(" ", "_")[:64],
                label=label,
                value="",
                found=False,
                confidence=0.0,
                source_page=None,
                message="لم يظهر اسم الحقل في النص المقروء",
            )

        start, end = spans[0]
        page = page_at[start] if start < len(page_at) else 1

        # Cut at next requested label if nearby.
        next_start = None
        next_label = None
        for other in requested:
            if normalize_text(other) == key:
                continue
            for os_, oe_ in self._find_label_spans(full_text, other):
                if os_ >= end and (next_start is None or os_ < next_start):
                    next_start = os_
                    next_label = other

        if next_start is not None and (next_start - end) <= _NEAR_LABEL_CHARS:
            value = self._finalize_value(label, full_text[end:next_start])
            if value:
                return FieldHit(
                    key=key.replace(" ", "_")[:64],
                    label=label,
                    value=value,
                    found=True,
                    confidence=0.8,
                    source_page=page,
                    message=f"نصي: بين «{label}» و«{next_label}»",
                    next_label=next_label,
                    zone="text-between",
                )

        # Local same/next line only.
        tail = full_text[end:]
        nl = tail.find("\n")
        same = self._finalize_value(label, tail if nl < 0 else tail[:nl])
        if same:
            return FieldHit(
                key=key.replace(" ", "_")[:64],
                label=label,
                value=same,
                found=True,
                confidence=0.78,
                source_page=page,
                message=f"نصي: سطر «{label}»",
                zone="text-same-line",
            )
        if nl >= 0:
            rest = tail[nl + 1 :]
            nl2 = rest.find("\n")
            nxt = self._finalize_value(label, rest if nl2 < 0 else rest[:nl2])
            if nxt:
                return FieldHit(
                    key=key.replace(" ", "_")[:64],
                    label=label,
                    value=nxt,
                    found=True,
                    confidence=0.75,
                    source_page=page,
                    message=f"نصي: السطر التالي لـ«{label}»",
                    zone="text-next-line",
                )

        return FieldHit(
            key=key.replace(" ", "_")[:64],
            label=label,
            value="",
            found=False,
            confidence=0.0,
            source_page=page,
            message=f"وُجد «{label}» بدون قيمة قريبة",
        )

    def _collect_blocks(self, pages: list[dict[str, Any]]) -> list[_Block]:
        out: list[_Block] = []
        index = 0
        for page in pages:
            page_no = int(page.get("page_number") or ((page.get("page_index") or 0) + 1))
            raw_blocks = page.get("blocks") or []
            if raw_blocks:
                for i, b in enumerate(raw_blocks):
                    text = str(b.get("text") or "").strip()
                    if not text:
                        continue
                    bbox = b.get("bbox") or [0, i * 40, 400, i * 40 + 30]
                    out.append(
                        _Block(
                            text=text,
                            page=page_no,
                            x0=float(bbox[0]),
                            y0=float(bbox[1]),
                            x1=float(bbox[2]),
                            y1=float(bbox[3]),
                            score=float(b.get("score") or 0.0),
                            index=index,
                        )
                    )
                    index += 1
                continue

            # No geometry: synthesize vertical stack so zoning degrades gracefully.
            lines = page.get("lines")
            if not lines:
                text = page.get("text") or ""
                lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
            for i, line in enumerate(lines):
                cleaned = str(line).strip()
                if not cleaned:
                    continue
                out.append(
                    _Block(
                        text=cleaned,
                        page=page_no,
                        x0=0.0,
                        y0=float(i * 40),
                        x1=500.0,
                        y1=float(i * 40 + 30),
                        score=0.0,
                        index=index,
                    )
                )
                index += 1
        return out

    def _find_text_index(
        self,
        blocks: list[_Block],
        text: str,
        prefer: str = "value",
        after_index: int | None = None,
    ) -> int | None:
        if not text:
            return None
        target = normalize_text(text)
        best: int | None = None
        best_score = -1
        for b in blocks:
            if after_index is not None and b.index <= after_index:
                continue
            bn = normalize_text(b.text)
            score = -1
            if bn == target:
                score = 100
            elif prefer == "label" and self._label_match_quality(b.text, text) > 0:
                score = self._label_match_quality(b.text, text)
            elif target and (target in bn or bn in target):
                score = 50
            if score > best_score:
                best_score = score
                best = b.index
        return best

    def _finalize_value(self, label: str, value: str) -> str:
        value = self._clean_value(value)
        if not value:
            return ""
        label_n = normalize_text(label)
        if any(h in label_n for h in _CODE_LABEL_HINTS):
            compact = value.replace(" ", "")
            m = _CODE_RE.search(compact)
            if m:
                return m.group(0)
            return value.split()[0][:32] if value.split() else value[:32]
        if len(value) > _LOCAL_VALUE_CHARS:
            return value[:_LOCAL_VALUE_CHARS].strip()
        return value

    def _block_matches_label(self, text: str, label: str) -> bool:
        return normalize_text(label) in normalize_text(text)

    def _text_is_label(self, text: str, label_norm: str) -> bool:
        t = normalize_text(text)
        return t == label_norm or t.startswith(label_norm + " ") or t.startswith(
            label_norm + ":"
        )

    def _value_after_label_in_text(self, text: str, label: str) -> str | None:
        parts = [re.escape(p) for p in label.split() if p]
        if not parts:
            return None
        pattern = re.compile(r"\s*".join(parts), re.IGNORECASE)
        m = pattern.search(text)
        if not m:
            return None
        rem = text[m.end() :].strip()
        rem = rem.lstrip(_SEP_CHARS + " \t")
        rem = self._clean_value(rem)
        return rem or None

    @staticmethod
    def _normalize_labels(labels: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in labels:
            label = re.sub(r"\s+", " ", str(raw or "").strip())
            if not label:
                continue
            key = normalize_text(label)
            if key in seen:
                continue
            seen.add(key)
            out.append(label)
        return out

    @staticmethod
    def _index_lines(pages: list[dict[str, Any]]) -> list[tuple[int, str]]:
        indexed: list[tuple[int, str]] = []
        for page in pages:
            page_no = int(page.get("page_number") or ((page.get("page_index") or 0) + 1))
            lines = page.get("lines")
            if not lines:
                text = page.get("text") or ""
                lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
            for line in lines:
                cleaned = str(line).strip()
                if cleaned:
                    indexed.append((page_no, cleaned))
        return indexed

    @staticmethod
    def _build_corpus(lines: list[tuple[int, str]]) -> tuple[str, list[int]]:
        parts: list[str] = []
        page_at: list[int] = []
        for page_no, line in lines:
            if parts:
                parts.append("\n")
                page_at.append(page_no)
            parts.append(line)
            page_at.extend([page_no] * len(line))
        return "".join(parts), page_at

    def _find_label_spans(self, text: str, label: str) -> list[tuple[int, int]]:
        parts = [p for p in label.split() if p]
        if not parts:
            return []
        pattern = re.compile(r"\s*".join(re.escape(p) for p in parts), re.IGNORECASE)
        return [(m.start(), m.end()) for m in pattern.finditer(text)]

    def _clean_value(self, value: str) -> str:
        value = value.replace("\n", " ").strip()
        value = value.strip(_SEP_CHARS + " \t\r\n\"'`")
        value = re.sub(rf"^[{re.escape(_SEP_CHARS)}]+", "", value).strip()
        value = re.sub(rf"[{re.escape(_SEP_CHARS)}]+$", "", value).strip()
        value = re.sub(r"\s+", " ", value)
        return value.strip()
