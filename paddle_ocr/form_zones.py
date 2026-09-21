"""Structured bank form zoning — find answer cells first (like Egyptian ID zones).

Goal: understand the form layout, draw where answers live, then OCR those boxes
so the user can verify zones are correct before trusting values.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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


def normalize_ar(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)
    t = t.translate(_AR_MAP)
    t = re.sub(r"\s+", " ", t)
    return t.casefold()


@dataclass
class OcrBox:
    index: int
    text: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def norm(self) -> str:
        return normalize_ar(self.text)


# ---------------------------------------------------------------------------
# Agricultural Bank — Corporate KYC Update Form
# Answer cells sit BETWEEN English (left) and Arabic (right) labels,
# or LEFT of the Arabic label on branch/CIF rows.
# Split rows: mid EN label + right AR label → value BETWEEN them
#   e.g. Commercial Register Number … رقم القيد فى السجل التجارى
# ---------------------------------------------------------------------------
KYC_FIELDS: list[dict[str, Any]] = [
    {
        "key": "dms_request_no",
        "label_ar": "رقم الطلب على نظام الـ DMS",
        "label_en": "DMS Request No.",
        "aliases": ("رقم الطلب على نظام", "رقم الطلب", "dms"),
        # RTL row: empty answer cell is LEFT of the Arabic label
        "strategy": "left_of_ar_label",
        "section": "header",
        "allow_empty": True,
    },
    {
        "key": "branch_name",
        "label_ar": "اسم الفرع",
        "label_en": "Branch Name",
        "aliases": ("اسم الفرع", "branch name"),
        "strategy": "left_of_ar_label",
        "section": "branch",
    },
    {
        "key": "branch_code",
        "label_ar": "كود الفرع",
        "label_en": "Branch Code",
        "aliases": ("كود الفرع", "branch code"),
        "strategy": "left_of_ar_label",
        "section": "branch",
    },
    {
        "key": "sector_name",
        "label_ar": "اسم القطاع",
        "label_en": "Sector Name",
        "aliases": ("اسم القطاع", "sector name"),
        "strategy": "left_of_ar_label",
        "section": "branch",
    },
    {
        "key": "kyc_update_date",
        "label_ar": "تاريخ تحديث البيانات",
        "label_en": "KYC Update Date",
        "aliases": ("تاريخ تحديث البيانات", "kyc update date"),
        "strategy": "left_of_ar_label",
        "section": "customer",
    },
    {
        "key": "cif_no",
        "label_ar": "رقم العميل",
        "label_en": "CIF No.",
        "aliases": ("رقم العميل", "cif no", "cif"),
        "strategy": "left_of_ar_label",
        "section": "customer",
    },
    {
        "key": "segment",
        "label_ar": "فئة العميل",
        "label_en": "Segment",
        "aliases": ("فئة العميل", "فنه العميل", "فنة العميل", "segment"),
        "strategy": "left_of_ar_label",
        "section": "customer",
    },
    {
        "key": "company_name",
        "label_ar": "اسم المنشأة بالكامل",
        "label_en": "Company Name",
        "aliases": ("اسم المنشأه بالكامل", "اسم المنشاه بالكامل", "اسم المنشأة", "company name"),
        "strategy": "between_en_ar",
        "en_aliases": ("company name",),
        "section": "company",
    },
    {
        "key": "trading_name",
        "label_ar": "السمة التجارية",
        "label_en": "Trading Name",
        "aliases": ("السمه التجاريه", "السمة التجارية", "trading name"),
        "strategy": "between_en_ar",
        "en_aliases": ("trading name",),
        "section": "company",
    },
    {
        "key": "commercial_name",
        "label_ar": "الاسم التجاري",
        "label_en": "Commercial Name",
        "aliases": ("الاسم التجاري", "الاسم التجارى", "commercial name"),
        "strategy": "between_en_ar",
        "en_aliases": ("commercial name",),
        "section": "company",
    },
    {
        "key": "legal_type",
        "label_ar": "الشكل القانوني",
        "label_en": "Company Legal Type",
        "aliases": ("الشكل القانونى", "الشكل القانوني"),
        "strategy": "between_en_ar",
        "en_aliases": ("company legal type",),
        "section": "company",
        "allow_empty": True,
    },
    {
        "key": "commercial_register",
        "label_ar": "رقم القيد فى السجل التجارى / الاشهار",
        "label_en": "Commercial Register Number / Announce",
        "aliases": (
            "رقم القيد فى السجل",
            "رقم القيد في السجل",
            "التجارى الاشهار",
            "التجاري الاشهار",
            "التجاريالاشهار",
        ),
        "strategy": "mid_cell",
        "en_aliases": (
            "commercial register number",
            "register number",
            "register number/",
            "announce",
        ),
        "section": "company",
        "allow_empty": True,
    },
    {
        "key": "activity_start_date",
        "label_ar": "تاريخ بدء مزاولة النشاط",
        "label_en": "Date of Incorporation",
        "aliases": ("تاريخ بدء مزاولة", "بدء مزاولة"),
        "strategy": "between_en_ar",
        "en_aliases": ("date of incorporation",),
        "section": "company",
        "allow_empty": True,
    },
    {
        "key": "registration_date",
        "label_ar": "تاريخ القيد فى السجل التجارى / الاشهار",
        "label_en": "Registration Date / Announce",
        "aliases": ("تاريخ القيد فى السجل", "تاريخ القيد في السجل"),
        "strategy": "mid_cell",
        "en_aliases": ("registration date", "registration date/", "announce"),
        "section": "company",
        "allow_empty": True,
    },
    {
        "key": "country_of_incorporation",
        "label_ar": "بلد التأسيس",
        "label_en": "Country of Incorporation",
        "aliases": ("بلد التاسيس", "بلد التأسيس"),
        "strategy": "between_en_ar",
        "en_aliases": ("country of incorporation", "country of"),
        "section": "company",
        "allow_empty": True,
    },
    {
        "key": "place_of_issue",
        "label_ar": "جهة الاصدار السجل التجارى",
        "label_en": "Place of Issue",
        "aliases": ("جهه الاصدار", "جهة الاصدار", "جهة الإصدار"),
        "strategy": "mid_cell",
        "en_aliases": ("place of issue",),
        "section": "company",
        "allow_empty": True,
    },
]

ZONE_COLORS = [
    (40, 160, 70),
    (40, 100, 220),
    (200, 120, 40),
    (180, 60, 180),
    (60, 180, 200),
    (80, 80, 220),
    (40, 140, 140),
    (160, 90, 40),
]

# Normalized answer zones — calibrated so mid-row fields (رقم القيد…) sit
# BETWEEN the English mid-label and the Arabic right-label.
FORM_ZONES_ABE_KYC: dict[str, tuple[float, float, float, float]] = {
    # رقم الطلب: الخانة البيضاء فقط (شمال التسمية) — من غير ما يغطي اللابل
    "dms_request_no": (0.05, 0.225, 0.285, 0.305),
    "branch_name": (0.72, 0.32, 0.88, 0.41),
    "branch_code": (0.42, 0.32, 0.58, 0.41),
    "sector_name": (0.08, 0.32, 0.28, 0.41),
    "kyc_update_date": (0.64, 0.41, 0.82, 0.50),
    "cif_no": (0.38, 0.41, 0.52, 0.50),
    "segment": (0.08, 0.41, 0.28, 0.50),
    "company_name": (0.18, 0.57, 0.87, 0.63),
    "trading_name": (0.17, 0.63, 0.88, 0.69),
    "commercial_name": (0.20, 0.69, 0.89, 0.76),
    "legal_type": (0.18, 0.78, 0.40, 0.84),
    "commercial_register": (0.55, 0.76, 0.87, 0.84),
    "activity_start_date": (0.18, 0.85, 0.38, 0.93),
    "registration_date": (0.55, 0.85, 0.86, 0.93),
    "country_of_incorporation": (0.14, 0.93, 0.42, 1.00),
    "place_of_issue": (0.55, 0.93, 0.82, 1.00),
}


def _field_display_label(fdef: dict[str, Any]) -> str:
    ar = str(fdef.get("label_ar") or fdef.get("label") or "").strip()
    en = str(fdef.get("label_en") or "").strip()
    if ar and en:
        return f"{ar} / {en}"
    return ar or en or str(fdef.get("key") or "")


FORM_ZONE_LABELS: dict[str, str] = {
    f["key"]: _field_display_label(f) for f in KYC_FIELDS
}


def _shape_arabic(text: str) -> str:
    """Reshape Arabic for correct visual order in PIL."""
    if not text or not re.search(r"[\u0600-\u06FF]", text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:  # noqa: BLE001
        return text


def _pil_font(size: int = 14):
    from PIL import ImageFont

    for path in (
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _bgr_to_pil(image_bgr: np.ndarray):
    from PIL import Image

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pil_to_bgr(image_pil) -> np.ndarray:
    rgb = np.asarray(image_pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_label_chip(
    image_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    font_size: int = 13,
) -> np.ndarray:
    """Draw a filled bilingual label chip (Arabic+English) with PIL."""
    from PIL import ImageDraw

    h, w = image_bgr.shape[:2]
    x, y = origin
    shaped = _shape_arabic(text)
    font = _pil_font(font_size)
    pil = _bgr_to_pil(image_bgr)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), shaped, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 5, 3
    chip_w, chip_h = tw + pad_x * 2, th + pad_y * 2
    # Place chip just above the zone; keep inside image
    lx = min(max(0, x), max(0, w - chip_w - 1))
    ly = max(0, y - chip_h - 2)
    if ly + chip_h > h:
        ly = max(0, h - chip_h)
    fill = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))  # RGB
    draw.rectangle([lx, ly, lx + chip_w, ly + chip_h], fill=fill)
    draw.text((lx + pad_x, ly + pad_y - 1), shaped, font=font, fill=(255, 255, 255))
    return _pil_to_bgr(pil)


def draw_form_zones_overlay(
    image_bgr: np.ndarray,
    zones: dict[str, tuple[float, float, float, float]] | None = None,
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw labeled form zones like egyptian_id.draw_id_zones / front_zones.jpg.

    Chip text = Arabic / English, e.g. «اسم المنشأة بالكامل / Company Name».
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    zones = zones or FORM_ZONES_ABE_KYC
    for i, (name, (x0, y0, x1, y1)) in enumerate(zones.items()):
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        p0 = (int(round(x0 * w)), int(round(y0 * h)))
        p1 = (int(round(x1 * w)), int(round(y1 * h)))
        cv2.rectangle(out, p0, p1, color, thickness)
        fdef = next((f for f in KYC_FIELDS if f["key"] == name), None)
        ar = str((fdef or {}).get("label_ar") or "").strip()
        en = str((fdef or {}).get("label_en") or name).strip()
        # Example: اسم المنشأة بالكامل / Company Name
        if ar and en:
            title = f"{ar} / {en}"
        else:
            title = ar or en or name
        label = f"{title}  ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f})"
        # Slightly smaller font on dense rows
        fs = 12 if len(title) > 42 else 13
        out = _draw_label_chip(out, label, p0, color, font_size=fs)
    return out


def save_form_zones_reference(
    image_path: str | Path,
    dest: str | Path | None = None,
) -> Path:
    """Build a front_zones-style reference image for the KYC form."""
    img = _load_bgr(image_path)
    overlay = draw_form_zones_overlay(img, FORM_ZONES_ABE_KYC)
    out = Path(dest) if dest else Path(image_path).resolve().parent / "form_zones.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


def _label_alias_set() -> set[str]:
    out: set[str] = set()
    for f in KYC_FIELDS:
        for a in f.get("aliases") or ():
            out.add(normalize_ar(a))
        for a in f.get("en_aliases") or ():
            out.add(normalize_ar(a))
    # Common English twins printed under Arabic headers
    for a in (
        "branch name",
        "branch code",
        "sector name",
        "segment",
        "cif no",
        "cif no.",
        "kyc update date",
        "company name",
        "trading name",
        "commercial name",
        "company legal type",
        "commercial",
        "register number",
        "announce",
        "date of incorporation",
        "registration date",
        "country of",
        "incorporation",
        "place of issue",
        "اسم الفرع",
        "كود الفرع",
        "اسم القطاع",
        "فئة العميل",
        "فنة العميل",
        "رقم العميل",
        "تاريخ تحديث البيانات",
    ):
        out.add(normalize_ar(a))
    return out


@dataclass
class AnswerZone:
    key: str
    label: str
    label_ar: str = ""
    label_en: str = ""
    section: str = ""
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    value: str | None = None
    found: bool = False
    confidence: float = 0.0
    strategy: str = ""
    message: str = ""
    crop_b64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FormZoneResult:
    form_type: str
    form_title: str
    message: str
    fields: list[AnswerZone] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    images: dict[str, Any] = field(default_factory=dict)
    ocr_list: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "form_type": self.form_type,
            "form_title": self.form_title,
            "message": self.message,
            "fields": [f.to_dict() for f in self.fields],
            "rules": self.rules,
            "images": self.images,
            "ocr_list": self.ocr_list,
            "found_count": sum(1 for f in self.fields if f.found),
            "zone_count": len(self.fields),
        }


def _encode_b64(image_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _load_bgr(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def _blocks_from_engine(engine: Any, image_bgr: np.ndarray) -> list[OcrBox]:
    blocks = engine.ocr_bgr(image_bgr)
    out: list[OcrBox] = []
    for i, b in enumerate(blocks or []):
        text = str(getattr(b, "text", "") or "").strip()
        if not text:
            continue
        out.append(
            OcrBox(
                index=i,
                text=text,
                score=float(getattr(b, "score", 0) or 0),
                x0=float(getattr(b, "x0", 0) or 0),
                y0=float(getattr(b, "y0", 0) or 0),
                x1=float(getattr(b, "x1", 0) or 0),
                y1=float(getattr(b, "y1", 0) or 0),
            )
        )
    return out


def _alias_hit(box: OcrBox, aliases: tuple[str, ...]) -> float:
    n = box.norm
    best = 0.0
    for a in aliases:
        an = normalize_ar(a)
        if not an:
            continue
        if n == an:
            best = max(best, 1.0)
        elif an in n or n in an:
            best = max(best, 0.85 if min(len(n), len(an)) >= 4 else 0.55)
        elif len(an) >= 6 and an[:6] in n:
            best = max(best, 0.6)
    return best


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _find_label(
    boxes: list[OcrBox],
    aliases: tuple[str, ...],
    *,
    prefer: str = "any",
) -> OcrBox | None:
    scored = [(_alias_hit(b, aliases), b) for b in boxes]
    scored = [x for x in scored if x[0] > 0]
    if prefer == "ar":
        scored = [x for x in scored if _has_arabic(x[1].text)] or scored
    elif prefer == "en":
        scored = [x for x in scored if not _has_arabic(x[1].text)] or scored
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1].y0, -x[1].x0 if prefer != "en" else x[1].x0))
    return scored[0][1]


def _same_row(a: OcrBox, b: OcrBox, tol: float = 18.0) -> bool:
    if abs(a.cy - b.cy) <= tol:
        return True
    ah = max(a.y1 - a.y0, 1.0)
    bh = max(b.y1 - b.y0, 1.0)
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap >= 0.35 * min(ah, bh)


def _is_labelish(text: str, all_aliases: set[str]) -> bool:
    n = normalize_ar(text)
    if not n:
        return False
    for a in all_aliases:
        if a and (n == a or a in n or n in a):
            return True
    # English title-case label words
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9 ./&\-]{2,40}", text.strip()):
        low = text.strip().casefold()
        if any(k in low for k in ("name", "code", "date", "type", "segment", "cif", "branch", "sector", "company", "trading", "commercial", "register", "announce", "country", "place", "incorporation")):
            return True
    return False


def _value_in_box(
    boxes: list[OcrBox],
    bbox: tuple[float, float, float, float],
    *,
    exclude_aliases: set[str],
) -> tuple[str | None, float, list[OcrBox]]:
    x0, y0, x1, y1 = bbox
    hits: list[OcrBox] = []
    for b in boxes:
        if b.cx < x0 or b.cx > x1 or b.cy < y0 or b.cy > y1:
            # also accept strong overlap
            ox0, oy0 = max(b.x0, x0), max(b.y0, y0)
            ox1, oy1 = min(b.x1, x1), min(b.y1, y1)
            if ox1 - ox0 < 4 or oy1 - oy0 < 4:
                continue
            area = (b.x1 - b.x0) * (b.y1 - b.y0)
            inter = (ox1 - ox0) * (oy1 - oy0)
            if area <= 0 or inter / area < 0.45:
                continue
        if _is_labelish(b.text, exclude_aliases):
            continue
        hits.append(b)
    if not hits:
        return None, 0.0, []
    hits.sort(key=lambda b: (b.y0, b.x0))
    texts = [h.text.strip() for h in hits if h.text.strip()]
    # Prefer non-empty meaningful content
    joined = " ".join(texts).strip()
    conf = float(sum(h.score for h in hits) / max(len(hits), 1))
    return (joined or None), conf, hits


def _estimate_answer_bbox(
    field: dict[str, Any],
    boxes: list[OcrBox],
    w: int,
    h: int,
) -> tuple[tuple[float, float, float, float] | None, OcrBox | None, str]:
    """Return (answer_bbox, label_box, note)."""
    aliases = tuple(field.get("aliases") or ())
    ar_aliases = tuple(a for a in aliases if _has_arabic(a) or not re.search(r"[A-Za-z]", a))
    if not ar_aliases:
        ar_aliases = aliases
    ar = _find_label(boxes, ar_aliases, prefer="ar")
    en = None
    en_aliases = tuple(field.get("en_aliases") or ())
    # Also allow Latin aliases listed under aliases
    en_extra = tuple(a for a in aliases if re.search(r"[A-Za-z]", a))
    if en_aliases or en_extra:
        en = _find_label(boxes, en_aliases + en_extra, prefer="en")

    strategy = field.get("strategy") or "left_of_ar_label"
    pad = 4.0

    if strategy == "right_of_label":
        if not ar:
            return None, None, "label missing"
        # empty box to the right of DMS label (form has value cell on the right)
        x0 = min(w - 8.0, ar.x1 + 8)
        x1 = float(w) - 12.0
        y0 = max(0.0, ar.y0 - 6)
        y1 = min(float(h), ar.y1 + 10)
        return (x0, y0, x1, y1), ar, "right_of_label"

    if strategy == "between_en_ar":
        if ar and en and _same_row(ar, en, tol=28):
            x0 = en.x1 + pad
            x1 = ar.x0 - pad
            y0 = min(en.y0, ar.y0) - 4
            y1 = max(en.y1, ar.y1) + 6
            if x1 - x0 < 20:
                # labels too close — expand center band
                mid = (en.cx + ar.cx) / 2
                x0, x1 = mid - 80, mid + 80
            return (max(0, x0), max(0, y0), min(w, x1), min(h, y1)), ar, "between_en_ar"
        if ar:
            # fallback: left of Arabic label in center band
            x1 = ar.x0 - pad
            x0 = max(60.0, x1 - 280)
            y0, y1 = ar.y0 - 4, ar.y1 + 6
            return (x0, y0, x1, y1), ar, "left_of_ar (no EN)"
        return None, None, "label missing"

    if strategy == "mid_cell":
        # Split-row right field: value BETWEEN mid English label and right Arabic label
        # e.g. [Register Number / Announce]  ⟵ VALUE ⟶  [رقم القيد فى السجل…]
        if not ar:
            return None, None, "label missing"
        mid_en = en
        if mid_en is None or mid_en.cx >= ar.cx:
            row_boxes = [b for b in boxes if _same_row(b, ar, tol=36)]
            mid_en = _find_label(
                row_boxes,
                en_aliases
                or (
                    "commercial register number",
                    "register number",
                    "registration date",
                    "place of issue",
                    "announce",
                ),
                prefer="en",
            )
        if mid_en and mid_en.cx < ar.cx - 8:
            x0 = mid_en.x1 + pad
            x1 = ar.x0 - pad
            y0 = min(mid_en.y0, ar.y0) - 8
            y1 = max(mid_en.y1, ar.y1) + 12
            # If labels are stacked vertically (Announce under Register), widen using cx
            if x1 - x0 < 30:
                x0 = max(mid_en.x0 - 10, ar.x0 - max(180.0, (ar.x0 - mid_en.cx)))
                x1 = ar.x0 - pad
            return (
                (max(0, x0), max(0, y0), min(w, x1), min(h, y1)),
                ar,
                "mid_cell between EN-AR",
            )
        # Fallback: cell immediately left of Arabic label (typical empty answer box)
        x1 = ar.x0 - pad
        x0 = max(0.0, x1 - 220)
        y0, y1 = ar.y0 - 10, ar.y1 + 14
        return (x0, y0, x1, y1), ar, "mid_cell left_of_ar"

    # left_of_ar_label (branch / CIF style): value immediately left of Arabic label
    if not ar:
        return None, None, "label missing"

    # Skip English twin labels sitting beside the Arabic label (Branch Name / اسم الفرع)
    left_content = [
        b
        for b in boxes
        if b.index != ar.index
        and _same_row(b, ar, tol=18)
        and b.cx < ar.cx - 10
        and not _is_labelish(b.text, _label_alias_set())
    ]
    left_content.sort(key=lambda b: ar.x0 - b.x1)

    # Also find left-side labels to bound the cell
    left_labels = [
        b
        for b in boxes
        if b.index != ar.index
        and _same_row(b, ar, tol=18)
        and b.cx < ar.cx - 10
        and _is_labelish(b.text, _label_alias_set())
    ]
    left_labels.sort(key=lambda b: ar.x0 - b.x1)
    # Nearest label to the left that is still left of the value (e.g. كود الفرع)
    # Prefer Arabic labels further left as the previous column header
    boundary_right = ar.x0 - pad
    boundary_left = 0.0
    for lb in left_labels:
        # English twin almost touching Arabic label — ignore as column boundary
        if ar.x0 - lb.x1 < 55 and abs(lb.cy - ar.cy) < 22:
            continue
        boundary_left = lb.x1 + pad
        break

    if left_content:
        nb = left_content[0]
        # Prefer the content box itself when it sits in the column gap
        if nb.x1 <= boundary_right and (boundary_left <= 0 or nb.cx >= boundary_left - 20):
            x0, y0, x1, y1 = nb.x0 - 4, min(nb.y0, ar.y0) - 4, nb.x1 + 4, max(nb.y1, ar.y1) + 4
            return (max(0, x0), max(0, y0), min(w, x1), min(h, y1)), ar, "left_of_ar_label"
        # Else use gap between previous label and this Arabic label
        x0 = max(boundary_left, nb.x0 - 4)
        x1 = boundary_right
        y0, y1 = min(nb.y0, ar.y0) - 4, max(nb.y1, ar.y1) + 4
        return (max(0, x0), max(0, y0), min(w, x1), min(h, y1)), ar, "left_of_ar_label"

    x0 = boundary_left if boundary_left > 0 else max(0.0, ar.x0 - 140)
    x1 = boundary_right
    y0, y1 = ar.y0 - 4, ar.y1 + 4
    if x1 - x0 < 20:
        x0 = max(0.0, x1 - 120)
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1)), ar, "left_of_ar_label"


def _draw_zones(
    image_bgr: np.ndarray,
    zones: list[AnswerZone],
) -> np.ndarray:
    canvas = image_bgr.copy()
    for i, z in enumerate(zones):
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        x0, y0, x1, y1 = [int(v) for v in z.bbox]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        # Prefer «عربي / English» like: اسم المنشأة بالكامل / Company Name
        if z.label_ar and z.label_en:
            tag = f"{z.label_ar} / {z.label_en}"
        else:
            tag = z.label or z.label_ar or z.label_en or z.key
        canvas = _draw_label_chip(
            canvas,
            f"{i + 1}. {tag}",
            (x0, y0),
            color,
            font_size=12 if len(tag) > 40 else 13,
        )
        if z.found and z.value:
            canvas = _draw_label_chip(
                canvas,
                str(z.value)[:48],
                (x0, min(image_bgr.shape[0] - 2, y1 + 18)),
                (40, 140, 40),
                font_size=12,
            )
    return canvas


def _crop_zone(image_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return image_bgr[y0:y1, x0:x1].copy()


def detect_form_type(boxes: list[OcrBox]) -> tuple[str, str]:
    blob = normalize_ar(" ".join(b.text for b in boxes[:12]))
    if "استماره تحديث" in blob or "شخصيات اعتباري" in blob or "corporate kyc" in blob:
        return "abe_corporate_kyc", "استمارة تحديث البيانات للشخصيات الاعتبارية (البنك الزراعي)"
    if "البنك الزراعي" in blob or "agricultural bank" in blob:
        return "abe_form", "نموذج البنك الزراعي المصري"
    return "generic_form", "نموذج / استمارة عامة"


def _rotate_image(image_bgr: np.ndarray, angle_cw: int) -> np.ndarray:
    ang = int(angle_cw) % 360
    if ang == 0 or image_bgr is None or not getattr(image_bgr, "size", 0):
        return image_bgr
    code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(ang)
    if code is None:
        return image_bgr
    return cv2.rotate(image_bgr, code)


def _horizontal_structure_score(image_bgr: np.ndarray) -> float:
    """Upright forms have strong horizontal text/table lines."""
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
        )
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 28))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
        h_n = float(np.count_nonzero(h_lines))
        v_n = float(np.count_nonzero(v_lines))
        return (h_n - v_n) / max(h_n + v_n, 1.0) * 20.0
    except Exception:  # noqa: BLE001
        return 0.0


def _top_header_score(image_bgr: np.ndarray) -> float:
    """Upright bank form: logo/title denser near the TOP (and often top-left)."""
    try:
        h, w = image_bgr.shape[:2]
        if h < 40 or w < 40:
            return 0.0
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        top = gray[: max(1, h // 5), :]
        bot = gray[4 * h // 5 :, :]
        top_ink = float((top < 110).mean())
        bot_ink = float((bot < 110).mean())
        score = (top_ink - bot_ink) * 35.0

        # Logo / brand colors (gold/green) tend to sit TOP-LEFT when upright
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gold = cv2.inRange(hsv, (12, 35, 70), (42, 255, 255))
        green = cv2.inRange(hsv, (35, 35, 40), (95, 255, 255))
        brand = cv2.bitwise_or(gold, green)
        tl = float(brand[: max(1, h // 4), : max(1, w // 3)].mean())
        br = float(brand[3 * h // 4 :, 2 * w // 3 :].mean())
        bl = float(brand[3 * h // 4 :, : max(1, w // 3)].mean())
        tr = float(brand[: max(1, h // 4), 2 * w // 3 :].mean())
        score += (tl - br) * 0.35
        score += (tl - bl) * 0.15
        score += (tr - br) * 0.05

        # Title band: more edge energy in upper fifth when upright
        edges = cv2.Canny(gray, 60, 160)
        top_e = float(edges[: max(1, h // 5), :].mean())
        bot_e = float(edges[4 * h // 5 :, :].mean())
        score += (top_e - bot_e) * 0.25
        return score
    except Exception:  # noqa: BLE001
        return 0.0


def score_form_upright(image_bgr: np.ndarray) -> float:
    """Higher = more likely upright ABE KYC-style form (usually landscape)."""
    if image_bgr is None or not getattr(image_bgr, "size", 0):
        return -1e9
    h, w = image_bgr.shape[:2]
    score = 0.0
    # Sample form is landscape when upright
    if w >= h:
        score += 4.0
    else:
        score -= 3.0
    score += _horizontal_structure_score(image_bgr)
    score += _top_header_score(image_bgr)
    return score


def correct_form_orientation(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, int, str]:
    """Auto-rotate form to upright (0/90/180/270)."""
    if image_bgr is None or not getattr(image_bgr, "size", 0):
        return image_bgr, 0, "ORIENT: تخطي"
    h, w = image_bgr.shape[:2]
    probe = image_bgr
    max_side = max(h, w)
    if max_side > 900:
        s = 900 / max_side
        probe = cv2.resize(
            image_bgr,
            (max(1, int(w * s)), max(1, int(h * s))),
            interpolation=cv2.INTER_AREA,
        )

    scored: list[tuple[float, int]] = []
    for ang in (0, 90, 180, 270):
        rot = _rotate_image(probe, ang)
        sc = score_form_upright(rot)
        scored.append((sc, ang))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_sc, best_ang = scored[0]

    # If 0 and 180 are close, prefer the one with stronger top-header cue
    near = [(sc, ang) for sc, ang in scored if abs(sc - best_sc) < 2.5]
    if len(near) >= 2:
        refined = []
        for sc, ang in near:
            rot = _rotate_image(probe, ang)
            refined.append((sc + _top_header_score(rot) * 0.5, ang))
        refined.sort(key=lambda x: x[0], reverse=True)
        best_ang = refined[0][1]

    fixed = _rotate_image(image_bgr, best_ang) if best_ang else image_bgr
    if best_ang:
        note = f"ORIENT: تدوير الفورم {best_ang}° قبل القراءة"
    else:
        note = "ORIENT: الاتجاه مضبوط"
    return fixed, best_ang, note


def auto_deskew_form(image_bgr: np.ndarray, max_angle: float = 12.0) -> tuple[np.ndarray, float]:
    """Correct small camera tilt on forms."""
    try:
        from egyptian_id import auto_deskew_card

        before = image_bgr
        after = auto_deskew_card(image_bgr, max_angle=max_angle)
        # Estimate applied angle roughly from difference — report if changed
        if after is before or after.shape != before.shape:
            return after, 0.0
        if np.array_equal(after, before):
            return after, 0.0
        return after, 1.0  # deskew applied (angle unknown precisely)
    except Exception:  # noqa: BLE001
        logger.debug("form deskew skipped", exc_info=True)
        return image_bgr, 0.0


class FormZoneExtractor:
    """Locate answer zones on structured forms, annotate, then read values."""

    def process_image_path(
        self,
        image_path: str | Path,
        *,
        use_handwriting: bool = False,
    ) -> FormZoneResult:
        from ocr_engine import ArabicOcrEngine

        original = _load_bgr(image_path)

        # 1) Orient first (handles phone photos / upside-down scans)
        oriented, orient_ang, orient_note = correct_form_orientation(original)
        # 2) Deskew small tilt
        deskewed, deskew_flag = auto_deskew_form(oriented, max_angle=12.0)
        # 3) Second orientation pass after deskew (rare)
        oriented2, orient_ang2, orient_note2 = correct_form_orientation(deskewed)
        if orient_ang2:
            oriented = oriented2
            orient_note = (
                f"{orient_note} → بعد تصحيح الميل: تدوير {orient_ang2}°"
                if orient_ang
                else orient_note2
            )
            orient_ang = (orient_ang + orient_ang2) % 360
        else:
            oriented = deskewed

        work = oriented
        oh, ow = oriented.shape[:2]
        scale = 1.0
        if max(oh, ow) > 1600:
            scale = 1600 / max(oh, ow)
            work = cv2.resize(
                oriented,
                (int(ow * scale), int(oh * scale)),
                interpolation=cv2.INTER_AREA,
            )
        h, w = work.shape[:2]
        engine = ArabicOcrEngine.get()
        boxes = _blocks_from_engine(engine, work)
        form_type, form_title = detect_form_type(boxes)
        rules = [
            f"FORM: {form_type}",
            orient_note,
            ("DESKEW: تم تصحيح الميل" if deskew_flag else "DESKEW: لا يوجد ميل يذكر"),
            f"OCR_BLOCKS: {len(boxes)}",
        ]
        hw_engine = None
        if use_handwriting:
            try:
                from handwritten_ocr import HandwrittenOcrEngine

                hw_engine = HandwrittenOcrEngine.get()
                rules.append(
                    "HANDWRITING: Arabic-English-handwritten-OCR-v3 "
                    f"({HandwrittenOcrEngine.model_source()})"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("handwriting engine unavailable")
                rules.append(f"HANDWRITING_FAIL: {exc}")
                hw_engine = None
        _ = orient_ang
        _ = scale

        schema = KYC_FIELDS if form_type.startswith("abe") else KYC_FIELDS
        all_aliases: set[str] = set()
        for f in schema:
            for a in f.get("aliases") or ():
                all_aliases.add(normalize_ar(a))
            for a in f.get("en_aliases") or ():
                all_aliases.add(normalize_ar(a))

        zones: list[AnswerZone] = []
        for fdef in schema:
            disp = _field_display_label(fdef)
            label_ar = str(fdef.get("label_ar") or "")
            label_en = str(fdef.get("label_en") or "")
            bbox, label_box, note = _estimate_answer_bbox(fdef, boxes, w, h)
            # Prefer calibrated mid-row answer cells when dynamic box is weak
            cal = FORM_ZONES_ABE_KYC.get(str(fdef["key"]))
            use_cal = False
            if cal:
                if bbox is None:
                    use_cal = True
                elif fdef["key"] == "dms_request_no":
                    # Always prefer calibrated DMS answer cell (left of Arabic label)
                    use_cal = True
                elif (bbox[2] - bbox[0]) < 40:
                    use_cal = True
                elif fdef.get("strategy") == "mid_cell" and bbox[2] - bbox[0] < w * 0.18:
                    use_cal = True
            if use_cal and cal:
                bbox = (cal[0] * w, cal[1] * h, cal[2] * w, cal[3] * h)
                note = (note or "calibrated") + "+FORM_ZONES"
            if bbox is None:
                zones.append(
                    AnswerZone(
                        key=fdef["key"],
                        label=disp,
                        label_ar=label_ar,
                        label_en=label_en,
                        section=str(fdef.get("section") or ""),
                        bbox=(0, 0, 0, 0),
                        strategy=str(fdef.get("strategy") or ""),
                        message=f"لم يُعثر على التسمية — {note}",
                    )
                )
                rules.append(f"ZONE_MISS: {fdef['key']}")
                continue

            value, conf, hits = _value_in_box(boxes, bbox, exclude_aliases=all_aliases)

            def _reject_junk(val: str | None) -> str | None:
                if not val:
                    return None
                n = normalize_ar(val)
                if not n or n in {"لنشاط", "اشهار", "announce", "commercial", "register"}:
                    return None
                if "لنشاط" in n or n.endswith("اشهار"):
                    return None
                if _is_labelish(val, all_aliases) and len(n) <= 18:
                    return None
                # OCR scraps from English mid-labels (Number → Jmber), etc.)
                if re.fullmatch(r"[A-Za-z0-9)(/.\-]{1,14}", val.strip()) and not re.search(
                    r"\d{2,}", val
                ):
                    low = val.casefold()
                    if any(
                        k in low
                        for k in (
                            "number",
                            "jmber",
                            "announce",
                            "register",
                            "commercial",
                            "date",
                            "place",
                            "issue",
                            "type",
                            "name",
                        )
                    ):
                        return None
                    if len(val.strip()) <= 8:
                        return None
                return val

            value = _reject_junk(value)
            if not value:
                conf, hits = 0.0, []
            # Re-OCR crop for empty/weak cells (Paddle) — or prefer handwriting VLM
            crop = _crop_zone(work, bbox)
            crop_b64 = _encode_b64(crop, 90) if crop is not None and crop.size else None
            hw_ok = False
            if hw_engine is not None and crop is not None and crop.size:
                try:
                    hw_label = disp or label_ar or label_en or str(fdef["key"])
                    hw_text = hw_engine.recognize_field_bgr(crop, label=hw_label)
                    hw_clean = _reject_junk(hw_text.strip() if hw_text else None)
                    if hw_clean:
                        value = hw_clean
                        conf = max(conf, 0.72)
                        hw_ok = True
                        rules.append(f"ZONE_HW: {fdef['key']} ← handwritten-OCR-v3")
                    else:
                        rules.append(f"ZONE_HW_EMPTY: {fdef['key']}")
                except Exception:  # noqa: BLE001
                    logger.debug("zone handwriting OCR failed", exc_info=True)
                    rules.append(f"ZONE_HW_ERR: {fdef['key']}")
            if (not hw_ok) and (not value or conf < 0.55) and crop is not None and crop.size:
                try:
                    big = crop
                    if max(crop.shape[:2]) < 80:
                        big = cv2.resize(crop, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
                    crop_blocks = _blocks_from_engine(engine, big)
                    texts = [
                        b.text.strip()
                        for b in crop_blocks
                        if b.text.strip() and not _is_labelish(b.text, all_aliases)
                    ]
                    joined = _reject_junk(" ".join(texts).strip() or None)
                    if joined:
                        value = joined
                        conf = float(sum(b.score for b in crop_blocks) / max(len(crop_blocks), 1))
                        rules.append(f"ZONE_OCR: {fdef['key']} ← crop")
                except Exception:  # noqa: BLE001
                    logger.debug("zone crop OCR failed", exc_info=True)

            found = bool(value and str(value).strip())
            zones.append(
                AnswerZone(
                    key=fdef["key"],
                    label=disp,
                    label_ar=label_ar,
                    label_en=label_en,
                    section=str(fdef.get("section") or ""),
                    bbox=bbox,
                    value=value if found else None,
                    found=found,
                    confidence=round(conf, 3),
                    strategy=str(fdef.get("strategy") or ""),
                    message=note if found else (f"منطقة محددة · فارغة ({note})" if fdef.get("allow_empty") else f"منطقة محددة · لا قيمة ({note})"),
                    crop_b64=crop_b64,
                )
            )
            rules.append(
                f"ZONE: {fdef['key']} [{label_ar} | {label_en}] "
                f"@ ({int(bbox[0])},{int(bbox[1])})-({int(bbox[2])},{int(bbox[3])})"
                + (f" = {value}" if found else " = ∅")
            )

        annotated = _draw_zones(work, [z for z in zones if z.bbox[2] > z.bbox[0]])
        preview = work
        if max(preview.shape[:2]) > 1200:
            s = 1200 / max(preview.shape[:2])
            preview = cv2.resize(preview, (int(preview.shape[1] * s), int(preview.shape[0] * s)))

        found_n = sum(1 for z in zones if z.found)
        zone_n = sum(1 for z in zones if z.bbox[2] > z.bbox[0])
        hw_note = " · خط يد (handwritten-OCR-v3)" if hw_engine is not None else ""
        message = (
            f"[form-zones] {orient_note} · تم تحديد {zone_n} منطقة إجابة · قُرئ منها {found_n} قيمة"
            f"{hw_note}. راجع المربعات على الصورة للتأكد."
        )

        ocr_list = [
            {
                "index": b.index,
                "text": b.text,
                "bbox": [b.x0, b.y0, b.x1, b.y1],
                "score": b.score,
            }
            for b in boxes
        ]

        return FormZoneResult(
            form_type=form_type,
            form_title=form_title,
            message=message,
            fields=zones,
            rules=rules,
            images={
                "original": _encode_b64(preview, 82),
                "annotated": _encode_b64(annotated, 90),
                "crops": {z.key: z.crop_b64 for z in zones if z.crop_b64},
            },
            ocr_list=ocr_list,
        )
