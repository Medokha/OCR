"""Egyptian national ID card extraction (open-source: PaddleOCR + OpenCV).

Field mapping uses layout zones + label rules + national-ID equations
so birth date / gender / governorate are derived mathematically from the 14 digits.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# ---------------------------------------------------------------------------
# Egyptian ID layout zones (normalized 0..1 of the warped card image)
# Format per box: (x0, y0, x1, y1)  → left, top, right, bottom
#
# FRONT (from real card photos):
#   photo LEFT | Arabic text RIGHT | national ID bottom strip
#
# BACK:
#   job / status / expiry above the 2D barcode
# ---------------------------------------------------------------------------
ID_ZONES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "front": {
        # مُعايرة من صورة الوجه الفعلية (grid 0.1) — نفس شكل المحدد في front_zones
        # الصورة الشخصية — يسار
        "face": (0.02, 0.08, 0.30, 0.58),
        # الهيدر الأخضر
        "header": (0.38, 0.02, 0.96, 0.22),
        # الاسم سطرين (محمد + باقي الاسم)
        "name": (0.48, 0.23, 0.97, 0.43),
        # العنوان سطرين (سمادون + مركز …)
        "address": (0.52, 0.43, 0.97, 0.64),
        # الرقم القومي
        "national_id": (0.38, 0.70, 0.97, 0.86),
        # رقم المصنع
        "serial": (0.04, 0.87, 0.35, 0.98),
    },
    "back": {
        # المهنة
        "job": (0.05, 0.04, 0.95, 0.30),
        # نوع / ديانة / حالة اجتماعية (سطر واحد)
        "status": (0.05, 0.26, 0.95, 0.44),
        # البطاقة سارية حتى …
        "expiry": (0.05, 0.40, 0.95, 0.58),
        # الباركود ثنائي الأبعاد
        "barcode": (0.02, 0.55, 0.98, 0.98),
    },
}

# Colors for drawing zone overlays (BGR)
ID_ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "face": (80, 180, 40),
    "header": (60, 160, 220),
    "name": (40, 140, 80),
    "address": (180, 110, 40),
    "national_id": (40, 40, 220),
    "serial": (160, 160, 160),
    "job": (40, 140, 80),
    "status": (180, 110, 40),
    "expiry": (40, 40, 220),
    "barcode": (120, 120, 120),
}

GOVERNORATES: dict[str, str] = {
    "01": "القاهرة",
    "02": "الإسكندرية",
    "03": "بورسعيد",
    "04": "السويس",
    "11": "دمياط",
    "12": "الدقهلية",
    "13": "الشرقية",
    "14": "القليوبية",
    "15": "كفر الشيخ",
    "16": "الغربية",
    "17": "المنوفية",
    "18": "البحيرة",
    "19": "الإسماعيلية",
    "21": "الجيزة",
    "22": "بني سويف",
    "23": "الفيوم",
    "24": "المنيا",
    "25": "أسيوط",
    "26": "سوهاج",
    "27": "قنا",
    "28": "أسوان",
    "29": "الأقصر",
    "31": "البحر الأحمر",
    "32": "الوادي الجديد",
    "33": "مطروح",
    "34": "شمال سيناء",
    "35": "جنوب سيناء",
    "88": "خارج الجمهورية",
}

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

_DIGITS_MAP = str.maketrans(
    {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "O": "0",
        "o": "0",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "|": "1",
        "I": "1",
        "l": "1",
        "S": "5",
        "B": "8",
        "G": "6",
        "Z": "2",
    }
)

FRONT_HINTS = (
    "جمهوريه مصر",
    "بطاقه تحقيق",
    "تحقيق الشخصيه",
    "محل اقامه",
    "محل الاقامه",
)
BACK_HINTS = (
    "ساريه حتى",
    "البطاقه ساريه",
    "المهنه",
    "مهنة",
    "تخصص",
    "مهندس",
    "الحاله الاجتماعيه",
    "الحالة الاجتماعية",
)
HEADER_NOISE = (
    "جمهوريه مصر العربيه",
    "جمهورية مصر العربية",
    "بطاقه تحقيق الشخصيه",
    "بطاقة تحقيق الشخصية",
    "وزاره الداخليه",
    "وزارة الداخلية",
    "الرقم القومي",
    "محل الاقامه",
    "محل الإقامة",
)

RELIGION_VALUES = ("مسلم", "مسيحي", "مسيحى", "يهودي", "يهودى", "بدون")
MARITAL_VALUES = (
    "اعزب",
    "أعزب",
    "متزوج",
    "متزوجه",
    "متزوجة",
    "مطلق",
    "مطلقه",
    "مطلقة",
    "أرمل",
    "ارمل",
    "ارمله",
    "أرملة",
    "آنسة",
    "انسه",
    "انسة",
)
GENDER_VALUES = ("ذكر", "انثى", "أنثى", "انثي")
JOB_HINTS = (
    "مهندس",
    "طبيب",
    "محاسب",
    "محامي",
    "مدرس",
    "معلم",
    "موظف",
    "طالب",
    "ربة منزل",
    "ربه منزل",
    "تخصص",
    "عامل",
    "فني",
    "صيدلي",
    "ممرض",
    "ضابط",
    "حرفي",
    "تاجر",
    "لا يعمل",
)


def normalize_ar(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.translate(_AR_MAP)
    text = re.sub(r"\s+", " ", text)
    return text


def to_ascii_digits(text: str) -> str:
    return (text or "").translate(_DIGITS_MAP)


def ensure_yunet_model() -> Path | None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 100_000:
        return YUNET_PATH
    try:
        import ssl
        import urllib.request

        ctx = ssl._create_unverified_context()  # noqa: SLF001
        req = urllib.request.Request(YUNET_URL, headers={"User-Agent": "egyptian-id-ocr"})
        logger.info("Downloading YuNet face model…")
        with urllib.request.urlopen(req, context=ctx, timeout=180) as resp:
            data = resp.read()
        if len(data) < 50_000:
            return None
        YUNET_PATH.write_bytes(data)
        return YUNET_PATH
    except Exception:  # noqa: BLE001
        logger.exception("Failed to download YuNet")
        return None if not YUNET_PATH.exists() else YUNET_PATH


def egyptian_id_checksum_ok(nid: str) -> bool:
    """Validate 14th check digit (common Egyptian NID weight formula)."""
    nid = re.sub(r"\D", "", nid or "")
    if len(nid) != 14:
        return False
    weights = (2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
    total = sum(int(nid[i]) * weights[i] for i in range(13))
    rem = total % 11
    check = (11 - rem) % 10
    # Some issuers use 11-rem directly with 10→1 / 11→0 variants — accept close matches.
    expected = int(nid[13])
    return expected == check or expected == ((11 - rem) % 11) % 10


def decode_national_id(nid: str) -> dict[str, Any]:
    nid = re.sub(r"\D", "", to_ascii_digits(nid or ""))
    if len(nid) != 14 or nid[0] not in "23":
        return {"valid": False, "national_id": nid or None, "checksum_ok": False}

    century = 1900 if nid[0] == "2" else 2000
    year = century + int(nid[1:3])
    month = int(nid[3:5])
    day = int(nid[5:7])
    gov_code = nid[7:9]
    gender_digit = int(nid[12])
    gender = "ذكر" if gender_digit % 2 == 1 else "أنثى"

    birth_ok = 1 <= month <= 12 and 1 <= day <= 31
    # Soft year sanity
    if birth_ok and not (1900 <= year <= 2100):
        birth_ok = False
    birth_date = f"{year:04d}-{month:02d}-{day:02d}" if birth_ok else None
    checksum_ok = egyptian_id_checksum_ok(nid)

    return {
        "valid": birth_ok,
        "checksum_ok": checksum_ok,
        "national_id": nid,
        "birth_date": birth_date,
        "birth_year": year if birth_ok else None,
        "governorate_code": gov_code,
        "governorate": GOVERNORATES.get(gov_code, "غير معروف"),
        "gender": gender,
        "serial": nid[9:13],
        "check_digit": nid[13],
        # Equations used:
        "equations": {
            "century": "1900 if d0==2 else 2000",
            "birth_date": "YYYY=century+d1d2, MM=d3d4, DD=d5d6",
            "governorate": "code=d7d8 → lookup",
            "gender": "ذكر if d12 odd else أنثى",
        },
    }


def _encode_image_b64(image_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _load_bgr(path: str | Path) -> np.ndarray:
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("تعذر قراءة الصورة.")
    return img


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_card_quad(image_bgr: np.ndarray) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    scale = 900 / max(h, w)
    if scale < 1:
        small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    else:
        small = image_bgr
        scale = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    img_area = small.shape[0] * small.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.10:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2).astype("float32")
        rect = _order_points(pts)
        width = np.linalg.norm(rect[1] - rect[0])
        height = np.linalg.norm(rect[3] - rect[0])
        if height < 1 or width < 1:
            continue
        ratio = max(width, height) / min(width, height)
        if 1.30 <= ratio <= 2.05:
            return (rect / scale).astype("float32")
    return None


def warp_card(image_bgr: np.ndarray, quad: np.ndarray, out_w: int = 1200) -> np.ndarray:
    rect = _order_points(quad.astype("float32"))
    width = int(max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3])))
    height = int(max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1])))
    width = max(width, 1)
    height = max(height, 1)
    if height > width:
        rect = np.array([rect[1], rect[2], rect[3], rect[0]], dtype="float32")
        width, height = height, width

    out_h = int(out_w * height / width)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image_bgr, m, (out_w, out_h))


def enhance_id_card(image_bgr: np.ndarray) -> np.ndarray:
    """Contrast boost tailored for plastic ID print (not handwriting-heavy)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    color = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    sharp = cv2.addWeighted(color, 1.25, cv2.GaussianBlur(color, (0, 0), 1.0), -0.25, 0)
    return sharp


def prepare_zone_roi(
    roi: np.ndarray,
    *,
    mode: str = "text",
    scale: float = 1.8,
) -> list[np.ndarray]:
    """Build OCR-ready variants of a layout crop."""
    if roi is None or not roi.size:
        return []
    h, w = roi.shape[:2]
    target = max(h, w) * scale
    if target > 900:
        scale = 900 / max(h, w)
    scale = max(1.25, min(scale, 2.2))
    big = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    enhanced = enhance_id_card(big)
    if mode == "digits":
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
        )
        return [enhanced, cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)]
    # Text: enhanced only (raw+enhanced doubled OCR and fragmented Arabic)
    return [enhanced]


def auto_deskew_card(image_bgr: np.ndarray, max_angle: float = 8.0) -> np.ndarray:
    """Correct small camera tilt; skip large rotations (card already upright)."""
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) < 200:
            return image_bgr
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.4 or abs(angle) > max_angle:
            return image_bgr
        h, w = image_bgr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            image_bgr,
            m,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception:  # noqa: BLE001
        return image_bgr


def pad_zone(
    box: tuple[float, float, float, float],
    pad: float = 0.02,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (
        max(0.0, x0 - pad),
        max(0.0, y0 - pad),
        min(1.0, x1 + pad),
        min(1.0, y1 + pad),
    )


def detect_faces(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = image_bgr.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    model = ensure_yunet_model()
    if model and hasattr(cv2, "FaceDetectorYN_create"):
        try:
            detector = cv2.FaceDetectorYN_create(str(model), "", (w, h), 0.55, 0.3, 5000)
            detector.setInputSize((w, h))
            _, faces = detector.detect(image_bgr)
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = [int(v) for v in f[:4]]
                    if bw > 20 and bh > 20:
                        boxes.append((x, y, bw, bh))
        except Exception:  # noqa: BLE001
            logger.debug("YuNet failed", exc_info=True)

    if not boxes:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        for x, y, bw, bh in cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        ):
            boxes.append((int(x), int(y), int(bw), int(bh)))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def face_box_norm(image_bgr: np.ndarray) -> tuple[float, float, float, float] | None:
    """Normalized (x0,y0,x1,y1) of the personal photo (LEFT side on Egyptian ID)."""
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    faces = detect_faces(image_bgr)
    leftish: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh in faces:
        cx = (x + bw / 2) / w
        # Egyptian ID personal photo sits on the LEFT of the front
        if cx <= 0.45:
            leftish.append((x, y, bw, bh))
    if leftish:
        leftish.sort(key=lambda b: b[2] * b[3], reverse=True)
        x, y, bw, bh = leftish[0]
        return (x / w, y / h, (x + bw) / w, (y + bh) / h)
    # Fixed layout fallback from ID_ZONES
    return ID_ZONES["front"]["face"]


def crop_face(image_bgr: np.ndarray, pad: float = 0.28) -> np.ndarray | None:
    faces = detect_faces(image_bgr)
    h, w = image_bgr.shape[:2]
    chosen = None
    if faces:
        leftish = [b for b in faces if (b[0] + b[2] / 2) / max(w, 1) <= 0.45]
        pool = leftish or faces
        chosen = max(pool, key=lambda b: b[2] * b[3])
    if chosen is None:
        x0, y0, x1, y1 = ID_ZONES["front"]["face"]
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)
        roi = image_bgr[py0:py1, px0:px1]
        return roi if roi.size else None
    x, y, bw, bh = chosen
    px, py = int(bw * pad), int(bh * pad)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(w, x + bw + px)
    y1 = min(h, y + bh + py)
    return image_bgr[y0:y1, x0:x1]


def draw_id_zones(
    image_bgr: np.ndarray,
    side: str,
    *,
    thickness: int = 3,
) -> np.ndarray:
    """Draw labeled ID_ZONES boxes — same markers used for OCR crops."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    zones = ID_ZONES.get(side) or {}
    for name, (x0, y0, x1, y1) in zones.items():
        color = ID_ZONE_COLORS.get(name, (0, 255, 255))
        p0 = (int(round(x0 * w)), int(round(y0 * h)))
        p1 = (int(round(x1 * w)), int(round(y1 * h)))
        cv2.rectangle(out, p0, p1, color, thickness)
        # Filled label bar like the reference overlay
        label = f"{name} ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(th + 6, p0[1] + th + 4)
        cv2.rectangle(
            out,
            (p0[0], ty - th - 6),
            (p0[0] + tw + 8, ty + 4),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (p0[0] + 4, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


@dataclass
class OcrToken:
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
    def norm_text(self) -> str:
        return normalize_ar(self.text)


def blocks_to_tokens(blocks: list[Any], *, width: float, height: float) -> list[OcrToken]:
    tokens: list[OcrToken] = []
    for i, b in enumerate(blocks):
        text = str(getattr(b, "text", "") or "").strip()
        if not text:
            continue
        x0 = float(getattr(b, "x0", 0.0))
        y0 = float(getattr(b, "y0", 0.0))
        x1 = float(getattr(b, "x1", 0.0))
        y1 = float(getattr(b, "y1", 0.0))
        # Normalize coords 0..1 using card size when available
        if width > 1 and height > 1 and max(x1, y1) > 2:
            x0, x1 = x0 / width, x1 / width
            y0, y1 = y0 / height, y1 / height
        tokens.append(
            OcrToken(
                index=i,
                text=text,
                score=float(getattr(b, "score", 0.0) or 0.0),
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
        )
    # Reading order: top→bottom, then right→left (Arabic card)
    tokens.sort(key=lambda t: (round(t.cy, 2), -t.cx))
    for i, t in enumerate(tokens):
        t.index = i
    return tokens


def merge_token_lists(*lists: list[OcrToken]) -> list[OcrToken]:
    """Merge OCR passes: keep unique texts, prefer higher score / denser set."""
    by_key: dict[str, OcrToken] = {}
    for lst in lists:
        for t in lst:
            key = normalize_ar(t.text)
            if not key:
                continue
            prev = by_key.get(key)
            if prev is None or t.score >= prev.score:
                by_key[key] = t
    merged = list(by_key.values())
    merged.sort(key=lambda t: (round(t.cy, 2), -t.cx))
    for i, t in enumerate(merged):
        t.index = i
    return merged


def _barcode_bottom_score(image_bgr: np.ndarray) -> float:
    """High score if bottom band looks like a dense 2D barcode (back of ID)."""
    h, w = image_bgr.shape[:2]
    if h < 40 or w < 40:
        return 0.0
    roi = image_bgr[int(h * 0.55) : int(h * 0.98), int(w * 0.05) : int(w * 0.95)]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Strong local contrast / many black-white transitions → barcode
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 160)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
    # Horizontal projection variance (PDF417 has dense columns)
    col_std = float(np.std(gray.astype(np.float32), axis=0).mean())
    score = 0.0
    if edge_ratio >= 0.12:
        score += 3.0
    elif edge_ratio >= 0.07:
        score += 1.5
    if col_std >= 35:
        score += 2.0
    elif col_std >= 22:
        score += 1.0
    return score


def _face_left_score(image_bgr: np.ndarray) -> float:
    """Front of Egyptian ID has the portrait on the left."""
    h, w = image_bgr.shape[:2]
    if w <= 0:
        return 0.0
    faces = detect_faces(image_bgr)
    for x, y, bw, bh in faces:
        cx = (x + bw / 2) / w
        area = (bw * bh) / float(max(h * w, 1))
        if cx <= 0.42 and area >= 0.03:
            return 4.0
        if cx <= 0.48 and area >= 0.02:
            return 2.5
    return 0.0


def guess_side(
    tokens: list[OcrToken],
    image_bgr: np.ndarray | None = None,
) -> str:
    """Decide front vs back using strong unique phrases + visual cues."""
    texts = [t.text for t in tokens]
    blob = normalize_ar(" ".join(texts))
    front = 0.0
    back = 0.0

    # --- Strong FRONT signals ---
    if "بطاقه تحقيق" in blob or "تحقيق الشخصيه" in blob:
        front += 6.0
    if "جمهوريه مصر" in blob or ("جمهوريه" in blob and "مصر" in blob):
        front += 4.0
    if "محل اقامه" in blob or "محل الاقامه" in blob:
        front += 5.0
    # 14-digit national ID is on the front (also sometimes printed faintly on back)
    nid = find_national_id_from_tokens(tokens)
    if nid and "ساريه" not in blob:
        front += 2.5
    elif nid:
        front += 0.5

    for h in FRONT_HINTS:
        if normalize_ar(h) in blob:
            front += 1.0

    # --- Strong BACK signals ---
    if "ساريه حتى" in blob or ("ساريه" in blob and "حتى" in blob):
        back += 7.0
    if "البطاقه ساريه" in blob:
        back += 5.0
    # Do NOT treat bare «بطاقه» as front when expiry phrase exists
    if any(normalize_ar(v) in blob for v in RELIGION_VALUES):
        back += 2.5
    if any(normalize_ar(v) in blob for v in MARITAL_VALUES):
        back += 2.5
    if "ذكر" in blob or "انثى" in blob:
        back += 2.0
    if any(k in blob for k in ("مهندس", "تخصص", "المهنه", "مهنه")):
        back += 3.0
    for h in BACK_HINTS:
        if normalize_ar(h) in blob:
            back += 1.0

    # --- Visual cues (no OCR) ---
    if image_bgr is not None and image_bgr.size:
        face_s = _face_left_score(image_bgr)
        bar_s = _barcode_bottom_score(image_bgr)
        front += face_s
        back += bar_s

    # Decisive thresholds
    if back >= front + 2 and back >= 4:
        return "back"
    if front >= back + 1.5 and front >= 3:
        return "front"
    if back > front and back >= 5:
        return "back"
    if front > back and front >= 2:
        return "front"
    if back >= 4:
        return "back"
    if front >= 2 or nid:
        return "front"
    if back >= 2:
        return "back"
    return "unknown"


def _match_known_value(text: str, values: tuple[str, ...]) -> str | None:
    n = normalize_ar(text)
    for v in values:
        vn = normalize_ar(v)
        if n == vn or vn == n:
            return v
    for part in re.split(r"\s+", n):
        for v in values:
            if part == normalize_ar(v):
                return v
    return None


def extract_expiry_date(texts: list[str]) -> str | None:
    for text in texts:
        n = normalize_ar(text)
        ascii_t = to_ascii_digits(text)
        interesting = ("ساريه" in n) or ("سارية" in n) or ("حتى" in n) or ("بطاقه" in n)
        if not interesting and "بطاقة" not in text:
            continue
        m = re.search(r"(20\d{2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(\d{1,2})", ascii_t)
        if m:
            y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
            return f"{y}-{mo:02d}-{d:02d}"
        m2 = re.search(r"(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(20\d{2})", ascii_t)
        if m2:
            a, b, y = int(m2.group(1)), int(m2.group(2)), m2.group(3)
            # prefer day-month-year if first <= 31
            if a <= 31:
                return f"{y}-{b:02d}-{a:02d}"
            return f"{y}-{a:02d}-{b:02d}"
    joined_n = normalize_ar(" ".join(texts))
    joined_a = to_ascii_digits(" ".join(texts))
    if "ساري" in joined_n or "حتى" in joined_n:
        m = re.search(r"(20\d{2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(\d{1,2})", joined_a)
        if m:
            y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
            return f"{y}-{mo:02d}-{d:02d}"
    return None


def extract_back_fields(tokens: list[OcrToken]) -> tuple[dict[str, Any], list[str]]:
    """Parse Egyptian ID back layout: job + status row + expiry."""
    notes: list[str] = []
    out: dict[str, Any] = {
        "job": None,
        "gender": None,
        "religion": None,
        "marital_status": None,
        "husband_name": None,
        "expiry_date": None,
        "national_id": None,
    }

    usable = [t for t in tokens if t.cy <= 0.85]
    if not usable:
        usable = list(tokens)

    lines = _cluster_tokens_into_lines(usable, y_thresh=0.045)
    line_texts = [_merge_line_scraps(_line_text(ln)) for ln in lines]
    line_texts = [t for t in line_texts if t]
    notes.append(f"BACK_LINES={len(line_texts)}")

    nid = find_national_id_from_tokens(usable)
    if nid:
        out["national_id"] = nid
        notes.append("BACK: رقم قومي")

    exp = extract_expiry_date(line_texts)
    if exp:
        out["expiry_date"] = exp
        notes.append("BACK: تاريخ السريان")

    status_norms = {
        normalize_ar(x) for x in RELIGION_VALUES + MARITAL_VALUES + GENDER_VALUES
    }

    for t in usable:
        text = t.text.strip()
        if not out["gender"]:
            g = _match_known_value(text, GENDER_VALUES)
            if g:
                out["gender"] = "ذكر" if normalize_ar(g) == "ذكر" else "أنثى"
                notes.append("BACK: النوع")
        if not out["religion"]:
            r = _match_known_value(text, RELIGION_VALUES)
            if r:
                out["religion"] = r
                notes.append("BACK: الديانة")
        if not out["marital_status"]:
            m = _match_known_value(text, MARITAL_VALUES)
            if m:
                out["marital_status"] = m
                notes.append("BACK: الحالة الاجتماعية")

    for text in line_texts:
        if not out["gender"]:
            g = _match_known_value(text, GENDER_VALUES)
            if g:
                out["gender"] = "ذكر" if normalize_ar(g) == "ذكر" else "أنثى"
        if not out["religion"]:
            r = _match_known_value(text, RELIGION_VALUES)
            if r:
                out["religion"] = r
        if not out["marital_status"]:
            m = _match_known_value(text, MARITAL_VALUES)
            if m:
                out["marital_status"] = m

    job_candidates: list[tuple[float, str]] = []
    for ln in lines:
        if not ln:
            continue
        cy = sum(t.cy for t in ln) / len(ln)
        text = _merge_line_scraps(_line_text(ln))
        if not text:
            continue
        n = normalize_ar(text)
        if len(_digit_soup(text)) >= 10:
            continue
        if "ساريه" in n or "سارية" in n or ("حتى" in n and re.search(r"20\d{2}", to_ascii_digits(text))):
            continue
        if n in status_norms:
            continue
        parts = [p for p in re.split(r"\s+", n) if p]
        if parts and all(p in status_norms for p in parts):
            continue
        if _is_header_noise(text):
            continue
        for lbl in ("المهنه", "المهنة", "مهنة"):
            if lbl in n:
                after = n.split(lbl, 1)[-1].strip(" :：-")
                if after:
                    text = after
                    n = normalize_ar(after)
                break
        score = min(len(text), 70) / 20.0
        if any(normalize_ar(h) in n for h in JOB_HINTS):
            score += 4.0
        if cy < 0.50:
            score += 1.5
        if 0.05 <= cy <= 0.58 and _arabic_ratio(text) >= 0.55 and len(text) >= 6:
            job_candidates.append((score, text.strip()))

    if job_candidates:
        job_candidates.sort(key=lambda x: x[0], reverse=True)
        out["job"] = job_candidates[0][1]
        notes.append("BACK: المهنة")

    for i, t in enumerate(usable):
        n = t.norm_text
        if "اسم الزوج" in n or (n.startswith("زوج") and "مهن" not in n):
            after = re.split(r"الزوج|زوج", n, maxsplit=1)[-1].strip(" :：-")
            if after and len(after) >= 3 and "مهن" not in after:
                out["husband_name"] = after
            elif i + 1 < len(usable):
                cand = usable[i + 1].text.strip()
                if cand and "مهن" not in normalize_ar(cand):
                    out["husband_name"] = cand
            if out["husband_name"]:
                notes.append("BACK: اسم الزوج")
            break

    return out, notes


def _digit_soup(text: str) -> str:
    return re.sub(r"\D", "", to_ascii_digits(text))


def find_national_id_from_tokens(tokens: list[OcrToken]) -> str | None:
    # 1) Exact 14 in any token / joined
    texts = [t.text for t in tokens]
    joined = " ".join(texts)
    for m in re.findall(r"[23][\dOoIl|]{13}", to_ascii_digits(joined)):
        cand = _digit_soup(m)
        if len(cand) == 14 and cand[0] in "23":
            decoded = decode_national_id(cand)
            if decoded.get("valid"):
                return cand

    # 2) Bottom-zone tokens (NID is near bottom of front)
    bottom = [t for t in tokens if t.cy >= 0.72]
    bottom_digits = _digit_soup("".join(t.text for t in bottom))
    m = re.search(r"[23]\d{13}", bottom_digits)
    if m:
        return m.group(0)

    # 3) Sliding window over all digits
    all_digits = _digit_soup(joined)
    best = None
    best_score = -1
    for i in range(0, max(0, len(all_digits) - 13)):
        cand = all_digits[i : i + 14]
        if cand[0] not in "23":
            continue
        decoded = decode_national_id(cand)
        score = 0
        if decoded.get("valid"):
            score += 3
        if decoded.get("checksum_ok"):
            score += 2
        if GOVERNORATES.get(cand[7:9]):
            score += 1
        if score > best_score:
            best_score = score
            best = cand
    return best if best_score >= 3 else best


def _is_header_noise(text: str) -> bool:
    n = normalize_ar(text)
    if not n or len(n) < 2:
        return True
    for h in HEADER_NOISE:
        if normalize_ar(h) == n or normalize_ar(h) in n and len(n) <= len(normalize_ar(h)) + 2:
            return True
    if any(k in n for k in ("جمهوريه مصر", "بطاقه تحقيق", "وزاره الداخليه")):
        return True
    # OCR scraps of the republic / ID title line
    if n.startswith("جمهور") or "جمهور" in n:
        return True
    if "بطاق" in n and ("شخص" in n or "تحقيق" in n):
        return True
    if "وزاره" in n or "داخليه" in n:
        return True
    return False


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    ar = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return ar / max(len(text), 1)


def extract_value_after_label(
    tokens: list[OcrToken],
    labels: tuple[str, ...],
    *,
    value_hints: tuple[str, ...] | None = None,
) -> str | None:
    label_norms = tuple(normalize_ar(x) for x in labels)
    for i, t in enumerate(tokens):
        n = t.norm_text
        if not any(lbl in n for lbl in label_norms):
            continue
        # same token after label
        for lbl in label_norms:
            if lbl in n:
                after = n.split(lbl, 1)[-1].strip(" :：-–=")
                if after and len(after) >= 2 and not any(x in after for x in label_norms):
                    return after
        # nearest neighbor to the left (RTL: value often left of label) or next line
        candidates: list[tuple[float, str]] = []
        for j, u in enumerate(tokens):
            if j == i:
                continue
            un = u.norm_text
            if _is_header_noise(u.text) or any(lbl in un for lbl in label_norms):
                continue
            if _arabic_ratio(u.text) < 0.25 and not re.search(r"\d", u.text):
                continue
            dy = abs(u.cy - t.cy)
            dx = t.cx - u.cx  # positive if u is to the left of label
            if dy < 0.06 and dx > -0.05:
                score = 2.0 - dy * 10 + min(dx, 0.4)
                candidates.append((score, u.text.strip()))
            elif 0 < (u.cy - t.cy) < 0.12 and abs(u.cx - t.cx) < 0.35:
                candidates.append((1.2 - abs(u.cy - t.cy), u.text.strip()))
        if value_hints:
            for hint in value_hints:
                hn = normalize_ar(hint)
                for u in tokens:
                    if hn in u.norm_text or u.norm_text in hn:
                        return hint if hint in u.text else u.text.strip()
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
    # global hint scan
    if value_hints:
        blob_tokens = tokens
        for hint in value_hints:
            hn = normalize_ar(hint)
            for u in blob_tokens:
                if u.norm_text == hn or hn in u.norm_text:
                    return u.text.strip()
    return None


ADDRESS_HINTS = (
    "شارع",
    "ش ",
    " ش",
    "مركز",
    "قسم",
    "قريه",
    "قرية",
    "نجع",
    "عزبه",
    "عزبة",
    "محافظه",
    "محافظة",
    "مدينه",
    "مدينة",
    "بلوك",
    "عماره",
    "عمارة",
    "منزل",
    "طريق",
    "كفر",
    "دائري",
    "مساكن",
    "حي ",
    "حى ",
    "الحكماء",
    "الزقازيق",
    "المنصوره",
    "المنصورة",
)

# Common city/gov words that belong to address, never name.
ADDRESS_PLACE_WORDS = {
    normalize_ar(x)
    for x in (
        "الزقازيق",
        "القاهره",
        "القاهرة",
        "الجيزه",
        "الجيزة",
        "الاسكندريه",
        "الإسكندرية",
        "المنصوره",
        "المنصورة",
        "الشرقيه",
        "الشرقية",
        "الغربيه",
        "الغربية",
        "الدقهليه",
        "الدقهلية",
        "القليوبيه",
        "القليوبية",
        "ثان",
        "اول",
        "أول",
    )
}

ADDRESS_LABELS = (
    "محل الاقامه",
    "محل الإقامة",
    "محل اقامه",
    "محل إقامة",
    "عنوان",
    "العنوان",
)

NAME_STOP_LABELS = ADDRESS_LABELS + (
    "الرقم القومي",
    "الرقم القومى",
    "المهنه",
    "المهنة",
    "الديانه",
    "الديانة",
    "الجنس",
    "الحاله",
    "الحالة",
)


def _is_address_label(text: str) -> bool:
    n = normalize_ar(text)
    return any(normalize_ar(lbl) in n for lbl in ADDRESS_LABELS)


def _is_name_stop_label(text: str) -> bool:
    n = normalize_ar(text)
    return any(normalize_ar(lbl) in n for lbl in NAME_STOP_LABELS)


def _looks_like_address(text: str) -> bool:
    n = normalize_ar(text)
    if _is_address_label(text):
        return False
    ascii_t = to_ascii_digits(n)
    # "9 ش" / "١٢ شارع" / starts with house number + ش
    if re.search(r"(^|\s)\d+\s*ش", ascii_t):
        return True
    if re.search(r"(^|\s)\d+\s*شارع", ascii_t):
        return True
    if any(h.strip() and h.strip() in n for h in ADDRESS_HINTS):
        return True
    if n in ADDRESS_PLACE_WORDS:
        return True
    return False


def _is_strong_address_start(text: str) -> bool:
    """Only switch from name→address on clear evidence (not a 5th name part)."""
    if _is_address_label(text):
        return True
    n = normalize_ar(text)
    ascii_t = to_ascii_digits(n)
    if re.search(r"(^|\s)\d+\s*ش", ascii_t) or re.search(r"(^|\s)\d+\s*شارع", ascii_t):
        return True
    if any(
        h in n
        for h in ("شارع", "مركز", "قسم", "قرية", "قريه", "نجع", "عزبة", "عزبه", "عمارة", "عماره", "بلوك")
    ):
        return True
    return False


def _looks_like_person_name(text: str) -> bool:
    """Heuristic: Arabic name line (1–5 words), not address/place."""
    raw = (text or "").strip()
    n = normalize_ar(raw)
    if len(n) < 2 or len(n) > 45:
        return False
    if _is_header_noise(raw) or _is_name_stop_label(raw):
        return False
    if _arabic_ratio(raw) < 0.70:
        return False
    if len(_digit_soup(raw)) >= 1:
        return False
    if _is_strong_address_start(raw) or _looks_like_address(raw):
        return False
    if n in ADDRESS_PLACE_WORDS:
        return False
    if n in {normalize_ar(x) for x in RELIGION_VALUES + MARITAL_VALUES + GENDER_VALUES}:
        return False
    if any(k in n for k in ("جمهوريه", "جمهور", "بطاقه", "وزاره", "تحقيق", "شخصيه", "قومي", "اقامه")):
        return False
    words = [w for w in re.split(r"\s+", n) if w]
    if not (1 <= len(words) <= 5):
        return False
    # Reject very short OCR scraps like "قية" / "شر"
    if len(words) == 1 and len(words[0]) < 2:
        return False
    return True


def _strip_address_label_prefix(text: str) -> str:
    out = (text or "").strip()
    labels = (
        "محل الإقامة",
        "محل الاقامة",
        "محل إقامة",
        "محل اقامه",
        "العنوان",
        "عنوان",
    )
    for lbl in labels:
        pattern = rf"^\s*{re.escape(lbl)}\s*[:：\-–]?\s*"
        out2 = re.sub(pattern, "", out)
        if out2 != out:
            out = out2
            break
        n_out = normalize_ar(out)
        n_lbl = normalize_ar(lbl)
        if n_out.startswith(n_lbl):
            out = (
                out[len(lbl) :].lstrip(" :：-–")
                if out.startswith(lbl)
                else re.sub(re.escape(lbl), "", out, count=1).strip(" :：-–")
            )
            break
    return out.strip(" :：-–=")


def _is_ocr_scrap(text: str, existing: list[str]) -> bool:
    """Drop fragmented OCR pieces already covered by a longer line."""
    n = normalize_ar(text)
    if not n:
        return True
    if len(n) <= 2 and not _digit_soup(n):
        return True
    for prev in existing:
        pn = normalize_ar(prev)
        if not pn:
            continue
        if n == pn:
            return True
        # "شر" / "قية" inside "الشرقية", "ثان" inside "الزقازيق ثان"
        if len(n) <= 5 and n in pn:
            return True
        if len(pn) <= 5 and pn in n and n != pn:
            # keep longer; mark shorter later — here if current is longer, ok
            continue
    return False


def _clean_address_parts(parts: list[str]) -> list[str]:
    cleaned: list[str] = []
    for p in parts:
        p = _strip_address_label_prefix(p).strip()
        if not p:
            continue
        if _is_ocr_scrap(p, cleaned):
            continue
        # If a previous short scrap is contained in this longer line, drop the scrap
        cleaned = [c for c in cleaned if not (len(normalize_ar(c)) <= 5 and normalize_ar(c) in normalize_ar(p) and normalize_ar(c) != normalize_ar(p))]
        if normalize_ar(p) in {normalize_ar(c) for c in cleaned}:
            continue
        cleaned.append(p)
    return cleaned


def _cluster_tokens_into_lines(
    tokens: list[OcrToken],
    *,
    y_thresh: float = 0.035,
) -> list[list[OcrToken]]:
    """Group OCR tokens into visual lines by vertical center, then RTL within line."""
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda t: t.cy)
    lines: list[list[OcrToken]] = []
    for t in ordered:
        placed = False
        for line in lines:
            # Compare to line mean cy
            mean_cy = sum(x.cy for x in line) / len(line)
            mean_h = sum(abs(x.y1 - x.y0) for x in line) / len(line)
            thr = max(y_thresh, mean_h * 0.65)
            if abs(t.cy - mean_cy) <= thr:
                line.append(t)
                placed = True
                break
        if not placed:
            lines.append([t])
    # Sort lines top→bottom; within each line right→left (Arabic visual)
    lines.sort(key=lambda ln: sum(t.cy for t in ln) / len(ln))
    for ln in lines:
        ln.sort(key=lambda t: t.cx, reverse=True)
    return lines


def _line_text(line: list[OcrToken]) -> str:
    parts = [t.text.strip() for t in line if t.text and t.text.strip()]
    return " ".join(parts).strip()


def _merge_line_scraps(text: str) -> str:
    """Clean duplicated/fragmented OCR inside one line."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.replace(" - ", " - ").replace("–", "-")
    words = text.split(" ")
    kept: list[str] = []
    for w in words:
        nw = normalize_ar(w)
        if not nw:
            continue
        # drop short scrap already covered by previous/next longer word
        if len(nw) <= 3 and any(nw in normalize_ar(k) and nw != normalize_ar(k) for k in kept):
            continue
        # drop if previous word contains this or vice-versa duplicate
        if kept and normalize_ar(kept[-1]) == nw:
            continue
        if kept and len(nw) <= 4 and nw in normalize_ar(kept[-1]):
            continue
        if kept and len(normalize_ar(kept[-1])) <= 4 and normalize_ar(kept[-1]) in nw:
            kept[-1] = w
            continue
        kept.append(w)
    # Normalize hyphen spacing
    out = " ".join(kept)
    out = re.sub(r"\s*-\s*", " - ", out)
    out = re.sub(r"\s+", " ", out).strip(" -")
    return out


def _score_name_line(text: str, cy: float) -> float:
    if not text or _is_header_noise(text) or _is_address_label(text):
        return -10.0
    if _is_strong_address_start(text) or (
        _looks_like_address(text) and not _looks_like_person_name(text)
    ):
        return -5.0
    if len(_digit_soup(text)) >= 1:
        return -3.0
    n = normalize_ar(text)
    score = 0.0
    if _looks_like_person_name(text):
        score += 4.0
    if _arabic_ratio(text) >= 0.75:
        score += 1.5
    words = [w for w in re.split(r"\s+", n) if w]
    if len(words) == 1 and 2 <= len(words[0]) <= 12:
        score += 2.0  # first-name line
    if 2 <= len(words) <= 5:
        score += 2.5  # remaining name line
    if 0.12 <= cy <= 0.48:
        score += 2.0
    elif cy > 0.55:
        score -= 2.0
    return score


def _score_address_line(text: str, cy: float) -> float:
    if not text or _is_header_noise(text):
        return -10.0
    rest = _strip_address_label_prefix(text)
    if _is_address_label(text) and len(normalize_ar(rest)) < 3:
        return 0.5  # label only
    n = normalize_ar(rest)
    digits = _digit_soup(rest)
    # Pure / mostly numeric OCR noise (years, serials) is not an address
    if len(digits) >= 6 and _arabic_ratio(rest) < 0.35:
        return -8.0
    if len(digits) >= 10 and "ش" not in n and not any(
        normalize_ar(p) in n for p in ADDRESS_PLACE_WORDS
    ):
        return -8.0
    score = 0.0
    if _is_strong_address_start(rest) or _is_strong_address_start(text):
        score += 5.0
    if _looks_like_address(rest):
        score += 3.0
    if "-" in text or "–" in text:
        score += 1.5
    if any(normalize_ar(p) in n for p in ADDRESS_PLACE_WORDS):
        score += 2.0
    if len(digits) >= 1 and "ش" in n:
        score += 3.0
    if _looks_like_person_name(rest) and not _looks_like_address(rest):
        score -= 4.0
    if 0.42 <= cy <= 0.80:
        score += 2.0
    if cy < 0.30:
        score -= 2.0
    if _arabic_ratio(rest) >= 0.4:
        score += 0.5
    return score


def extract_name_and_address(
    tokens: list[OcrToken],
    side: str,
    *,
    face_box: tuple[float, float, float, float] | None = None,
) -> tuple[str | None, str | None, list[str]]:
    """Egyptian ID front: exactly 2 name lines + 2 address lines.

    Ground truth:
      name L1: محمود
      name L2: عصام عبدالعزيز قطب محمد
      addr L1: ٩ ش عبدالقادر - الحكماء
      addr L2: الزقازيق ثان - الشرقية
    """
    notes: list[str] = []
    if side == "back":
        return None, None, notes

    # Drop tokens whose center sits inside the personal photo bbox.
    # If filtering wipes the text column (bad/oversized face), keep all tokens.
    if face_box is not None:
        fx0, fy0, fx1, fy1 = face_box
        notes.append(f"FACE_BOX=({fx0:.2f},{fy0:.2f})-({fx1:.2f},{fy1:.2f})")
        filtered = [
            t
            for t in tokens
            if not (fx0 <= t.cx <= fx1 and fy0 <= t.cy <= fy1)
        ]
        useful = [
            t
            for t in filtered
            if _arabic_ratio(t.text) >= 0.5 and len(_digit_soup(t.text)) < 8
        ]
        if len(useful) >= 3:
            col = filtered
        else:
            notes.append("FACE_FILTER_SKIP: عمود النص اختفى")
            col = list(tokens)
    else:
        notes.append("FACE_BOX=none")
        col = list(tokens)
    if len(col) < 3:
        col = list(tokens)

    lines = _cluster_tokens_into_lines(col, y_thresh=0.045)
    line_meta: list[tuple[float, str, list[OcrToken]]] = []
    for ln in lines:
        raw = _line_text(ln)
        if not raw:
            continue
        cy = sum(t.cy for t in ln) / len(ln)
        if len(_digit_soup(raw)) >= 10 and _arabic_ratio(raw) < 0.35:
            continue
        if _is_header_noise(raw):
            continue
        text = _merge_line_scraps(raw)
        if not text:
            continue
        line_meta.append((cy, text, ln))

    notes.append(f"LINES={len(line_meta)}")
    if not line_meta:
        return None, None, notes

    # Score every line for name vs address
    scored = []
    for cy, text, ln in line_meta:
        scored.append(
            {
                "cy": cy,
                "text": text,
                "name_s": _score_name_line(text, cy),
                "addr_s": _score_address_line(text, cy),
                "tokens": ln,
            }
        )

    # Prefer vertical order: first name-like pair, then address-like pair
    name_idxs: list[int] = []
    for i, row in enumerate(scored):
        if row["cy"] < 0.10:
            continue
        if row["name_s"] >= 3.0 and row["name_s"] > row["addr_s"]:
            name_idxs.append(i)
            if i + 1 < len(scored):
                nxt = scored[i + 1]
                if (
                    nxt["name_s"] >= 2.0
                    and nxt["name_s"] >= nxt["addr_s"]
                    and not _is_strong_address_start(nxt["text"])
                    and not _is_address_label(nxt["text"])
                ):
                    name_idxs.append(i + 1)
            break

    # If only one name line found, try pull previous short first-name
    if len(name_idxs) == 1 and name_idxs[0] > 0:
        prev = scored[name_idxs[0] - 1]
        if prev["name_s"] >= 2.5 and len(normalize_ar(prev["text"]).split()) <= 2:
            name_idxs = [name_idxs[0] - 1, name_idxs[0]]

    # If still empty: take the top-most 1–2 Arabic non-address lines
    if not name_idxs:
        for i, row in enumerate(scored):
            if row["cy"] < 0.12 or row["cy"] > 0.55:
                continue
            if row["addr_s"] > row["name_s"] and row["addr_s"] >= 3.0:
                continue
            if _arabic_ratio(row["text"]) >= 0.7 and len(_digit_soup(row["text"])) == 0:
                name_idxs.append(i)
            if len(name_idxs) >= 2:
                break

    name_idxs = name_idxs[:2]
    name_lines = [scored[i]["text"] for i in name_idxs]

    # Address: next 2 address-like lines in reading order (not score-reshuffled scraps)
    start_i = (max(name_idxs) + 1) if name_idxs else 0
    addr_lines: list[str] = []
    for i in range(start_i, len(scored)):
        row = scored[i]
        text = row["text"]
        if _is_address_label(text) and len(_strip_address_label_prefix(text)) < 3:
            notes.append("LINE: لابل محل الإقامة")
            continue
        rest = _merge_line_scraps(_strip_address_label_prefix(text))
        if not rest:
            continue
        if any(normalize_ar(rest) == normalize_ar(n) for n in name_lines):
            continue
        if len(_digit_soup(rest)) >= 10 and _arabic_ratio(rest) < 0.4:
            break
        s = _score_address_line(rest, row["cy"])
        if s >= 2.0 or _is_strong_address_start(rest) or _looks_like_address(rest):
            addr_lines.append(rest)
        if len(addr_lines) >= 2:
            break
    addr_lines = _clean_address_parts(addr_lines)[:2]

    # Fallbacks
    if not name_lines:
        name_like = [
            r for r in scored if r["name_s"] >= 2.5 and r["name_s"] >= r["addr_s"]
        ]
        name_lines = [r["text"] for r in name_like[:2]]
        notes.append("FALLBACK: أعلى سطور الاسم درجة")

    if name_lines and not addr_lines:
        after = False
        tmp = []
        for r in scored:
            if r["text"] in name_lines:
                after = True
                continue
            if not after:
                continue
            rest = _merge_line_scraps(_strip_address_label_prefix(r["text"]))
            if rest and _score_address_line(rest, r["cy"]) >= 1.5:
                tmp.append(rest)
            if len(tmp) >= 2:
                break
        addr_lines = _clean_address_parts(tmp)[:2]
        if addr_lines:
            notes.append("FALLBACK: عنوان بعد سطور الاسم")

    # Egyptian ID front is typically exactly 2 + 2
    name_lines = name_lines[:2]
    addr_lines = addr_lines[:2]

    full_name = " ".join(name_lines) if name_lines else None
    address = " ".join(addr_lines) if addr_lines else None

    if full_name:
        notes.append(f"NAME← {' || '.join(name_lines)}")
    if address:
        notes.append(f"ADDR← {' || '.join(addr_lines)}")

    return full_name, address, notes


def _line_quality(text: str, *, kind: str) -> float:
    if not text:
        return -1.0
    n = normalize_ar(text)
    words = [w for w in re.split(r"\s+", n) if w]
    score = float(len(n)) * 0.15 + float(len(words))
    if kind == "name":
        if _is_header_noise(text) or _is_strong_address_start(text):
            return -5.0
        if _looks_like_person_name(text):
            score += 4.0
        if _arabic_ratio(text) >= 0.7:
            score += 2.0
        if len(_digit_soup(text)) >= 1:
            score -= 3.0
    else:
        if _is_header_noise(text):
            return -5.0
        if _looks_like_address(text) or _is_strong_address_start(text):
            score += 4.0
        if "-" in text or "–" in text:
            score += 1.5
        if _arabic_ratio(text) >= 0.5:
            score += 1.5
        if any(normalize_ar(p) in n for p in ADDRESS_PLACE_WORDS):
            score += 2.0
    return score


def _lines_from_zone_ocr(raw_lines: list[str], *, kind: str) -> list[str]:
    """Pick up to 2 cleaned lines from a dedicated name/address zone OCR."""
    candidates: list[tuple[float, int, str]] = []
    for i, ln in enumerate(raw_lines):
        if kind == "name":
            if _is_header_noise(ln) or _is_address_label(ln):
                continue
            if _is_strong_address_start(ln):
                break
            if len(_digit_soup(ln)) >= 6:
                continue
            cleaned = _merge_line_scraps(ln)
            if not cleaned or len(normalize_ar(cleaned)) < 2:
                continue
            if _arabic_ratio(cleaned) < 0.55:
                continue
            candidates.append((_line_quality(cleaned, kind="name"), i, cleaned))
        else:
            if _is_address_label(ln) and len(_strip_address_label_prefix(ln)) < 3:
                continue
            rest = _merge_line_scraps(_strip_address_label_prefix(ln))
            if not rest or _is_header_noise(rest):
                continue
            if len(_digit_soup(rest)) >= 10 and _arabic_ratio(rest) < 0.35:
                continue
            if _arabic_ratio(rest) < 0.35 and not _digit_soup(rest):
                continue
            candidates.append((_line_quality(rest, kind="address"), i, rest))

    # Prefer reading order among good lines (not only top score)
    good = [c for c in candidates if c[0] >= 1.5] or candidates
    good.sort(key=lambda x: x[1])
    out: list[str] = []
    for _, _, text in good:
        if any(normalize_ar(text) == normalize_ar(o) for o in out):
            continue
        out.append(text)
        if len(out) >= 2:
            break
    if kind == "address":
        out = _clean_address_parts(out)[:2]
    return out[:2]


def pick_best_front_zones(
    engine: Any,
    card: np.ndarray,
    zone_lines_fn: Any,
    *,
    face_box: tuple[float, float, float, float] | None = None,
) -> tuple[str | None, str | None, list[str]]:
    """OCR ID_ZONES name/address (slight pad for reading; display boxes stay exact)."""
    _ = face_box
    notes: list[str] = []
    nx0, ny0, nx1, ny1 = pad_zone(ID_ZONES["front"]["name"], 0.015)
    ax0, ay0, ax1, ay1 = pad_zone(ID_ZONES["front"]["address"], 0.015)

    notes.append(
        f"ZONE_NAME=exact{ID_ZONES['front']['name']} ocr({nx0:.2f},{ny0:.2f})-({nx1:.2f},{ny1:.2f})"
    )
    notes.append(
        f"ZONE_ADDR=exact{ID_ZONES['front']['address']} ocr({ax0:.2f},{ay0:.2f})-({ax1:.2f},{ay1:.2f})"
    )

    # One solid scale keeps Arabic words intact; padding helps edge glyphs
    name_raw = zone_lines_fn(engine, card, ny0, ny1, nx0, nx1, scale=1.7)
    addr_raw = zone_lines_fn(engine, card, ay0, ay1, ax0, ax1, scale=1.7)

    name_lines = _lines_from_zone_ocr(name_raw, kind="name")
    addr_lines = _lines_from_zone_ocr(addr_raw, kind="address")
    if name_lines:
        addr_lines = [
            a
            for a in addr_lines
            if not any(normalize_ar(a) == normalize_ar(n) for n in name_lines)
        ][:2]

    name = " ".join(name_lines) if name_lines else None
    addr = " ".join(addr_lines) if addr_lines else None
    if name:
        notes.append(f"BAND_NAME← {' || '.join(name_lines)}")
    if addr:
        notes.append(f"BAND_ADDR← {' || '.join(addr_lines)}")
    return name, addr, notes


def extract_name_from_zones(tokens: list[OcrToken], side: str) -> str | None:
    name, _, _ = extract_name_and_address(tokens, side)
    return name


def extract_address_from_zones(
    tokens: list[OcrToken], name: str | None, side: str
) -> str | None:
    _, address, _ = extract_name_and_address(tokens, side)
    return address


def apply_field_rules(
    tokens: list[OcrToken],
    side: str,
    *,
    face_box: tuple[float, float, float, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Map OCR tokens → fields using conditions + national-ID equations."""
    rules_fired: list[str] = []
    fields: dict[str, Any] = {
        "full_name": None,
        "national_id": None,
        "address": None,
        "birth_date": None,
        "governorate": None,
        "gender": None,
        "job": None,
        "religion": None,
        "marital_status": None,
        "husband_name": None,
        "expiry_date": None,
        "card_side": "الوجه" if side == "front" else "الظهر" if side == "back" else "غير محدد",
    }

    nid = find_national_id_from_tokens(tokens)
    decoded = decode_national_id(nid) if nid else {"valid": False, "checksum_ok": False}

    # RULE 1: national ID + derived fields from equations
    if nid and decoded.get("valid"):
        fields["national_id"] = decoded["national_id"]
        fields["birth_date"] = decoded.get("birth_date")
        fields["governorate"] = decoded.get("governorate")
        fields["gender"] = decoded.get("gender")
        rules_fired.append(
            "EQ: من الرقم القومي → تاريخ الميلاد + النوع + المحافظة"
            + (" + checksum✓" if decoded.get("checksum_ok") else "")
        )
    elif nid:
        fields["national_id"] = nid
        rules_fired.append("COND: عُثر على 14 رقم لكن فك التاريخ فشل")

    # FRONT: name + address
    if side != "back":
        name, address, seq_notes = extract_name_and_address(
            tokens, side, face_box=face_box
        )
        if name:
            fields["full_name"] = name
            rules_fired.append("SEQ: الاسم من سطور OCR بعد الهيدر")
        if address:
            fields["address"] = address
            rules_fired.append("SEQ: العنوان/محل الإقامة")
        rules_fired.extend(seq_notes)

    # BACK: job / religion / marital / gender / expiry
    if side == "back" or side == "unknown":
        back_fields, back_notes = extract_back_fields(tokens)
        for k, v in back_fields.items():
            if not v:
                continue
            if k == "national_id" and fields.get("national_id"):
                continue
            if k == "gender" and fields.get("gender"):
                # keep equation gender if present; note OCR
                if normalize_ar(str(fields["gender"])) != normalize_ar(str(v)):
                    rules_fired.append("BACK: نوع OCR مختلف عن معادلة الرقم — اعتُمدت المعادلة")
                continue
            fields[k] = v
        rules_fired.extend(back_notes)
        if side == "unknown" and (
            back_fields.get("job")
            or back_fields.get("expiry_date")
            or back_fields.get("marital_status")
        ):
            fields["card_side"] = "الظهر"
            side = "back"
            rules_fired.append("DETECT: اعتُبرت ظهر البطاقة")

    # Labeled fallbacks — only for the active side (avoid front/back bleed)
    if side == "back":
        if not fields.get("job"):
            job = extract_value_after_label(tokens, ("المهنه", "المهنة", "مهنة"))
            if job:
                fields["job"] = job
                rules_fired.append("LABEL: المهنة")

        if not fields.get("religion"):
            religion = extract_value_after_label(
                tokens, ("الديانه", "الديانة", "ديانة"), value_hints=RELIGION_VALUES
            )
            if religion:
                fields["religion"] = religion
                rules_fired.append("LABEL/HINT: الديانة")

        if not fields.get("marital_status"):
            marital = extract_value_after_label(
                tokens,
                ("الحاله الاجتماعيه", "الحالة الاجتماعية", "الحاله", "الحالة", "اجتماعية"),
                value_hints=MARITAL_VALUES,
            )
            if marital:
                fields["marital_status"] = marital
                rules_fired.append("LABEL/HINT: الحالة الاجتماعية")

        if not fields.get("gender"):
            gender_ocr = extract_value_after_label(
                tokens, ("الجنس",), value_hints=GENDER_VALUES
            )
            if gender_ocr:
                fields["gender"] = (
                    "ذكر" if normalize_ar(gender_ocr) == "ذكر" else "أنثى"
                )
                rules_fired.append("LABEL: الجنس")

        if not fields.get("husband_name"):
            husband = extract_value_after_label(tokens, ("اسم الزوج", "الزوج", "زوج"))
            if husband and "مهن" not in normalize_ar(husband):
                fields["husband_name"] = husband
                rules_fired.append("LABEL: اسم الزوج")

        if not fields.get("expiry_date"):
            exp = extract_expiry_date([t.text for t in tokens])
            if exp:
                fields["expiry_date"] = exp
                rules_fired.append("BACK: تاريخ السريان")

        if not fields.get("marital_status"):
            for t in tokens:
                m = _match_known_value(t.text, MARITAL_VALUES)
                if m:
                    fields["marital_status"] = m
                    rules_fired.append("HINT: حالة اجتماعية")
                    break

        if not fields.get("religion"):
            for t in tokens:
                r = _match_known_value(t.text, RELIGION_VALUES)
                if r:
                    fields["religion"] = r
                    rules_fired.append("HINT: ديانة")
                    break

    # Clear opposite-side fields so JSON/UI never mix
    if side == "front":
        for k in (
            "job",
            "religion",
            "marital_status",
            "husband_name",
            "expiry_date",
        ):
            fields[k] = None
        fields["card_side"] = "الوجه"
    elif side == "back":
        for k in ("full_name", "address"):
            fields[k] = None
        # Keep national_id/derived if found on back; card_side = ظهر
        fields["card_side"] = "الظهر"

    return fields, decoded, rules_fired


@dataclass
class EgyptianIdResult:
    side: str
    fields: dict[str, Any] = field(default_factory=dict)
    decoded: dict[str, Any] = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)
    ocr_list: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    message: str = ""
    rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "fields": self.fields,
            "decoded": self.decoded,
            "images": self.images,
            "ocr_list": self.ocr_list,
            "raw_text": self.raw_text,
            "message": self.message,
            "rules": self.rules,
        }


class EgyptianIdExtractor:
    """Local PaddleOCR + OpenCV + rule/equation field mapper."""

    def __init__(self) -> None:
        ensure_yunet_model()

    def _ocr_card_passes(
        self,
        engine: Any,
        card: np.ndarray,
        *,
        enhance_handwriting: bool,
        face_box: tuple[float, float, float, float] | None = None,
    ) -> list[OcrToken]:
        _ = face_box  # reserved: zones removed (they fragmented Arabic lines)
        h, w = card.shape[:2]
        blocks_a = engine.ocr_bgr(card, enhance=False)
        tokens = blocks_to_tokens(blocks_a, width=w, height=h)

        # Extra ROI OCR for bottom strip (national ID) — often missed
        nx0, ny0, nx1, ny1 = ID_ZONES["front"]["national_id"]
        y0, y1 = int(h * ny0), int(h * ny1)
        strip = card[y0:y1, int(w * nx0) : int(w * nx1)]
        if strip.size:
            strip_big = cv2.resize(strip, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
            blocks_c = engine.ocr_bgr(strip_big, enhance=enhance_handwriting)
            # Remap y into full-card normalized space roughly
            tokens_c: list[OcrToken] = []
            sh, sw = strip_big.shape[:2]
            for i, b in enumerate(blocks_c):
                text = str(b.text or "").strip()
                if not text:
                    continue
                tokens_c.append(
                    OcrToken(
                        index=1000 + i,
                        text=text,
                        score=float(b.score or 0),
                        x0=nx0 + (b.x0 / max(sw, 1)) * (nx1 - nx0),
                        y0=ny0 + (b.y0 / max(sh, 1)) * (ny1 - ny0),
                        x1=nx0 + (b.x1 / max(sw, 1)) * (nx1 - nx0),
                        y1=ny0 + (b.y1 / max(sh, 1)) * (ny1 - ny0),
                    )
                )
            tokens = merge_token_lists(tokens, tokens_c)

        # Name/address come from the full-card pass (zones fragmented Arabic lines).
        return tokens

    @staticmethod
    def _zone_tokens(
        engine: Any,
        card: np.ndarray,
        y0r: float,
        y1r: float,
        x0r: float,
        x1r: float,
        *,
        index_base: int,
        scale: float = 1.8,
        mode: str = "text",
    ) -> list[OcrToken]:
        h, w = card.shape[:2]
        y0, y1 = int(h * y0r), int(h * y1r)
        x0, x1 = int(w * x0r), int(w * x1r)
        roi = card[y0:y1, x0:x1]
        if not roi.size:
            return []
        variants = prepare_zone_roi(roi, mode=mode, scale=scale)
        out: list[OcrToken] = []
        for vi, big in enumerate(variants):
            blocks = engine.ocr_bgr(big, enhance=False)
            nh, nw = big.shape[:2]
            for i, b in enumerate(blocks):
                text = str(b.text or "").strip()
                if not text:
                    continue
                out.append(
                    OcrToken(
                        index=index_base + vi * 100 + i,
                        text=text,
                        score=float(b.score or 0),
                        x0=x0r + (b.x0 / max(nw, 1)) * (x1r - x0r),
                        y0=y0r + (b.y0 / max(nh, 1)) * (y1r - y0r),
                        x1=x0r + (b.x1 / max(nw, 1)) * (x1r - x0r),
                        y1=y0r + (b.y1 / max(nh, 1)) * (y1r - y0r),
                    )
                )
        return merge_token_lists(out) if out else []

    def _zone_lines(
        self,
        engine: Any,
        card: np.ndarray,
        y0r: float,
        y1r: float,
        x0r: float,
        x1r: float,
        *,
        scale: float = 1.8,
        mode: str = "text",
    ) -> list[str]:
        toks = self._zone_tokens(
            engine,
            card,
            y0r,
            y1r,
            x0r,
            x1r,
            index_base=0,
            scale=scale,
            mode=mode,
        )
        lines = _cluster_tokens_into_lines(toks, y_thresh=0.08)
        out: list[str] = []
        for ln in lines:
            text = _merge_line_scraps(_line_text(ln))
            if text:
                out.append(text)
        return out

    def process_image_path(
        self,
        image_path: str | Path,
        *,
        enhance_handwriting: bool = False,
    ) -> EgyptianIdResult:
        from ocr_engine import ArabicOcrEngine

        original = _load_bgr(image_path)
        preview = original.copy()
        ph, pw = preview.shape[:2]
        if max(ph, pw) > 1200:
            s = 1200 / max(ph, pw)
            preview = cv2.resize(preview, (int(pw * s), int(ph * s)))

        card = original
        quad = detect_card_quad(original)
        if quad is not None:
            try:
                card = warp_card(original, quad)
            except Exception:  # noqa: BLE001
                logger.debug("warp failed", exc_info=True)
                card = original
        card = auto_deskew_card(card)

        # PaddleOCR can AV-crash on very large ID photos on some Windows setups;
        # keep a display card, OCR a capped working copy.
        ocr_card = card
        ch, cw = card.shape[:2]
        # ~1000px keeps Arabic lines intact; larger often fragments words.
        max_side = 1000
        if max(ch, cw) > max_side:
            scale = max_side / max(ch, cw)
            ocr_card = cv2.resize(
                card,
                (int(cw * scale), int(ch * scale)),
                interpolation=cv2.INTER_AREA,
            )
        # Mild global enhance helps faint black print over pyramid background
        ocr_card_enh = enhance_id_card(ocr_card)

        face_box = face_box_norm(ocr_card)
        engine = ArabicOcrEngine.get()
        tokens = self._ocr_card_passes(
            engine,
            ocr_card,
            enhance_handwriting=enhance_handwriting,
            face_box=face_box,
        )

        side = guess_side(tokens, ocr_card)
        rules_side = [f"SIDE_DETECT: {side}"]
        fields, decoded, rules = apply_field_rules(
            tokens, side, face_box=face_box if side != "back" else None
        )
        rules = rules_side + rules

        # FRONT: name/address from layout zones (enhanced card for faint ink)
        if side != "back":
            z_name, z_addr, z_notes = pick_best_front_zones(
                engine, ocr_card_enh, self._zone_lines, face_box=face_box
            )
            rules.extend(z_notes)
            if z_name:
                fields["full_name"] = z_name
                rules.append("ZONE_EXACT: الاسم من منطقة name في ID_ZONES")
            if z_addr:
                fields["address"] = z_addr
                rules.append("ZONE_EXACT: العنوان من منطقة address في ID_ZONES")

            # If enhanced crop missed a field, retry once on raw ocr_card
            if not fields.get("full_name") or not fields.get("address"):
                n2, a2, notes2 = pick_best_front_zones(
                    engine, ocr_card, self._zone_lines, face_box=face_box
                )
                rules.extend(notes2)
                if n2 and not fields.get("full_name"):
                    fields["full_name"] = n2
                if a2 and (
                    not fields.get("address")
                    or _line_quality(a2, kind="address")
                    > _line_quality(str(fields.get("address") or ""), kind="address")
                ):
                    fields["address"] = a2

            # Dedicated NID digits pass if missing
            if not fields.get("national_id"):
                nx0, ny0, nx1, ny1 = pad_zone(ID_ZONES["front"]["national_id"], 0.02)
                nid_lines = self._zone_lines(
                    engine, ocr_card_enh, ny0, ny1, nx0, nx1, scale=2.0, mode="digits"
                )
                nid = find_national_id_from_tokens(
                    [
                        OcrToken(9000 + i, ln, 0.9, nx0, ny0, nx1, ny1)
                        for i, ln in enumerate(nid_lines)
                    ]
                )
                if nid:
                    fields["national_id"] = nid
                    decoded2 = decode_national_id(nid)
                    if decoded2.get("valid"):
                        decoded = decoded2
                        fields["birth_date"] = decoded2.get("birth_date")
                        fields["governorate"] = decoded2.get("governorate")
                        fields["gender"] = decoded2.get("gender")
                    rules.append("ZONE_NID: الرقم القومي من منطقة national_id")

        # Back-side dedicated zones (job / status / expiry) — above barcode
        if side == "back":
            jx0, jy0, jx1, jy1 = pad_zone(ID_ZONES["back"]["job"], 0.02)
            sx0, sy0, sx1, sy1 = pad_zone(ID_ZONES["back"]["status"], 0.02)
            ex0, ey0, ex1, ey1 = pad_zone(ID_ZONES["back"]["expiry"], 0.02)
            job_zone = self._zone_lines(engine, ocr_card_enh, jy0, jy1, jx0, jx1)
            status_zone = self._zone_lines(engine, ocr_card_enh, sy0, sy1, sx0, sx1)
            expiry_zone = self._zone_lines(engine, ocr_card_enh, ey0, ey1, ex0, ex1)
            zone_toks = merge_token_lists(
                tokens,
                self._zone_tokens(engine, ocr_card_enh, jy0, jy1, jx0, jx1, index_base=4000),
                self._zone_tokens(engine, ocr_card_enh, sy0, sy1, sx0, sx1, index_base=5000),
                self._zone_tokens(engine, ocr_card_enh, ey0, ey1, ex0, ex1, index_base=6000),
            )
            back2, notes2 = extract_back_fields(zone_toks)
            for k, v in back2.items():
                if v and (not fields.get(k) or k in {"job", "expiry_date", "marital_status", "religion"}):
                    if k == "gender" and fields.get("gender"):
                        continue
                    if k == "national_id" and fields.get("national_id"):
                        continue
                    fields[k] = v
            rules.extend(notes2)
            rules.append("ZONE_BACK: قراءة مناطق الظهر (مهنة/حالة/سريان)")
            for ln in job_zone:
                n = normalize_ar(ln)
                if any(normalize_ar(h) in n for h in JOB_HINTS) and len(ln) >= 8:
                    fields["job"] = ln
                    break
            if not fields.get("expiry_date"):
                exp = extract_expiry_date(expiry_zone + status_zone)
                if exp:
                    fields["expiry_date"] = exp
            # Parse status line for gender/religion/marital if missing
            status_blob = normalize_ar(" ".join(status_zone))
            if not fields.get("gender"):
                if "ذكر" in status_blob:
                    fields["gender"] = "ذكر"
                elif "انثى" in status_blob:
                    fields["gender"] = "أنثى"
            if not fields.get("religion"):
                for r in RELIGION_VALUES:
                    if normalize_ar(r) in status_blob:
                        fields["religion"] = r
                        break
            if not fields.get("marital_status"):
                for m in MARITAL_VALUES:
                    if normalize_ar(m) in status_blob:
                        fields["marital_status"] = m
                        break

        ocr_list = [
            {
                "index": t.index,
                "text": t.text,
                "score": round(t.score, 4),
                "bbox_norm": [round(t.x0, 4), round(t.y0, 4), round(t.x1, 4), round(t.y1, 4)],
                "page": 1,
            }
            for t in tokens
        ]
        raw_text = "\n".join(t.text for t in tokens)

        face = crop_face(card)
        images = {
            "original": _encode_image_b64(preview, quality=80),
            "card": _encode_image_b64(card, quality=88),
        }
        crop_pack = build_field_crops(card, tokens, fields, face, side=side)
        if crop_pack.get("face"):
            images["face"] = crop_pack["face"]
        elif face is not None and face.size:
            images["face"] = _encode_image_b64(face, quality=90)
        if crop_pack.get("annotated"):
            images["annotated"] = crop_pack["annotated"]
        images["crops"] = crop_pack.get("crops") or {}
        images["crop_meta"] = crop_pack.get("crop_meta") or {}

        filled = sum(1 for k, v in fields.items() if v and k != "card_side")
        crop_n = len(images["crops"]) + (1 if images.get("face") else 0)
        message = (
            f"تم ضبط {filled} حقل · {crop_n} قصّة من الصورة ({fields.get('card_side')})."
            if filled
            else "القراءة ضعيفة — صوّر البطاقة بشكل أوضح أو ارفع الوجه الآخر."
        )

        return EgyptianIdResult(
            side=side,
            fields=fields,
            decoded=decoded,
            images=images,
            ocr_list=ocr_list,
            raw_text=raw_text,
            message=message,
            rules=rules,
        )


def tokens_for_value(tokens: list[OcrToken], value: str | None) -> list[OcrToken]:
    """Find OCR tokens that contributed to an extracted field value."""
    if not value:
        return []
    vn = normalize_ar(value)
    digits_v = _digit_soup(value)
    hits: list[OcrToken] = []
    for t in tokens:
        tn = t.norm_text
        if not tn:
            continue
        if tn in vn or vn in tn:
            hits.append(t)
            continue
        if digits_v and len(digits_v) >= 10:
            td = _digit_soup(t.text)
            if td and (td in digits_v or digits_v in td):
                hits.append(t)
    # Prefer spatially clustered hits (same field block)
    if len(hits) > 8:
        hits = sorted(hits, key=lambda t: t.cy)[:8]
    return hits


def union_norm_bbox(
    tokens: list[OcrToken], pad: float = 0.02
) -> tuple[float, float, float, float] | None:
    if not tokens:
        return None
    return (
        max(0.0, min(t.x0 for t in tokens) - pad),
        max(0.0, min(t.y0 for t in tokens) - pad),
        min(1.0, max(t.x1 for t in tokens) + pad),
        min(1.0, max(t.y1 for t in tokens) + pad),
    )


def crop_norm_region(
    image_bgr: np.ndarray,
    box: tuple[float, float, float, float] | None,
    *,
    min_h: int = 28,
) -> np.ndarray | None:
    if box is None:
        return None
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = box
    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = int(x1 * w), int(y1 * h)
    px0, py0 = max(0, px0), max(0, py0)
    px1 = min(w, max(px0 + 2, px1))
    py1 = min(h, max(py0 + 2, py1))
    if py1 - py0 < min_h:
        mid = (py0 + py1) // 2
        py0 = max(0, mid - min_h // 2)
        py1 = min(h, py0 + min_h)
    roi = image_bgr[py0:py1, px0:px1]
    return roi if roi.size else None


def annotate_field_regions(
    image_bgr: np.ndarray,
    regions: dict[str, list[OcrToken]],
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    colors = {
        "full_name": (40, 140, 80),
        "address": (180, 110, 40),
        "national_id": (40, 90, 200),
        "job": (140, 60, 160),
        "religion": (60, 60, 180),
        "marital_status": (40, 150, 180),
        "gender": (90, 90, 90),
        "husband_name": (160, 80, 120),
    }
    tags = {
        "full_name": "name",
        "address": "addr",
        "national_id": "nid",
        "job": "job",
        "religion": "rel",
        "marital_status": "mar",
        "gender": "sex",
        "husband_name": "husb",
    }
    for key, toks in regions.items():
        box = union_norm_bbox(toks, pad=0.012)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        p1 = (int(x0 * w), int(y0 * h))
        p2 = (int(x1 * w), int(y1 * h))
        color = colors.get(key, (20, 92, 76))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(
            out,
            tags.get(key, key[:6]),
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


# Field key → ID_ZONES box name (must match the overlay the user was shown)
FIELD_TO_ZONE: dict[str, tuple[str, str]] = {
    # field_key: (side, zone_name)
    "full_name": ("front", "name"),
    "address": ("front", "address"),
    "national_id": ("front", "national_id"),
    "face": ("front", "face"),
    "job": ("back", "job"),
    "religion": ("back", "status"),
    "marital_status": ("back", "status"),
    "gender": ("back", "status"),
    "husband_name": ("back", "status"),
    "expiry_date": ("back", "expiry"),
}


def zone_box_for_field(field_key: str, side: str) -> tuple[float, float, float, float] | None:
    mapping = FIELD_TO_ZONE.get(field_key)
    if not mapping:
        return None
    z_side, z_name = mapping
    if side == "front" and z_side != "front":
        return None
    if side == "back" and z_side != "back":
        return None
    return ID_ZONES.get(z_side, {}).get(z_name)


def build_field_crops(
    card: np.ndarray,
    tokens: list[OcrToken],
    fields: dict[str, Any],
    face: np.ndarray | None,
    *,
    side: str = "front",
) -> dict[str, Any]:
    """Crop + annotate using fixed ID_ZONES (same boxes as layout overlay)."""
    _ = tokens  # zones are layout-fixed; OCR tokens not used for boxes
    crop_keys_front = ["full_name", "address", "national_id"]
    crop_keys_back = [
        "job",
        "religion",
        "marital_status",
        "gender",
        "husband_name",
        "expiry_date",
    ]
    crop_keys = crop_keys_front if side != "back" else crop_keys_back

    crops: dict[str, str] = {}
    meta: dict[str, Any] = {}

    for key in crop_keys:
        box = zone_box_for_field(key, "front" if side != "back" else "back")
        if box is None:
            continue
        roi = crop_norm_region(card, box)
        if roi is not None and roi.size:
            crops[key] = _encode_image_b64(roi, quality=90)
            meta[key] = {
                "bbox_norm": [round(x, 4) for x in box],
                "zone": FIELD_TO_ZONE[key][1],
                "value": fields.get(key),
            }

    # Annotated image = exact ID_ZONES overlay (matches front_zones.jpg layout)
    layout_side = "back" if side == "back" else "front"
    annotated = draw_id_zones(card, layout_side, thickness=2)

    out: dict[str, Any] = {
        "annotated": _encode_image_b64(annotated, quality=88),
        "crops": crops,
        "crop_meta": meta,
    }
    if layout_side == "front":
        face_roi = crop_norm_region(card, ID_ZONES["front"]["face"])
        if face is not None and face.size:
            out["face"] = _encode_image_b64(face, quality=90)
        elif face_roi is not None and face_roi.size:
            out["face"] = _encode_image_b64(face_roi, quality=90)
    return out
