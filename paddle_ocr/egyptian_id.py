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
        # الاسم سطرين — تحت الهيدر مباشرة (يشمل الاسم الأول)
        "name": (0.42, 0.24, 0.98, 0.46),
        # العنوان سطرين — فوق الرقم القومي فقط
        "address": (0.45, 0.46, 0.98, 0.68),
        # الرقم القومي — كامل الـ 14 رقم (يسارًا بما يكفي للـ ٣ الأولى)
        "national_id": (0.30, 0.68, 0.99, 0.90),
        # تاريخ الميلاد المطبوع — سطر كامل تحت الصورة (يشمل النسر ونهاية اليوم)
        "birth_date": (0.02, 0.60, 0.52, 0.86),
        # رقم المصنع
        "serial": (0.04, 0.87, 0.35, 0.98),
    },
    "back": {
        # المهنة (أعلى الظهر) — بدون تداخل مع شريط الحالة
        "job": (0.04, 0.02, 0.96, 0.25),
        # نوع / ديانة / حالة اجتماعية
        "status": (0.04, 0.25, 0.96, 0.40),
        # البطاقة سارية حتى … (فوق الباركود مباشرة)
        "expiry": (0.04, 0.38, 0.96, 0.62),
        # الباركود ثنائي الأبعاد
        "barcode": (0.02, 0.60, 0.98, 0.98),
    },
}

# Colors for drawing zone overlays (BGR)
ID_ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "face": (80, 180, 40),
    "header": (60, 160, 220),
    "name": (40, 140, 80),
    "address": (180, 110, 40),
    "national_id": (40, 40, 220),
    "birth_date": (200, 80, 160),
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
    "كهرباء",
    "مدني",
    "ميكانيك",
    "زراعه",
    "زراعة",
    "حاسب",
    "حاسبات",
    "برمج",
    "اداري",
    "إداري",
    "سائق",
    "نجار",
    "حداد",
)

# Common Paddle/OnnxTR confusions on Egyptian ID print (keys normalized later)
_OCR_WORD_FIXES_RAW: dict[str, str] = {
    "صحمد": "محمد",
    "محمو": "محمود",
    "خالاد": "خالد",
    "خالا": "خالد",
    "علس": "على",
    "الاه": "الله",
    "الآه": "الله",
    "اللة": "الله",
    "اشمون": "أشمون",
    "سمادون": "سمادون",
    "اعزب": "أعزب",
    "انثي": "أنثى",
    "انثى": "أنثى",
    "مسيحى": "مسيحي",
    "يهودى": "يهودي",
}
OCR_WORD_FIXES: dict[str, str] = {}


def normalize_ar(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.translate(_AR_MAP)
    text = re.sub(r"\s+", " ", text)
    return text


# Populate after normalize_ar exists
OCR_WORD_FIXES.update({normalize_ar(k): v for k, v in _OCR_WORD_FIXES_RAW.items()})


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


def extract_printed_birth_date(texts: list[str]) -> str | None:
    """Parse DOB printed under the photo (YYYY/MM/DD or DD/MM/YYYY)."""
    candidates: list[str] = []
    for text in texts:
        ascii_t = to_ascii_digits(text or "")
        if not ascii_t:
            continue
        # Normalize common OCR junk around the date
        ascii_t = (
            ascii_t.replace("\\", "/")
            .replace("|", "/")
            .replace(",", "/")
            .replace(" ", "")
        )
        for m in re.finditer(
            r"(19\d{2}|20\d{2})\s*[/\-.\u066B\u066C]?\s*(\d{1,2})\s*[/\-.\u066B\u066C]?\s*(\d{1,2})",
            ascii_t,
        ):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                candidates.append(f"{y:04d}-{mo:02d}-{d:02d}")
        for m in re.finditer(
            r"(\d{1,2})\s*[/\-.\u066B\u066C]?\s*(\d{1,2})\s*[/\-.\u066B\u066C]?\s*(19\d{2}|20\d{2})",
            ascii_t,
        ):
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a <= 31 and 1 <= b <= 12:
                candidates.append(f"{y:04d}-{b:02d}-{a:02d}")
            elif b <= 31 and 1 <= a <= 12:
                candidates.append(f"{y:04d}-{a:02d}-{b:02d}")
        # Spaced: 2000 09 20
        for m in re.finditer(
            r"(19\d{2}|20\d{2})\s+(\d{1,2})\s+(\d{1,2})",
            to_ascii_digits(text or ""),
        ):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                candidates.append(f"{y:04d}-{mo:02d}-{d:02d}")
        digits = _digit_soup(ascii_t)
        if len(digits) >= 8:
            for i in range(0, len(digits) - 7):
                chunk = digits[i : i + 8]
                if chunk[:2] in {"19", "20"}:
                    y, mo, d = int(chunk[:4]), int(chunk[4:6]), int(chunk[6:8])
                    if 1 <= mo <= 12 and 1 <= d <= 31:
                        candidates.append(f"{y:04d}-{mo:02d}-{d:02d}")
    if not candidates:
        return None
    from collections import Counter

    return Counter(candidates).most_common(1)[0][0]


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
    if target > 1000:
        scale = 1000 / max(h, w)
    scale = max(1.25, min(scale, 2.6 if mode == "digits" else 2.2))
    big = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    enhanced = enhance_id_card(big)
    if mode == "digits":
        # Soft prep only — harsh threshold kills DOB over the eagle watermark
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        return [
            enhanced,
            cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        ]
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


def rotate_card_image(image_bgr: np.ndarray, angle_cw: int) -> np.ndarray:
    """Rotate card by 0/90/180/270 degrees clockwise."""
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


def _green_header_orientation_score(image_bgr: np.ndarray) -> float:
    """Upright front ID: teal/green header sits in the TOP band."""
    try:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Teal/green header ink on Egyptian front cards
        mask = cv2.inRange(hsv, (35, 35, 35), (100, 255, 255))
        h, w = mask.shape[:2]
        if h < 20 or w < 20:
            return 0.0
        top = float(mask[: max(1, h // 4), :].mean())
        bot = float(mask[3 * h // 4 :, :].mean())
        return top - bot
    except Exception:  # noqa: BLE001
        return 0.0


def _magenta_header_orientation_score(image_bgr: np.ndarray) -> float:
    """Upright back ID: pink/magenta republic ink sits in the TOP band."""
    try:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Magenta / fuchsia header common on Egyptian ID backs
        m1 = cv2.inRange(hsv, (135, 35, 40), (175, 255, 255))
        m2 = cv2.inRange(hsv, (0, 40, 50), (18, 255, 255))  # warm pink-red
        mask = cv2.bitwise_or(m1, m2)
        h, w = mask.shape[:2]
        if h < 20 or w < 20:
            return 0.0
        top = float(mask[: max(1, h // 4), :].mean())
        bot = float(mask[3 * h // 4 :, :].mean())
        return top - bot
    except Exception:  # noqa: BLE001
        return 0.0


def _barcode_band_density(image_bgr: np.ndarray, y0: float, y1: float) -> float:
    """How much a horizontal band looks like a dense PDF417 barcode."""
    h, w = image_bgr.shape[:2]
    if h < 40 or w < 40:
        return 0.0
    ya, yb = int(h * y0), int(h * y1)
    ya, yb = max(0, ya), min(h, max(ya + 1, yb))
    roi = image_bgr[ya:yb, int(w * 0.04) : int(w * 0.96)]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 160)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
    col_std = float(np.std(gray.astype(np.float32), axis=0).mean())
    score = 0.0
    if edge_ratio >= 0.12:
        score += 3.0
    elif edge_ratio >= 0.07:
        score += 1.5
    elif edge_ratio >= 0.04:
        score += 0.5
    if col_std >= 35:
        score += 2.0
    elif col_std >= 22:
        score += 1.0
    return score


def _barcode_orientation_score(image_bgr: np.ndarray) -> float:
    """Positive when dense barcode is at the BOTTOM (upright Egyptian ID back)."""
    bot = _barcode_band_density(image_bgr, 0.55, 0.98)
    top = _barcode_band_density(image_bgr, 0.02, 0.45)
    return bot - top


def score_card_upright(image_bgr: np.ndarray) -> float:
    """Higher = more likely upright Egyptian ID (front or back landscape)."""
    if image_bgr is None or not getattr(image_bgr, "size", 0):
        return -1e9
    h, w = image_bgr.shape[:2]
    score = 0.0
    # Cropped ID is usually landscape
    if w >= h:
        score += 2.0
    else:
        score -= 1.5

    faces = detect_faces(image_bgr)
    face_bonus = 0.0
    if faces:
        x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
        cx = (x + bw * 0.5) / max(w, 1)
        cy = (y + bh * 0.5) / max(h, 1)
        # Front upright: portrait on the LEFT, upper half
        if cx <= 0.40:
            face_bonus += 16.0
        elif cx >= 0.55:
            face_bonus -= 12.0
        else:
            face_bonus -= 2.0
        if cy <= 0.58:
            face_bonus += 5.0
        else:
            face_bonus -= 4.0
        area = (bw * bh) / float(max(w * h, 1))
        if 0.02 <= area <= 0.28:
            face_bonus += 2.0

    g = _green_header_orientation_score(image_bgr)
    m = _magenta_header_orientation_score(image_bgr)
    bar = _barcode_orientation_score(image_bgr)
    # Strong barcode asymmetry → treat as BACK (no reliable face cue)
    looks_like_back = abs(bar) >= 1.2 and face_bonus < 8.0

    if looks_like_back:
        # Back upright: PDF417 at bottom, pink/magenta header at top
        score += max(-14.0, min(14.0, bar * 4.0))
        score += max(-8.0, min(8.0, m / 5.0))
        score += max(-4.0, min(4.0, g / 10.0))
        score += face_bonus * 0.15  # faces on back are rare / false positives
    else:
        score += face_bonus
        # Green republic header → top when upright (front 180° flips)
        score += max(-8.0, min(8.0, g / 6.0))
        score += max(-3.0, min(3.0, m / 12.0))
        score += max(-3.0, min(3.0, bar * 0.8))
    return score


def correct_card_orientation(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, int, str]:
    """Auto-rotate ID card to upright (0/90/180/270 — front or back)."""
    if image_bgr is None or not getattr(image_bgr, "size", 0):
        return image_bgr, 0, "ORIENT: تخطي"
    h, w = image_bgr.shape[:2]
    # Always try quarter turns — phone photos of back/front often land sideways
    angles = [0, 90, 180, 270]

    # Score on a smaller probe for speed (YuNet × 4)
    probe = image_bgr
    max_side = max(h, w)
    if max_side > 720:
        s = 720 / max_side
        probe = cv2.resize(
            image_bgr,
            (max(1, int(w * s)), max(1, int(h * s))),
            interpolation=cv2.INTER_AREA,
        )

    best_ang = 0
    best_sc = -1e18
    for ang in angles:
        rot = rotate_card_image(probe, ang)
        sc = score_card_upright(rot)
        if sc > best_sc:
            best_sc = sc
            best_ang = ang

    best_img = rotate_card_image(image_bgr, best_ang) if best_ang else image_bgr
    if best_ang:
        note = f"ORIENT: تدوير البطاقة {best_ang}° قبل القراءة"
    else:
        note = "ORIENT: الاتجاه مضبوط"
    return best_img, best_ang, note


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

    if not boxes and hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            for x, y, bw, bh in cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            ):
                boxes.append((int(x), int(y), int(bw), int(bh)))
        except Exception:  # noqa: BLE001
            logger.debug("Haar cascade unavailable", exc_info=True)

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


def apply_ocr_word_fixes(text: str) -> str:
    """Fix frequent Arabic OCR substitutions on ID cards."""
    parts = []
    for w in (text or "").split():
        nw = normalize_ar(w)
        parts.append(OCR_WORD_FIXES.get(nw, w))
    return " ".join(parts).strip()


_ID_HEADER_WORDS = {
    normalize_ar(w)
    for w in (
        "بطاقة",
        "بطاقه",
        "تحقيق",
        "الشخصية",
        "الشخصيه",
        "شخصية",
        "شخصيه",
        "جمهورية",
        "جمهوريه",
        "مصر",
        "العربية",
        "العربيه",
        "وزارة",
        "وزاره",
        "الداخلية",
        "الداخليه",
    )
}


def _contains_id_header(text: str) -> bool:
    n = normalize_ar(text)
    if any(k in n for k in ("بطاق", "تحقيق", "شخص", "جمهور", "وزاره", "داخليه")):
        return True
    return any(normalize_ar(w) in _ID_HEADER_WORDS for w in (text or "").split())


def finalize_front_name(text: str | None) -> str | None:
    """Drop card title/header words; keep person-name tokens only."""
    if not text:
        return None
    text = apply_ocr_word_fixes(_strip_embedded_header(text))
    kept: list[str] = []
    for w in text.split():
        nw = normalize_ar(w)
        if nw in _ID_HEADER_WORDS:
            continue
        if "بطاق" in nw or "تحقيق" in nw or "شخص" in nw or "جمهور" in nw:
            continue
        if _has_ocr_garbage(w):
            continue
        kept.append(w)
    out = " ".join(kept).strip()
    if not out or len(normalize_ar(out)) < 2:
        return None
    # Reject if still mostly header (e.g. «تحقيق الشخصية عوض الله» without first name)
    if _contains_id_header(out):
        out2 = " ".join(
            w
            for w in kept
            if not any(k in normalize_ar(w) for k in ("تحقيق", "شخص", "بطاق"))
        ).strip()
        out = out2 or None
    return out


def finalize_front_address(
    text: str | None,
    *,
    national_id: str | None = None,
) -> str | None:
    """Remove NID bleed, latin junk, and repeated place tokens."""
    if not text:
        return None
    text = apply_ocr_word_fixes(_strip_address_label_prefix(text))
    nid_digits = _digit_soup(national_id or "")
    words: list[str] = []
    for w in text.split():
        if not w.strip():
            continue
        if re.fullmatch(r"[A-Za-z]{1,3}", w):
            continue
        if len(_digit_soup(w)) >= 6:
            continue
        if nid_digits and nid_digits in _digit_soup(w):
            continue
        if _has_ocr_garbage(w) and not _looks_like_address(w):
            continue
        words.append(w)
    # De-dupe normalized tokens (اشمون المنوفية … اشمون المنوفية)
    seen: set[str] = set()
    deduped: list[str] = []
    for w in words:
        nw = normalize_ar(w)
        if not nw or nw in seen:
            continue
        # Near-duplicates / OCR variants of already-kept place words
        if any(
            (nw in s or s in nw)
            and abs(len(nw) - len(s)) <= 2
            and min(len(nw), len(s)) >= 4
            for s in seen
        ):
            continue
        seen.add(nw)
        deduped.append(w)
    out = " ".join(deduped).strip()
    out = re.sub(r"\s+", " ", out)
    if len(_digit_soup(out)) >= 10:
        return None
    return out or None


def _effective_name_box(tokens: list[OcrToken]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = ID_ZONES["front"]["name"]
    if tokens:
        # Keep a small gap under header, but never cut the first-name line
        y0 = max(y0, min(_header_bottom_y(tokens) + 0.008, 0.30))
    return (x0, max(0.22, min(y0, 0.30)), x1, y1)


def _effective_address_box(tokens: list[OcrToken]) -> tuple[float, float, float, float]:
    ax0, ay0, ax1, ay1 = ID_ZONES["front"]["address"]
    _, ny0, _, ny1 = _effective_name_box(tokens) if tokens else ID_ZONES["front"]["name"]
    ay0 = max(ay0, ny1 + 0.008)
    nid_top = ID_ZONES["front"]["national_id"][1]
    ay1 = min(ay1, nid_top - 0.02)
    return (ax0, ay0, ax1, max(ay1, ay0 + 0.08))


def _match_known_value(text: str, values: tuple[str, ...]) -> str | None:
    n = normalize_ar(text)
    if not n:
        return None
    # Prefer longer labels first (متزوجة before متزوج)
    ordered = sorted(values, key=lambda v: len(normalize_ar(v)), reverse=True)
    for v in ordered:
        vn = normalize_ar(v)
        if not vn:
            continue
        if n == vn:
            return v
    for part in re.split(r"[\s/|،,·•]+", n):
        part = part.strip(" :：-–=")
        if not part:
            continue
        for v in ordered:
            vn = normalize_ar(v)
            if part == vn:
                return v
    for v in ordered:
        vn = normalize_ar(v)
        if re.search(rf"(^|\s){re.escape(vn)}(\s|$)", n):
            return v
    return None


def extract_expiry_date(texts: list[str]) -> str | None:
    """Parse card validity date near «سارية حتى» (YYYY-MM-DD).

    OCR often drops slashes (٢٠٣٢/٠٢/١٥ → ٢٠٣٢٠٢١٥) or turns '/' into '1'
    (→ ٢٠٣٢١٠٢١٥). Prefer dates on سارية/حتى lines and recover 9-digit glitches.
    """

    def _ymd_ok(y: int, mo: int, d: int) -> bool:
        return 2009 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31

    def _fmt(y: int, mo: int, d: int) -> str:
        return f"{y:04d}-{mo:02d}-{d:02d}"

    def _from_digit_run(run: str) -> list[tuple[float, str]]:
        found: list[tuple[float, str]] = []
        if not run.startswith("20"):
            return found
        if len(run) == 8:
            y, mo, d = int(run[:4]), int(run[4:6]), int(run[6:8])
            if _ymd_ok(y, mo, d):
                found.append((8.0, _fmt(y, mo, d)))
            return found
        if len(run) == 9:
            # Extra digit (slash misread as 1, etc.) — try deleting one
            for drop in range(9):
                chunk = run[:drop] + run[drop + 1 :]
                y, mo, d = int(chunk[:4]), int(chunk[4:6]), int(chunk[6:8])
                if not _ymd_ok(y, mo, d):
                    continue
                score = 7.0
                # '/' → '1' right after year is the common glitch: 2032 1 0215
                if drop == 4 and run[4] == "1":
                    score += 3.0
                if mo <= 9:
                    score += 0.5
                found.append((score, _fmt(y, mo, d)))
            return found
        # Longer digit soup — sliding windows, but penalize leftovers
        for i in range(0, len(run) - 7):
            chunk = run[i : i + 8]
            if not chunk.startswith("20"):
                continue
            y, mo, d = int(chunk[:4]), int(chunk[4:6]), int(chunk[6:8])
            if _ymd_ok(y, mo, d):
                found.append((3.0, _fmt(y, mo, d)))
        for i in range(0, len(run) - 8):
            chunk9 = run[i : i + 9]
            found.extend((sc - 1.0, dt) for sc, dt in _from_digit_run(chunk9))
        return found

    def _candidates_from_text(text: str) -> list[tuple[float, str]]:
        ascii_t = to_ascii_digits(text or "")
        if not ascii_t:
            return []
        found: list[tuple[float, str]] = []
        # Clear YYYY/MM/DD (or - . space) — highest trust
        for m in re.finditer(
            r"(20\d{2})\s*[/\-.\s]\s*(\d{1,2})\s*[/\-.\s]\s*(\d{1,2})",
            ascii_t,
        ):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if _ymd_ok(y, mo, d):
                found.append((12.0, _fmt(y, mo, d)))
        for m in re.finditer(
            r"(\d{1,2})\s*[/\-.\s]\s*(\d{1,2})\s*[/\-.\s]\s*(20\d{2})",
            ascii_t,
        ):
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if _ymd_ok(y, b, a) and a <= 31 and b <= 12:
                found.append((11.0, _fmt(y, b, a)))
            elif _ymd_ok(y, a, b) and b <= 31 and a <= 12:
                found.append((10.0, _fmt(y, a, b)))
        # Digit runs (compact / glitched)
        for m in re.finditer(r"20\d{6,12}", re.sub(r"\D", "", ascii_t)):
            found.extend(_from_digit_run(m.group(0)))
        # Also whole digit soup if no run matched oddly
        digits = re.sub(r"\D", "", ascii_t)
        if digits.startswith("20") and 8 <= len(digits) <= 14:
            found.extend(_from_digit_run(digits[:9] if len(digits) >= 9 else digits[:8]))
        return found

    scored: list[tuple[float, str]] = []
    texts_list = [t for t in (texts or []) if t]
    for i, text in enumerate(texts_list):
        n = normalize_ar(text)
        ctx = 0.0
        if "ساري" in n:
            ctx += 20.0
        if "حتى" in n or "حتي" in n:
            ctx += 8.0
        if "بطاق" in n:
            ctx += 4.0
        if re.search(r"20\d{2}", to_ascii_digits(text)):
            ctx += 2.0
        # Skip pure noise without year / validity cue
        if ctx < 2.0:
            continue
        for sc, dt in _candidates_from_text(text):
            scored.append((ctx + sc, dt))
        # Neighbor lines often hold the date when label/date split
        if ctx >= 8.0:
            for j in (i - 1, i + 1):
                if 0 <= j < len(texts_list):
                    for sc, dt in _candidates_from_text(texts_list[j]):
                        scored.append((ctx * 0.5 + sc, dt))

    if not scored:
        # Last resort: any date-like pattern in joined text
        joined = to_ascii_digits(" ".join(texts_list))
        for sc, dt in _candidates_from_text(joined):
            scored.append((sc, dt))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def finalize_back_job(text: str | None) -> str | None:
    """Clean profession field — strip labels and collapse duplicated OCR lines."""
    if not text:
        return None
    t = apply_ocr_word_fixes(str(text)).strip()
    if not t:
        return None
    n = normalize_ar(t)
    for lbl in ("المهنه", "المهنة", "مهنة"):
        if lbl in n:
            # Keep text after label if present, else drop label word only
            if n.startswith(lbl) or f" {lbl} " in f" {n} ":
                after = n.split(lbl, 1)[-1].strip(" :：-")
                if after and len(after) >= 3:
                    # Rebuild from original roughly by removing label once
                    t2 = re.sub(
                        r"المهنة|المهنه|مهنة",
                        "",
                        t,
                        count=1,
                    ).strip(" :：-")
                    t = t2 or t
                    n = normalize_ar(t)
            break
    words = [w for w in t.split() if w]
    out_w: list[str] = []
    for w in words:
        if out_w and normalize_ar(out_w[-1]) == normalize_ar(w):
            continue
        out_w.append(w)
    joined = " ".join(out_w).strip()
    # Full-line duplicate: "X Y X Y" or "X X"
    parts = joined.split()
    if len(parts) >= 2:
        mid = len(parts) // 2
        left, right = " ".join(parts[:mid]), " ".join(parts[mid:])
        if normalize_ar(left) == normalize_ar(right):
            joined = left
        elif normalize_ar(right) and normalize_ar(right) in normalize_ar(left):
            joined = left
        elif normalize_ar(left) and normalize_ar(left) in normalize_ar(right):
            joined = right
    # Drop trailing status words accidentally glued into job
    cleaned: list[str] = []
    status_stop = {
        normalize_ar(x)
        for x in RELIGION_VALUES + MARITAL_VALUES + GENDER_VALUES + ("انثى", "أنثى")
    }
    for w in joined.split():
        if normalize_ar(w) in status_stop:
            break
        cleaned.append(w)
    joined = " ".join(cleaned).strip() or joined
    return joined or None


def _jobs_near_duplicate(a: str, b: str) -> bool:
    na, nb = normalize_ar(a), normalize_ar(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    wa, wb = set(na.split()), set(nb.split())
    if not wa or not wb:
        return False
    overlap = len(wa & wb) / max(len(wa), len(wb))
    return overlap >= 0.75


def _parse_status_triplet(text: str) -> dict[str, str | None]:
    """Pull gender/religion/marital from a status band line."""
    out: dict[str, str | None] = {
        "gender": None,
        "religion": None,
        "marital_status": None,
    }
    fixed = apply_ocr_word_fixes(text)
    g = _match_known_value(fixed, GENDER_VALUES)
    if g:
        out["gender"] = "ذكر" if normalize_ar(g) == "ذكر" else "أنثى"
    r = _match_known_value(fixed, RELIGION_VALUES)
    if r:
        # Canonical forms
        rn = normalize_ar(r)
        out["religion"] = (
            "مسلم"
            if rn == "مسلم"
            else "مسيحي"
            if rn.startswith("مسيح")
            else "يهودي"
            if rn.startswith("يهود")
            else r
        )
    m = _match_known_value(fixed, MARITAL_VALUES)
    if m:
        mn = normalize_ar(m)
        canon = {
            "اعزب": "أعزب",
            "متزوج": "متزوج",
            "متزوجه": "متزوجة",
            "مطلق": "مطلق",
            "مطلقه": "مطلقة",
            "ارمل": "أرمل",
            "ارمله": "أرملة",
            "انسه": "آنسة",
        }
        # Keep feminine forms when original text has feminine marker
        if mn == "متزوج" and ("متزوجه" in normalize_ar(fixed) or "متزوجة" in fixed):
            out["marital_status"] = "متزوجة"
        else:
            out["marital_status"] = canon.get(mn, m)
    return out


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

    # Ignore barcode band noise for text fields
    usable = [t for t in tokens if t.cy <= 0.62]
    if len(usable) < 2:
        usable = list(tokens)

    lines = _cluster_tokens_into_lines(usable, y_thresh=0.045)
    line_texts = [
        apply_ocr_word_fixes(_merge_line_scraps(_line_text(ln))) for ln in lines
    ]
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

    # Status band first (middle of card) — often one RTL line
    for text in line_texts:
        st = _parse_status_triplet(text)
        for k, v in st.items():
            if v and not out.get(k):
                out[k] = v
                notes.append(f"BACK: {k}")

    for t in usable:
        text = apply_ocr_word_fixes(t.text.strip())
        st = _parse_status_triplet(text)
        for k, v in st.items():
            if v and not out.get(k):
                out[k] = v

    job_candidates: list[tuple[float, str]] = []
    for ln, text in zip(lines, line_texts):
        if not text:
            continue
        cy = sum(t.cy for t in ln) / len(ln)
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
        # Skip pure status triplets even with separators
        st = _parse_status_triplet(text)
        status_hits = sum(1 for v in st.values() if v)
        if status_hits >= 2 and len(parts) <= 5:
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
        if cy < 0.32:
            score += 2.0
        elif cy < 0.45:
            score += 0.5
        if cy > 0.50:
            score -= 2.0
        if 0.02 <= cy <= 0.45 and _arabic_ratio(text) >= 0.55 and len(text) >= 4:
            # Reject short junk / barcode hallucinations (e.g. «مسلح»)
            if len(normalize_ar(text)) < 6 and not any(
                normalize_ar(h) in n for h in JOB_HINTS
            ):
                continue
            if _has_ocr_garbage(text) and not any(
                normalize_ar(h) in n for h in JOB_HINTS
            ):
                continue
            job_candidates.append((score, text.strip()))

    if job_candidates:
        job_candidates.sort(key=lambda x: x[0], reverse=True)
        top_score, top = job_candidates[0]
        has_hint = any(normalize_ar(h) in normalize_ar(top) for h in JOB_HINTS)
        if top_score < 3.5 and not has_hint:
            top = None
        if top:
            # Only append a second line when it's a real continuation (تخصص…),
            # never a near-duplicate re-OCR of the same profession.
            extra = None
            for _s, t in job_candidates[1:3]:
                if _jobs_near_duplicate(top, t):
                    continue
                tn = normalize_ar(t)
                if "تخصص" in tn or (
                    len(t) >= 6
                    and _arabic_ratio(t) >= 0.6
                    and not _parse_status_triplet(t)["gender"]
                ):
                    extra = t
                    break
            raw_job = f"{top} {extra}".strip() if extra else top
            out["job"] = finalize_back_job(raw_job)
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
    texts = [t.text for t in tokens]
    direct = best_national_id_from_texts(texts)
    if direct:
        return direct
    bottom = [t for t in tokens if t.cy >= 0.65]
    if bottom:
        return best_national_id_from_texts([t.text for t in bottom])
    return None


def _nid_grounded_in_texts(nid: str, texts: list[str]) -> bool:
    """True if NID (or its 11+ digit core) appears in a single OCR line."""
    d = _digit_soup(nid)
    if len(d) != 14:
        return False
    core = d[3:]  # MM DD gov serial check — stable across century/year OCR errors
    for t in texts:
        soup = _digit_soup(t)
        if len(soup) < 10:
            continue
        if d in soup or soup in d:
            return True
        if core and core in soup:
            return True
        if len(soup) >= 11 and (soup[-11:] == d[-11:] or soup[:11] == d[:11]):
            return True
        # 15-digit strip with one garbage digit
        if len(soup) == 15:
            for i in range(15):
                if soup[:i] + soup[i + 1 :] == d:
                    return True
    return False


def score_national_id_candidate(cand: str) -> int:
    cand = _digit_soup(cand)
    if len(cand) != 14 or cand[0] not in "23":
        return -1
    decoded = decode_national_id(cand)
    # Must decode to a real calendar date — checksum alone is not enough
    if not decoded.get("valid") or not decoded.get("birth_date"):
        return -1
    # Require checksum so garbled OCR cannot invent a plausible DOB
    if not decoded.get("checksum_ok"):
        return 1
    score = 11
    gov = GOVERNORATES.get(cand[7:9])
    if gov and gov != "غير معروف":
        score += 2
    # Prefer more recent births when checksum+age already match (breaks 1950 vs 2000 ties)
    year = decoded.get("birth_year")
    if isinstance(year, int):
        age = 2026 - year
        if 5 <= age <= 90:
            score += 4
        elif 1 <= age <= 100:
            score += 2
        elif age > 110 or age < 0:
            score -= 6
        if year >= 1990:
            score += 2
        elif year >= 1970:
            score += 1
    return score


def recover_national_id_from_fragments(texts: list[str]) -> str | None:
    """Complete truncated NID OCR (often missing leading century/year digits)."""
    best: str | None = None
    best_score = -1
    blobs: list[str] = []
    for t in texts:
        d = _digit_soup(t)
        if 8 <= len(d) <= 16:
            blobs.append(d)
    joined = _digit_soup(" ".join(texts))
    if 8 <= len(joined) <= 32:
        blobs.append(joined)

    seen: set[str] = set()
    # Hint from individual OCR lines only (joined soup may start with a wrong ID)
    century_hint = None
    hint_len = 0
    for t in texts:
        d = _digit_soup(t)
        if d and d[0] in "23" and len(d) > hint_len:
            century_hint = d[0]
            hint_len = len(d)

    def consider(cand: str) -> None:
        nonlocal best, best_score
        if cand in seen or len(cand) != 14 or cand[0] not in "23":
            return
        seen.add(cand)
        s = score_national_id_candidate(cand)
        if century_hint and cand[0] == century_hint:
            s += 3
        if s > best_score:
            best_score, best = s, cand

    for blob in blobs:
        # Substrings: OCR often prepends garbage digits (15–16 chars)
        parts = [blob]
        if len(blob) > 14:
            for L in (14, 13, 12, 11, 10):
                for i in range(0, len(blob) - L + 1):
                    parts.append(blob[i : i + L])
        for part in parts:
            # Exact / sliding 14
            if len(part) >= 14:
                for i in range(0, len(part) - 13):
                    consider(part[i : i + 14])
            # Truncated: try inserting one missing digit (13→14)
            if len(part) == 13:
                for pos in range(14):
                    for d in "0123456789":
                        consider(part[:pos] + d + part[pos:])
            # Truncated: try prepending digits so result is a valid checksum NID
            if 10 <= len(part) <= 13:
                missing = 14 - len(part)
                centuries = (
                    (century_hint,) if century_hint in {"2", "3"} else ("3", "2")
                )
                if missing == 1:
                    prefixes = (
                        [century_hint]
                        if century_hint in {"2", "3"}
                        else [str(d) for d in range(10)]
                    )
                elif missing == 2:
                    prefixes = [f"{a}{b}" for a in centuries for b in range(10)]
                elif missing == 3:
                    prefixes = []
                    for a in centuries:
                        years = range(0, 30) if a == "3" else range(50, 100)
                        prefixes.extend(f"{a}{b:02d}" for b in years)
                        rest = range(30, 100) if a == "3" else range(0, 50)
                        prefixes.extend(f"{a}{b:02d}" for b in rest)
                else:
                    prefixes = []
                    for a in centuries:
                        for b in list(range(0, 30)) + list(range(30, 100)):
                            for c in range(10):
                                prefixes.append(f"{a}{b:02d}{c}")
                for pref in prefixes:
                    consider(pref + part)
            # Truncated from the right (less common)
            if 11 <= len(part) <= 13 and part[0] in "23":
                missing = 14 - len(part)
                if missing <= 2:
                    for tail in range(10**missing):
                        consider(part + f"{tail:0{missing}d}")
        # Also: drop one garbage digit from 15-digit OCR
        if len(blob) == 15:
            for i in range(15):
                consider(blob[:i] + blob[i + 1 :])
    return best if best_score >= 11 else None


def best_national_id_from_texts(texts: list[str]) -> str | None:
    """Pick the best 14-digit NID from OCR lines (valid date + checksum)."""
    blob = _digit_soup(" ".join(texts))
    century_hint = None
    hint_len = 0
    for t in texts:
        d = _digit_soup(t)
        if len(d) >= 11 and d[0] in "23" and len(d) > hint_len:
            century_hint = d[0]
            hint_len = len(d)
    if century_hint is None and blob and blob[0] in "23":
        century_hint = blob[0]

    best: str | None = None
    best_score = -1

    def consider(cand: str, bonus: int = 0) -> None:
        nonlocal best, best_score
        s = score_national_id_candidate(cand)
        if s < 11:
            return
        if century_hint and cand[0] == century_hint:
            s += 3
        elif century_hint and cand[0] != century_hint:
            s -= 2
        s += bonus
        if s > best_score:
            best_score, best = s, cand

    # Prefer recovery from over-long strip OCR (15–16 digits) — beats false exact-14 soup
    for t in texts:
        d = _digit_soup(t)
        if 15 <= len(d) <= 16:
            rec = recover_national_id_from_fragments([t])
            if rec:
                consider(rec, bonus=4)

    for t in texts:
        d = _digit_soup(t)
        if len(d) == 14 and d[0] in "23":
            consider(d)
    for i in range(0, max(0, len(blob) - 13)):
        cand = blob[i : i + 14]
        if cand[0] in "23":
            consider(cand)
    recovered = recover_national_id_from_fragments(texts)
    if recovered:
        consider(recovered, bonus=2)
    return best if best_score >= 11 else None


def birth_date_zone_box(
    face_box: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """DOB sits under the personal photo — use stable layout band (full date line)."""
    # Fixed layout is more reliable than YuNet face bottom (often truncates the date)
    _ = face_box
    return ID_ZONES["front"]["birth_date"]


def _has_ocr_garbage(text: str) -> bool:
    """Detect corrupted OCR (random diacritics / latin scraps on Arabic ID print)."""
    raw = text or ""
    if not raw:
        return True
    harakat = sum(1 for c in raw if "\u064b" <= c <= "\u0652")
    letters = sum(1 for c in raw if ("\u0600" <= c <= "\u06FF") or c.isalpha())
    if harakat >= 2 and letters > 0 and harakat / letters >= 0.18:
        return True
    latin = sum(1 for c in raw if ("a" <= c.lower() <= "z"))
    if latin >= 2 and _arabic_ratio(raw) < 0.85:
        return True
    # Corrupted republic/title scraps
    n = normalize_ar(raw)
    if any(x in n for x in ("حرب", "ينمص", "حمه", "الحبيه", "العبيه", "موري", "حرهو")):
        if len(n.split()) <= 4 and not _is_strong_address_start(raw):
            return True
    # Lone weird 2-word scraps with no common name particle
    words = [w for w in n.split() if w]
    if len(words) == 2 and all(len(w) <= 6 for w in words):
        if harakat >= 1 and not _looks_like_address(raw):
            # Likely title garbage like "فص الحبية"
            if not any(w in _COMMON_FIRST_NAMES for w in words):
                return True
    return False


# Frequent Egyptian first names — used to keep real short name lines
_COMMON_FIRST_NAMES = {
    normalize_ar(x)
    for x in (
        "محمد", "محمود", "احمد", "أحمد", "مصطفى", "مصطفي", "علي", "على",
        "حسن", "حسين", "ابراهيم", "إبراهيم", "يوسف", "يوسف", "خالد", "عمر",
        "ياسر", "طارق", "سامي", "سامى", "سعيد", "عبدالله", "عبدالله",
        "فاطمه", "فاطمة", "عائشه", "عائشة", "مريم", "نور", "هدى", "هدي",
        "ايه", "آية", "سارة", "ساره", "نورا", "دينا", "منى", "مني",
    )
}


def _header_bottom_y(tokens: list[OcrToken]) -> float:
    """Lowest edge of header/title tokens — name starts below this."""
    bottoms: list[float] = []
    for t in tokens:
        n = normalize_ar(t.text)
        if not n:
            continue
        if _is_header_noise(t.text) or "بطاق" in n or "جمهور" in n or "وزاره" in n:
            bottoms.append(t.y1)
            continue
        if _has_ocr_garbage(t.text) and t.cy < 0.35:
            bottoms.append(t.y1)
    return max(bottoms) if bottoms else 0.18


def _strip_embedded_header(text: str) -> str:
    """Remove header phrases glued onto a real name/address line by OCR."""
    out = re.sub(r"\s+", " ", (text or "").strip())
    if not out:
        return out
    phrases = (
        "بطاقة تحقيق الشخصية",
        "بطاقه تحقيق الشخصيه",
        "جمهورية مصر العربية",
        "جمهوريه مصر العربيه",
        "وزارة الداخلية",
        "وزاره الداخليه",
    )
    for phr in phrases:
        out = out.replace(phr, " ")
    # Drop individual header words when the line mixed name + title
    header_words = {
        normalize_ar(w)
        for phr in phrases
        for w in phr.split()
    }
    n_all = normalize_ar(out)
    if any(normalize_ar(p) in n_all or any(hw in n_all for hw in ("بطاق", "جمهور", "تحقيق", "شخصيه", "شخصية")) for p in phrases):
        kept = [
            w
            for w in out.split()
            if normalize_ar(w) not in header_words
            and "بطاق" not in normalize_ar(w)
            and "جمهور" not in normalize_ar(w)
            and "تحقيق" not in normalize_ar(w)
            and normalize_ar(w) not in {"مصر", "العربيه", "العربية", "الداخليه", "الداخلية", "وزاره", "وزارة"}
        ]
        out = " ".join(kept)
    return re.sub(r"\s+", " ", out).strip(" -–=:،,")


def _is_header_noise(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    stripped = _strip_embedded_header(raw)
    n_raw = normalize_ar(raw)
    n_strip = normalize_ar(stripped)

    # Mixed OCR: "محمود بطاقة تحقيق الشخصية" → keep as usable after strip
    if stripped and n_strip and n_strip != n_raw:
        if "بطاق" not in n_strip and "جمهور" not in n_strip and len(n_strip) >= 2:
            return False

    n = n_raw
    if len(n) < 2:
        return True
    for h in HEADER_NOISE:
        hn = normalize_ar(h)
        if hn == n or (hn in n and len(n) <= len(hn) + 2):
            return True
    if any(k in n for k in ("جمهوريه مصر", "بطاقه تحقيق", "وزاره الداخليه")):
        return True
    if n.startswith("جمهور"):
        return True
    if "جمهور" in n and len(n.split()) <= 5 and not stripped:
        return True
    if "بطاق" in n and ("شخص" in n or "تحقيق" in n):
        if not stripped or "بطاق" in n_strip or "شخص" in n_strip or "تحقيق" in n_strip:
            return True
        return False
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
        "المنوفيه",
        "المنوفية",
        "الشرقيه",
        "الشرقية",
        "الغربيه",
        "الغربية",
        "الدقهليه",
        "الدقهلية",
        "القليوبيه",
        "القليوبية",
        "اشمون",
        "أشمون",
        "الحكماء",
        "الحكما",
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
    if _has_ocr_garbage(raw):
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
        p = re.sub(r"[.…·]{2,}", " ", p)
        p = re.sub(r"\s+", " ", p).strip(" .…·:;<>\"'")
        if not p or _has_ocr_garbage(p):
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
    if _has_ocr_garbage(text):
        return -8.0
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
        if _is_header_noise(raw) or _has_ocr_garbage(raw):
            continue
        text = _merge_line_scraps(_strip_embedded_header(raw))
        if not text or _has_ocr_garbage(text):
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

    # Prefer the classic Egyptian ID pattern:
    #   L1 short first name, L2 remaining 3–5 name words (highest signal)
    name_idxs: list[int] = []
    best_main = -1
    best_main_score = -1.0
    for i, row in enumerate(scored):
        if row["cy"] < 0.12 or row["cy"] > 0.55:
            continue
        words = [w for w in normalize_ar(row["text"]).split() if w]
        if len(words) < 3:
            continue
        if row["name_s"] < 3.0 or row["name_s"] <= row["addr_s"]:
            continue
        if _has_ocr_garbage(row["text"]) or _is_strong_address_start(row["text"]):
            continue
        # Prefer fuller remaining-name lines
        s = row["name_s"] + min(len(words), 5) * 0.5
        if s > best_main_score:
            best_main_score = s
            best_main = i
    if best_main >= 0:
        name_idxs = [best_main]
        # Short first-name line directly above
        if best_main > 0:
            prev = scored[best_main - 1]
            pw = [w for w in normalize_ar(prev["text"]).split() if w]
            if (
                prev["name_s"] >= 2.0
                and prev["name_s"] >= prev["addr_s"]
                and 1 <= len(pw) <= 2
                and not _has_ocr_garbage(prev["text"])
                and not _is_strong_address_start(prev["text"])
            ):
                name_idxs = [best_main - 1, best_main]

    # Fallback: first contiguous name-like pair in reading order
    if not name_idxs:
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
                        and not _has_ocr_garbage(nxt["text"])
                    ):
                        name_idxs.append(i + 1)
                break

    # If only one name line found, try pull previous short first-name
    if len(name_idxs) == 1 and name_idxs[0] > 0:
        prev = scored[name_idxs[0] - 1]
        if (
            prev["name_s"] >= 2.5
            and len(normalize_ar(prev["text"]).split()) <= 2
            and not _has_ocr_garbage(prev["text"])
        ):
            name_idxs = [name_idxs[0] - 1, name_idxs[0]]

    # If still empty: take the top-most 1–2 Arabic non-address lines
    if not name_idxs:
        for i, row in enumerate(scored):
            if row["cy"] < 0.12 or row["cy"] > 0.55:
                continue
            if _has_ocr_garbage(row["text"]):
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
    text = _strip_embedded_header(text)
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
        # Penalize latin OCR junk / name bleed
        latin = sum(1 for c in text if "a" <= c.lower() <= "z")
        if latin >= 2:
            score -= 3.0
    return score


def _lines_from_zone_ocr(raw_lines: list[str], *, kind: str) -> list[str]:
    """Pick up to 2 cleaned lines from a dedicated name/address zone OCR."""
    candidates: list[tuple[float, int, str]] = []
    for i, ln in enumerate(raw_lines):
        ln = _strip_embedded_header(ln)
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
            if _has_ocr_garbage(cleaned):
                continue
            if _arabic_ratio(cleaned) < 0.55:
                continue
            # Name lines must look like a person name (drop header scraps)
            if not _looks_like_person_name(cleaned) and len(normalize_ar(cleaned).split()) > 2:
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


def _tokens_in_box(
    tokens: list[OcrToken],
    box: tuple[float, float, float, float],
    *,
    pad: float = 0.02,
) -> list[OcrToken]:
    x0, y0, x1, y1 = pad_zone(box, pad)
    out: list[OcrToken] = []
    for t in tokens:
        # Prefer tokens whose center is inside; allow slight edge bleed
        if y0 <= t.cy <= y1 and x0 <= t.cx <= x1:
            out.append(t)
            continue
        # Also keep short tokens mostly overlapping the box (first name far right)
        overlap_x = max(0.0, min(t.x1, x1) - max(t.x0, x0))
        overlap_y = max(0.0, min(t.y1, y1) - max(t.y0, y0))
        tw = max(t.x1 - t.x0, 1e-6)
        th = max(t.y1 - t.y0, 1e-6)
        if (overlap_x / tw) >= 0.55 and (overlap_y / th) >= 0.45 and t.cx >= x0 - 0.05:
            out.append(t)
    return out


def _lines_from_tokens_in_box(
    tokens: list[OcrToken],
    box: tuple[float, float, float, float],
    *,
    kind: str,
    pad: float = 0.02,
) -> list[str]:
    """Build cleaned lines from full-card tokens that fall inside a layout zone."""
    x0, y0, x1, y1 = box
    if kind == "name":
        # Never read above the printed header/title band
        y0 = max(y0, _header_bottom_y(tokens) + 0.01)
    zone_toks = _tokens_in_box(tokens, (x0, y0, x1, y1), pad=pad)
    if not zone_toks:
        return []
    clustered = _cluster_tokens_into_lines(zone_toks, y_thresh=0.04)
    raw: list[str] = []
    for ln in clustered:
        text = _merge_line_scraps(_strip_embedded_header(_line_text(ln)))
        if not text:
            continue
        if kind == "name" and (_has_ocr_garbage(text) or _is_header_noise(text)):
            continue
        raw.append(text)
    return _lines_from_zone_ocr(raw, kind=kind)


def _join_field_lines(lines: list[str] | None) -> str | None:
    if not lines:
        return None
    text = " ".join(x.strip() for x in lines if x and x.strip()).strip()
    return text or None


def _name_field_score(text: str | None) -> float:
    if not text:
        return -1.0
    text = _strip_embedded_header(text)
    words = [w for w in re.split(r"\s+", normalize_ar(text)) if w]
    if not words:
        return -1.0
    score = 0.0
    # Ideal Egyptian ID: short first name + longer remainder (3–5 words total min)
    if 2 <= len(words) <= 8:
        score += 3.0
    if len(words) >= 3:
        score += 2.0
    if _looks_like_person_name(text) or all(
        _looks_like_person_name(w) or len(w) >= 2 for w in words[:2]
    ):
        score += 3.0
    if _is_strong_address_start(text) or _looks_like_address(text):
        score -= 4.0
    if _is_header_noise(text):
        score -= 5.0
    if _contains_id_header(text):
        score -= 15.0
    score += min(len(words), 6) * 0.6
    score += _arabic_ratio(text) * 2.0
    return score


def _address_field_score(text: str | None) -> float:
    if not text:
        return -1.0
    n = normalize_ar(text)
    words = [w for w in re.split(r"\s+", n) if w]
    score = 0.0
    if _looks_like_address(text) or _is_strong_address_start(text):
        score += 4.0
    if any(normalize_ar(p) in n for p in ADDRESS_PLACE_WORDS):
        score += 3.0
    if any(h in n for h in ("شارع", "مركز", "قسم", "قريه", "قرية", "ش ")):
        score += 2.0
    if "-" in text or "–" in text:
        score += 1.0
    # Prefer 2 logical chunks
    if 2 <= len(words) <= 12:
        score += 2.0
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")
    if latin >= 2:
        score -= 4.0
    if len(_digit_soup(text)) >= 10:
        score -= 12.0
    # Scrambled zone OCR often dumps too many words
    if len(words) > 12:
        score -= 3.0
    score += _arabic_ratio(text) * 1.5
    return score


def _choose_better_field(
    a: str | None,
    b: str | None,
    *,
    kind: str,
) -> str | None:
    if not a:
        return b
    if not b:
        return a
    if kind == "name":
        if _contains_id_header(a) and not _contains_id_header(b):
            return b
        if _contains_id_header(b) and not _contains_id_header(a):
            return a
        sa, sb = _name_field_score(a), _name_field_score(b)
    else:
        sa, sb = _address_field_score(a), _address_field_score(b)
    # Prefer more complete when scores close
    if abs(sa - sb) < 0.8:
        wa = len(normalize_ar(a).split())
        wb = len(normalize_ar(b).split())
        # Prefer fuller text for both name and address
        return a if wa >= wb else b
    return a if sa >= sb else b


def _scrub_name_bleed_from_address(address: str | None, name: str | None) -> str | None:
    if not address or not name:
        return address
    name_words = {normalize_ar(w) for w in name.split() if len(normalize_ar(w)) >= 3}
    if not name_words:
        return address
    words = address.split()
    # Drop leading person-name tokens before street/place content
    while words and normalize_ar(words[0]) in name_words:
        words.pop(0)
    kept: list[str] = []
    for i, w in enumerate(words):
        nw = normalize_ar(w)
        if nw in name_words and not _looks_like_address(w) and nw not in ADDRESS_PLACE_WORDS:
            # Keep only if surrounded by strong address cues
            prev_n = normalize_ar(kept[-1]) if kept else ""
            next_n = normalize_ar(words[i + 1]) if i + 1 < len(words) else ""
            if _is_strong_address_start(prev_n) or _is_strong_address_start(next_n):
                kept.append(w)
            continue
        kept.append(w)
    out = " ".join(kept).strip()
    return out or address


def pick_best_front_zones(
    engine: Any,
    card: np.ndarray,
    zone_lines_fn: Any,
    *,
    face_box: tuple[float, float, float, float] | None = None,
    tokens: list[OcrToken] | None = None,
) -> tuple[str | None, str | None, list[str]]:
    """Name/address from full-card tokens inside ID_ZONES; crop OCR only as fallback.

    Cropping zones and re-OCR often fragments Arabic lines; the full-card pass is
    usually cleaner. Zone boxes only select which tokens belong to each field.
    """
    _ = face_box
    notes: list[str] = []
    name_box = _effective_name_box(tokens or [])
    addr_box = _effective_address_box(tokens or [])
    notes.append(f"ZONE_NAME=exact{name_box}")
    notes.append(f"ZONE_ADDR=exact{addr_box}")

    name_lines: list[str] = []
    addr_lines: list[str] = []
    if tokens:
        name_lines = _lines_from_tokens_in_box(tokens, name_box, kind="name", pad=0.02)
        addr_lines = _lines_from_tokens_in_box(tokens, addr_box, kind="address", pad=0.015)
        if name_lines:
            notes.append(f"TOK_NAME← {' || '.join(name_lines)}")
        if addr_lines:
            notes.append(f"TOK_ADDR← {' || '.join(addr_lines)}")

    # Crop OCR only to fill missing lines (never primary on Egyptian print)
    need_name = len(name_lines) < 2
    need_addr = len(addr_lines) < 2
    if need_name or need_addr:
        # Non-overlapping pads: shrink shared border to avoid name↔address bleed
        nx0, ny0, nx1, ny1 = name_box
        ax0, ay0, ax1, ay1 = addr_box
        mid_y = (ny1 + ay0) / 2.0
        name_ocr_box = (max(0.0, nx0 - 0.01), max(0.0, ny0 - 0.01), min(1.0, nx1 + 0.01), mid_y)
        addr_ocr_box = (max(0.0, ax0 - 0.01), mid_y, min(1.0, ax1 + 0.01), min(1.0, ay1 + 0.01))
        if need_name:
            name_raw = zone_lines_fn(
                engine, card, name_ocr_box[1], name_ocr_box[3], name_ocr_box[0], name_ocr_box[2], scale=1.7
            )
            crop_name = _lines_from_zone_ocr(name_raw, kind="name")
            if crop_name:
                notes.append(f"CROP_NAME← {' || '.join(crop_name)}")
                crop_j = _join_field_lines(crop_name)
                tok_j = _join_field_lines(name_lines)
                # Never let a weaker/incomplete crop overwrite clean full-card tokens
                # (common: خالاد بدل خالد، أو فقدان الاسم الأول «محمد»)
                if name_lines and len(name_lines) >= len(crop_name):
                    if _name_field_score(crop_j) <= _name_field_score(tok_j) + 1.2:
                        pass
                    elif _name_field_score(crop_j) > _name_field_score(tok_j) + 2.5:
                        name_lines = crop_name
                elif _name_field_score(crop_j) > _name_field_score(tok_j):
                    name_lines = crop_name
                elif len(name_lines) < len(crop_name):
                    # Merge missing short first name
                    for ln in crop_name:
                        if not any(normalize_ar(ln) == normalize_ar(x) for x in name_lines):
                            if _looks_like_person_name(ln) and len(normalize_ar(ln).split()) <= 2:
                                name_lines = [ln, *name_lines][:2]
                                break
                elif not name_lines:
                    name_lines = crop_name
        if need_addr:
            addr_raw = zone_lines_fn(
                engine, card, addr_ocr_box[1], addr_ocr_box[3], addr_ocr_box[0], addr_ocr_box[2], scale=1.7
            )
            crop_addr = _lines_from_zone_ocr(addr_raw, kind="address")
            if crop_addr:
                notes.append(f"CROP_ADDR← {' || '.join(crop_addr)}")
                if _address_field_score(_join_field_lines(crop_addr)) > _address_field_score(
                    _join_field_lines(addr_lines)
                ):
                    addr_lines = crop_addr

    if name_lines:
        addr_lines = [
            a
            for a in addr_lines
            if not any(normalize_ar(a) == normalize_ar(n) for n in name_lines)
        ][:2]

    name = _join_field_lines(name_lines)
    addr = _join_field_lines(addr_lines)
    addr = _scrub_name_bleed_from_address(addr, name)
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
        soft = decode_national_id(nid)
        if soft.get("birth_date"):
            fields["birth_date"] = soft["birth_date"]
            fields["governorate"] = soft.get("governorate") or fields.get("governorate")
            fields["gender"] = soft.get("gender") or fields.get("gender")
            decoded = soft
            rules_fired.append("EQ_SOFT: تاريخ الميلاد من الرقم القومي")
        else:
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
            if k == "job":
                fields[k] = finalize_back_job(v)
            else:
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
                fields["job"] = finalize_back_job(job)
                rules_fired.append("LABEL: المهنة")
        else:
            fields["job"] = finalize_back_job(fields.get("job"))

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
            strip_big = cv2.resize(strip, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
            blocks_c = engine.ocr_bgr(strip_big, enhance=False)
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

    @staticmethod
    def _onnxtr_tokens(
        image_bgr: np.ndarray,
        *,
        index_base: int = 7000,
        origin: tuple[float, float, float, float] | None = None,
    ) -> list[OcrToken]:
        """Run secondary OnnxTR Arabic OCR; map boxes into card-normalized space."""
        from arabic_onnxtr import ArabicOnnxTrEngine

        if image_bgr is None or not getattr(image_bgr, "size", 0):
            return []
        if not ArabicOnnxTrEngine.available():
            return []
        engine = ArabicOnnxTrEngine.get()
        blocks = engine.ocr_bgr(image_bgr)
        h, w = image_bgr.shape[:2]
        ox0 = oy0 = 0.0
        ox1 = oy1 = 1.0
        if origin is not None:
            ox0, oy0, ox1, oy1 = origin
        out: list[OcrToken] = []
        for i, b in enumerate(blocks):
            text = apply_ocr_word_fixes(str(b.get("text") or "").strip())
            if not text:
                continue
            # Pixel → local norm → card norm
            lx0 = float(b["x0"]) / max(w, 1)
            ly0 = float(b["y0"]) / max(h, 1)
            lx1 = float(b["x1"]) / max(w, 1)
            ly1 = float(b["y1"]) / max(h, 1)
            out.append(
                OcrToken(
                    index=index_base + i,
                    text=text,
                    score=float(b.get("score") or 0.0),
                    x0=ox0 + lx0 * (ox1 - ox0),
                    y0=oy0 + ly0 * (oy1 - oy0),
                    x1=ox0 + lx1 * (ox1 - ox0),
                    y1=oy0 + ly1 * (oy1 - oy0),
                )
            )
        return out

    def _read_national_id_lines(
        self,
        engine: Any,
        card: np.ndarray,
    ) -> list[str]:
        """One tight upscaled NID strip + optional OnnxTR — avoid multi-variant crashes."""
        box = ID_ZONES["front"]["national_id"]
        classic = (max(0.30, box[0]), max(0.68, box[1]), min(0.99, box[2]), min(0.90, box[3]))
        lines: list[str] = []
        for zone in (classic, box):
            roi = crop_norm_region(card, zone, min_h=32)
            if roi is None or not roi.size:
                continue
            big = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            try:
                blocks = engine.ocr_bgr(big, enhance=False)
                lines.extend(str(b.text or "").strip() for b in blocks if str(b.text or "").strip())
            except Exception:  # noqa: BLE001
                logger.debug("NID paddle strip failed", exc_info=True)
            # Secondary engine — often better on Indic digits
            try:
                import os

                if os.environ.get("OCR_SKIP_ONNXTR", "").strip() in {"1", "true", "yes"}:
                    pass
                else:
                    from arabic_onnxtr import ArabicOnnxTrEngine

                    if ArabicOnnxTrEngine.available():
                        for b in ArabicOnnxTrEngine.get().ocr_bgr(big):
                            t = str(b.get("text") or "").strip()
                            if t:
                                lines.append(t)
            except Exception:  # noqa: BLE001
                logger.debug("NID OnnxTR strip failed", exc_info=True)
            if best_national_id_from_texts(lines):
                break
        return [ln for ln in lines if ln]

    def process_image_path(
        self,
        image_path: str | Path,
        *,
        enhance_handwriting: bool = False,
        forced_side: str | None = None,
    ) -> EgyptianIdResult:
        from ocr_engine import ArabicOcrEngine

        original = _load_bgr(image_path)
        # 1) Orient the photo FIRST — DeepLab/zones fail on upside-down cards
        oriented, orient_ang, orient_note = correct_card_orientation(original)
        preview = oriented.copy()
        ph, pw = preview.shape[:2]
        if max(ph, pw) > 1200:
            s = 1200 / max(ph, pw)
            preview = cv2.resize(preview, (int(pw * s), int(ph * s)))

        # 2) Crop card from the upright image
        card = oriented
        crop_method = "raw"
        try:
            from card_crop import crop_id_card

            card, crop_method = crop_id_card(
                oriented,
                fallback_quad_fn=detect_card_quad,
                warp_fn=warp_card,
            )
        except Exception:  # noqa: BLE001
            logger.debug("ML card crop unavailable", exc_info=True)
            quad = detect_card_quad(oriented)
            if quad is not None:
                try:
                    card = warp_card(oriented, quad)
                    crop_method = "contour"
                except Exception:  # noqa: BLE001
                    logger.debug("warp failed", exc_info=True)
                    card = oriented
        card = auto_deskew_card(card)
        # 3) Second pass after crop (warp can leave card rotated)
        card, orient_ang2, orient_note2 = correct_card_orientation(card)
        if orient_ang2:
            orient_note = (
                f"{orient_note} → بعد القص: تدوير {orient_ang2}°"
                if orient_ang
                else orient_note2
            )
            orient_ang = (orient_ang + orient_ang2) % 360

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

        # Secondary open-source Arabic engine (OnnxTR) — keep SEPARATE from Paddle
        # tokens so noisy words do not pollute name/address line picking.
        onnx_tokens: list[OcrToken] = []
        onnx_notes: list[str] = []
        try:
            import os

            if os.environ.get("OCR_SKIP_ONNXTR", "").strip() in {"1", "true", "yes"}:
                onnx_notes.append("ONNXTR: skipped")
            else:
                onnx_tokens = self._onnxtr_tokens(ocr_card_enh, index_base=7000)
                if onnx_tokens:
                    onnx_notes.append(f"ONNXTR: {len(onnx_tokens)} توكن ثانوي (منفصل)")
        except Exception:  # noqa: BLE001
            logger.debug("OnnxTR secondary OCR skipped", exc_info=True)
            onnx_notes.append("ONNXTR: غير متاح / تخطي")

        if forced_side in {"front", "back"}:
            side = forced_side
            rules_side = [
                f"SIDE_FORCED: {side}",
                f"CROP: {crop_method}",
                orient_note,
            ]
        else:
            side = guess_side(tokens + onnx_tokens, ocr_card)
            rules_side = [
                f"SIDE_DETECT: {side}",
                f"CROP: {crop_method}",
                orient_note,
            ]
        _ = orient_ang
        fields, decoded, rules = apply_field_rules(
            tokens, side, face_box=face_box if side != "back" else None
        )
        rules = rules_side + onnx_notes + rules

        # FRONT: name/address — zone token pick + quality merge (never blind overwrite)
        if side != "back":
            seq_name = fields.get("full_name")
            seq_addr = fields.get("address")
            z_name, z_addr, z_notes = pick_best_front_zones(
                engine,
                ocr_card_enh,
                self._zone_lines,
                face_box=face_box,
                tokens=tokens,
            )
            rules.extend(z_notes)

            best_name = _choose_better_field(seq_name, z_name, kind="name")
            best_addr = _choose_better_field(seq_addr, z_addr, kind="address")

            o_name: str | None = None
            o_addr: str | None = None
            # OnnxTR: village fill only — never replace full name/address
            if onnx_tokens:
                o_name, o_addr, o_notes = pick_best_front_zones(
                    engine,
                    ocr_card_enh,
                    self._zone_lines,
                    face_box=face_box,
                    tokens=onnx_tokens,
                )
                rules.extend([f"ONNXTR_{n}" if not n.startswith("ONNX") else n for n in o_notes[:6]])
                o_name = finalize_front_name(o_name)
                o_addr = finalize_front_address(o_addr, national_id=fields.get("national_id"))
                if o_name and _name_field_score(o_name) > _name_field_score(best_name) + 3.0:
                    if not _contains_id_header(o_name):
                        best_name = o_name
                        rules.append("ONNXTR_WIN: اسم (مُصفّى)")
                if o_addr and best_addr:
                    for w in (o_addr or "").split():
                        nw = normalize_ar(w)
                        if not nw or len(nw) < 3:
                            continue
                        if any(ch in w for ch in ".…·:;<>\"'"):
                            continue
                        if _has_ocr_garbage(w):
                            continue
                        if nw in normalize_ar(best_addr):
                            continue
                        if (
                            len(nw.split()) == 1
                            and _arabic_ratio(w) >= 0.8
                            and not _is_strong_address_start(w)
                            and (
                                nw in ADDRESS_PLACE_WORDS
                                or (
                                    _looks_like_person_name(w)
                                    and not any(
                                        normalize_ar(x) == nw for x in (best_name or "").split()
                                    )
                                )
                            )
                        ):
                            best_addr = f"{w} {best_addr}".strip()
                            rules.append(f"ONNXTR_FILL: {w}")
                            break

            best_name = finalize_front_name(best_name)
            best_addr = finalize_front_address(
                best_addr, national_id=fields.get("national_id")
            )
            # Fallback chain if sanitization wiped a bad pick
            if not best_name:
                for cand in (z_name, seq_name, o_name if onnx_tokens else None):
                    best_name = finalize_front_name(cand)
                    if best_name:
                        break
            if not best_addr:
                for cand in (z_addr, seq_addr, o_addr if onnx_tokens else None):
                    best_addr = finalize_front_address(
                        cand, national_id=fields.get("national_id")
                    )
                    if best_addr:
                        break

            best_addr = _scrub_name_bleed_from_address(best_addr, best_name)
            best_addr = finalize_front_address(
                best_addr, national_id=fields.get("national_id")
            )

            if best_name:
                fields["full_name"] = best_name
                rules.append("ZONE_MERGE: الاسم (تسلسل OCR + مناطق)")
            if best_addr:
                fields["address"] = best_addr
                rules.append("ZONE_MERGE: العنوان (تسلسل OCR + مناطق)")

            # NID strip: read from ORIENTED card (after 180° fix). Raw original
            # is upside-down when user flips the photo and pollutes digits.
            nid_lines = self._read_national_id_lines(engine, ocr_card_enh)
            if not best_national_id_from_texts(nid_lines):
                nid_lines += self._read_national_id_lines(engine, ocr_card)
            # Prefer strip-only result — full-card token soup invents false NIDs
            # (e.g. 275… / 298…) that still pass checksum.
            strip_nid = best_national_id_from_texts(nid_lines)
            tok_nid = best_national_id_from_texts([t.text for t in tokens])
            nid = strip_nid
            if not nid:
                nid = tok_nid
            elif tok_nid and tok_nid != strip_nid:
                # Keep strip unless token match is clearly better AND grounded
                ss = score_national_id_candidate(strip_nid)
                ts = score_national_id_candidate(tok_nid)
                if ts >= ss + 4 and _nid_grounded_in_texts(tok_nid, nid_lines):
                    nid = tok_nid
                    rules.append(f"NID_TOK_WIN: {strip_nid} ← {tok_nid}")
            if not nid:
                nid = find_national_id_from_tokens(tokens)
            if nid and score_national_id_candidate(nid) >= 11:
                prev = fields.get("national_id")
                if prev and str(prev) != nid:
                    rules.append(f"NID_REPLACE: {prev} ← {nid}")
                fields["national_id"] = nid
                decoded2 = decode_national_id(nid)
                if decoded2.get("birth_date"):
                    decoded = decoded2
                    fields["birth_date"] = decoded2.get("birth_date")
                    fields["governorate"] = decoded2.get("governorate") or fields.get(
                        "governorate"
                    )
                    fields["gender"] = decoded2.get("gender") or fields.get("gender")
                    rules.append(
                        f"ZONE_NID: {fields['national_id']} → ميلاد {fields.get('birth_date')}"
                    )
            elif fields.get("national_id") and score_national_id_candidate(
                str(fields["national_id"])
            ) < 11:
                rules.append(f"NID_CLEAR: رفض {fields.get('national_id')}")
                fields["national_id"] = None
                fields["birth_date"] = None
                fields["governorate"] = None
                fields["gender"] = None

            # Printed DOB under photo — only if equation DOB still missing
            # (print is faint over eagle watermark; extra OCR often crashes Paddle on Win)
            dob_box = birth_date_zone_box(face_box)
            dob_lines: list[str] = []
            if not fields.get("birth_date"):
                bx0, by0, bx1, by1 = pad_zone(dob_box, 0.01)
                dob_lines = self._zone_lines(
                    engine, ocr_card_enh, by0, by1, bx0, bx1, scale=2.0, mode="text"
                )
                printed_dob = extract_printed_birth_date(dob_lines)
                if printed_dob:
                    fields["birth_date"] = printed_dob
                    rules.append(f"ZONE_DOB: تاريخ الميلاد المطبوع ← {printed_dob}")
            else:
                rules.append(
                    f"DOB_EQ: تاريخ الميلاد من الرقم القومي ← {fields.get('birth_date')}"
                )

            # Final guarantee: DOB from national ID equation
            if fields.get("national_id"):
                decoded3 = decode_national_id(str(fields["national_id"]))
                if decoded3.get("birth_date"):
                    if not fields.get("birth_date"):
                        rules.append("EQ_FALLBACK: تاريخ الميلاد من الرقم القومي")
                    fields["birth_date"] = decoded3["birth_date"]
                    fields["governorate"] = decoded3.get("governorate") or fields.get(
                        "governorate"
                    )
                    fields["gender"] = decoded3.get("gender") or fields.get("gender")
                    decoded = decoded3

            # Last resort: any date-like OCR already gathered
            if not fields.get("birth_date"):
                printed_all = extract_printed_birth_date(
                    [t.text for t in tokens] + dob_lines
                )
                if printed_all:
                    fields["birth_date"] = printed_all
                    rules.append(f"DOB_SCAN: {printed_all}")

            # Drop invalid NID that cannot produce a birth date
            if fields.get("national_id") and not fields.get("birth_date"):
                if score_national_id_candidate(str(fields["national_id"])) < 11:
                    rules.append(
                        f"NID_REJECT: {fields.get('national_id')} (تاريخ غير صالح)"
                    )
                    fields["national_id"] = None

        # Back-side dedicated zones (job / status / expiry) — above barcode
        if side == "back":
            jx0, jy0, jx1, jy1 = pad_zone(ID_ZONES["back"]["job"], 0.015)
            sx0, sy0, sx1, sy1 = pad_zone(ID_ZONES["back"]["status"], 0.01)
            ex0, ey0, ex1, ey1 = pad_zone(ID_ZONES["back"]["expiry"], 0.015)
            job_zone = [
                apply_ocr_word_fixes(x)
                for x in self._zone_lines(engine, ocr_card_enh, jy0, jy1, jx0, jx1)
            ]
            status_zone = [
                apply_ocr_word_fixes(x)
                for x in self._zone_lines(engine, ocr_card_enh, sy0, sy1, sx0, sx1)
            ]
            expiry_zone = self._zone_lines(
                engine, ocr_card_enh, ey0, ey1, ex0, ex1, scale=2.0, mode="text"
            )
            if not extract_expiry_date(expiry_zone):
                expiry_zone = expiry_zone + self._zone_lines(
                    engine, ocr_card_enh, ey0, ey1, ex0, ex1, scale=2.2, mode="digits"
                )
            # Also scan mid-band tokens (سارية حتى often sits just above barcode)
            expiry_band_texts = [
                t.text
                for t in tokens
                if 0.34 <= t.cy <= 0.68
            ]
            zone_toks = merge_token_lists(
                tokens,
                self._zone_tokens(engine, ocr_card_enh, jy0, jy1, jx0, jx1, index_base=4000),
                self._zone_tokens(engine, ocr_card_enh, sy0, sy1, sx0, sx1, index_base=5000),
                self._zone_tokens(engine, ocr_card_enh, ey0, ey1, ex0, ex1, index_base=6000),
                [t for t in onnx_tokens if t.cy <= 0.62],
            )
            # OnnxTR zone pass for back text bands
            try:
                for band, (y0b, y1b, x0b, x1b), base in (
                    ("job", (jy0, jy1, jx0, jx1), 7400),
                    ("status", (sy0, sy1, sx0, sx1), 7500),
                    ("expiry", (ey0, ey1, ex0, ex1), 7600),
                ):
                    _ = band
                    h, w = ocr_card_enh.shape[:2]
                    roi = ocr_card_enh[int(h * y0b) : int(h * y1b), int(w * x0b) : int(w * x1b)]
                    zone_toks = merge_token_lists(
                        zone_toks,
                        self._onnxtr_tokens(
                            roi, index_base=base, origin=(x0b, y0b, x1b, y1b)
                        ),
                    )
            except Exception:  # noqa: BLE001
                logger.debug("OnnxTR back zones skipped", exc_info=True)

            back2, notes2 = extract_back_fields(zone_toks)
            for k, v in back2.items():
                if not v:
                    continue
                if k == "gender" and fields.get("gender"):
                    continue
                if k == "national_id" and fields.get("national_id"):
                    continue
                if k == "job":
                    fields[k] = finalize_back_job(v)
                elif k in {"expiry_date", "marital_status", "religion", "gender", "husband_name"}:
                    fields[k] = v
                elif not fields.get(k):
                    fields[k] = v
            rules.extend(notes2)
            rules.append("ZONE_BACK: قراءة مناطق الظهر (مهنة/حالة/سريان)")

            # Job: prefer a single clean zone line (no duplicate merge)
            if job_zone:
                best_job = None
                best_s = -1.0
                for ln in job_zone:
                    cleaned = finalize_back_job(ln)
                    if not cleaned or len(cleaned) < 3:
                        continue
                    n = normalize_ar(cleaned)
                    if len(n) < 6 and not any(normalize_ar(h) in n for h in JOB_HINTS):
                        continue
                    st = _parse_status_triplet(cleaned)
                    if sum(1 for v in st.values() if v) >= 2:
                        continue
                    if _has_ocr_garbage(cleaned) and not any(
                        normalize_ar(h) in n for h in JOB_HINTS
                    ):
                        continue
                    s = float(len(cleaned))
                    if any(normalize_ar(h) in n for h in JOB_HINTS):
                        s += 20
                    if s > best_s:
                        best_s = s
                        best_job = cleaned
                if best_job:
                    cur = fields.get("job")
                    if not cur or _jobs_near_duplicate(str(cur), best_job):
                        fields["job"] = best_job
                    elif any(normalize_ar(h) in normalize_ar(best_job) for h in JOB_HINTS) and not any(
                        normalize_ar(h) in normalize_ar(str(cur)) for h in JOB_HINTS
                    ):
                        fields["job"] = best_job

            fields["job"] = finalize_back_job(fields.get("job"))
            # Drop weak hallucinated profession on partial back crops
            jn = normalize_ar(str(fields.get("job") or ""))
            if jn and len(jn) < 6 and not any(normalize_ar(h) in jn for h in JOB_HINTS):
                fields["job"] = None
                rules.append("BACK_JOB: تجاهل مهنة ضعيفة/مشوهة")

            if not fields.get("expiry_date"):
                exp = extract_expiry_date(
                    expiry_zone + expiry_band_texts + status_zone + job_zone
                    + [t.text for t in tokens if t.cy <= 0.70]
                )
                if exp:
                    fields["expiry_date"] = exp
                    rules.append(f"ZONE_EXPIRY: سريان ← {exp}")
            else:
                # Prefer سارية-line parse over an earlier digit-soup guess
                prefer = [
                    t
                    for t in (expiry_zone + expiry_band_texts + [t.text for t in tokens])
                    if t
                    and (
                        "ساري" in normalize_ar(t)
                        or "حتى" in normalize_ar(t)
                        or "حتي" in normalize_ar(t)
                    )
                ]
                exp2 = extract_expiry_date(prefer + expiry_zone + expiry_band_texts)
                if exp2 and exp2 != fields.get("expiry_date"):
                    fields["expiry_date"] = exp2
                    rules.append(f"ZONE_EXPIRY: تحديث السريان ← {exp2}")
                elif prefer:
                    exp3 = extract_expiry_date(prefer)
                    if exp3:
                        fields["expiry_date"] = exp3
                        rules.append(f"ZONE_EXPIRY: سريان من سطر البطاقة ← {exp3}")

            # Status band: parse gender/religion/marital explicitly
            for ln in status_zone + job_zone:
                st = _parse_status_triplet(ln)
                for k, v in st.items():
                    if v and (not fields.get(k) or k != "gender"):
                        if k == "gender" and fields.get("gender"):
                            continue
                        fields[k] = v

            status_blob = " ".join(status_zone)
            st = _parse_status_triplet(status_blob)
            for k, v in st.items():
                if v and not fields.get(k):
                    fields[k] = v

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
        crop_pack = build_field_crops(
            card, tokens, fields, face, side=side, face_box=face_box
        )
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
            f"[v2026-09-14e · قص={crop_method}] تم ضبط {filled} حقل · {crop_n} قصّة ({fields.get('card_side')})."
            if filled
            else f"[v2026-09-14e · قص={crop_method}] القراءة ضعيفة — صوّر أوضح أو ارفع الوجه الآخر."
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
    "birth_date": ("front", "birth_date"),
    "face": ("front", "face"),
    "job": ("back", "job"),
    "religion": ("back", "status"),
    "marital_status": ("back", "status"),
    "gender": ("back", "status"),
    "husband_name": ("back", "status"),
    "expiry_date": ("back", "expiry"),
}


def zone_box_for_field(
    field_key: str,
    side: str,
    *,
    tokens: list[OcrToken] | None = None,
    face_box: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    mapping = FIELD_TO_ZONE.get(field_key)
    if not mapping:
        return None
    z_side, z_name = mapping
    if side == "front" and z_side != "front":
        return None
    if side == "back" and z_side != "back":
        return None
    if z_side == "front" and tokens is not None:
        if z_name == "name":
            return _effective_name_box(tokens)
        if z_name == "address":
            return _effective_address_box(tokens)
        if z_name == "birth_date":
            return birth_date_zone_box(face_box)
    if z_side == "front" and z_name == "birth_date":
        return birth_date_zone_box(face_box)
    return ID_ZONES.get(z_side, {}).get(z_name)


def build_field_crops(
    card: np.ndarray,
    tokens: list[OcrToken],
    fields: dict[str, Any],
    face: np.ndarray | None,
    *,
    side: str = "front",
    face_box: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Crop previews: prefer OCR-token union clipped to layout zone (tight & accurate)."""
    crop_keys_front = ["full_name", "address", "national_id", "birth_date"]
    crop_keys_back = [
        "job",
        "religion",
        "marital_status",
        "gender",
        "husband_name",
        "expiry_date",
    ]
    crop_keys = crop_keys_front if side != "back" else crop_keys_back
    layout = "front" if side != "back" else "back"

    crops: dict[str, str] = {}
    meta: dict[str, Any] = {}

    for key in crop_keys:
        zone_box = zone_box_for_field(
            key, layout, tokens=tokens, face_box=face_box
        )
        if zone_box is None:
            continue
        value = fields.get(key)
        # NID/DOB: always use full layout zone — token union often clips
        # leading digits (NID) or misses faint print (DOB under eagle).
        if key in {"birth_date", "national_id"}:
            tight = None
        else:
            tight = _token_crop_box_for_field(tokens, key, value, zone_box)
        box = tight or zone_box
        # Extra pad on digit fields so UI crops aren't edge-clipped
        if key in {"birth_date", "national_id"}:
            box = pad_zone(box, 0.01 if key == "national_id" else 0.008)
        roi = crop_norm_region(card, box, min_h=56 if key == "birth_date" else 36)
        if roi is not None and roi.size:
            crops[key] = _encode_image_b64(roi, quality=90)
            meta[key] = {
                "bbox_norm": [round(x, 4) for x in box],
                "zone": FIELD_TO_ZONE[key][1],
                "value": value,
                "crop_mode": "tokens" if tight else "zone",
            }

    annotated = draw_id_zones(card, layout, thickness=2)
    # Draw actual crop boxes used in UI (cyan) so user sees what was cropped
    h, w = annotated.shape[:2]
    for key, info in meta.items():
        bb = info.get("bbox_norm")
        if not bb:
            continue
        x0, y0, x1, y1 = bb
        cv2.rectangle(
            annotated,
            (int(x0 * w), int(y0 * h)),
            (int(x1 * w), int(y1 * h)),
            (255, 200, 40),
            2,
        )

    out: dict[str, Any] = {
        "annotated": _encode_image_b64(annotated, quality=88),
        "crops": crops,
        "crop_meta": meta,
    }
    if layout == "front":
        face_roi = crop_norm_region(card, ID_ZONES["front"]["face"])
        if face is not None and face.size:
            out["face"] = _encode_image_b64(face, quality=90)
        elif face_roi is not None and face_roi.size:
            out["face"] = _encode_image_b64(face_roi, quality=90)
    return out


def _token_crop_box_for_field(
    tokens: list[OcrToken],
    field_key: str,
    value: Any,
    zone_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Build a tight bbox from tokens whose text appears in the field value,
    constrained to stay inside the layout zone (± small pad).
    """
    if not tokens or not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    n_val = normalize_ar(text)
    words = {w for w in n_val.split() if len(w) >= 2}
    if not words and field_key != "national_id":
        return None

    zx0, zy0, zx1, zy1 = zone_box
    # Slightly expand zone for matching; final box still clipped
    pad_z = 0.03
    mx0, my0 = max(0.0, zx0 - pad_z), max(0.0, zy0 - pad_z)
    mx1, my1 = min(1.0, zx1 + pad_z), min(1.0, zy1 + pad_z)

    hits: list[OcrToken] = []
    for t in tokens:
        if not (mx0 <= t.cx <= mx1 and my0 <= t.cy <= my1):
            continue
        tn = t.norm_text
        if not tn:
            continue
        if field_key == "national_id":
            digits = _digit_soup(t.text)
            if len(digits) >= 6 and digits in _digit_soup(text):
                hits.append(t)
            continue
        # Token is a word from the field, or field contains the token
        if tn in words or any(tn in w or w in tn for w in words if len(w) >= 3):
            # Skip header bleed for name
            if field_key == "full_name" and (
                "بطاق" in tn or "تحقيق" in tn or "شخص" in tn or "جمهور" in tn
            ):
                continue
            if field_key == "address" and len(_digit_soup(t.text)) >= 10:
                continue
            hits.append(t)

    if len(hits) < 1:
        return None
    # Need at least 2 hits for multi-word fields, else keep zone
    if field_key in {"full_name", "address", "job"} and len(hits) < 2:
        # Single strong hit still OK if it covers most of the value
        if len(hits) == 1 and len(normalize_ar(hits[0].text)) < max(4, len(n_val) // 2):
            return None

    bx0 = min(t.x0 for t in hits)
    by0 = min(t.y0 for t in hits)
    bx1 = max(t.x1 for t in hits)
    by1 = max(t.y1 for t in hits)
    # Pad slightly, then clamp to zone (name: allow a bit above zone for first line)
    pad = 0.015
    top_slack = 0.04 if field_key == "full_name" else 0.0
    bx0 = max(max(0.0, zx0 - 0.02), bx0 - pad)
    by0 = max(max(0.0, zy0 - top_slack), by0 - pad)
    bx1 = min(min(1.0, zx1 + 0.02), bx1 + pad)
    by1 = min(zy1 + 0.01, by1 + pad)
    if bx1 - bx0 < 0.05 or by1 - by0 < 0.02:
        return None
    return (bx0, by0, bx1, by1)