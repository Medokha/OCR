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

FRONT_HINTS = ("جمهوريه", "جمهورية", "مصر", "العربيه", "العربية", "بطاقه", "بطاقة", "تحقيق", "الشخصيه", "الشخصية", "قومي")
BACK_HINTS = ("المهنه", "المهنة", "الديانه", "الديانة", "الحاله", "الحالة", "اجتماعيه", "اجتماعية", "الجنس", "ذكر", "انثى", "أنثى")
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
MARITAL_VALUES = ("اعزب", "أعزب", "متزوج", "مطلق", "أرمل", "ارمل", "آنسة", "انسه", "متزوجة", "مطلقة", "ارملة")
GENDER_VALUES = ("ذكر", "انثى", "أنثى", "انثي")


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


def crop_face(image_bgr: np.ndarray, pad: float = 0.28) -> np.ndarray | None:
    faces = detect_faces(image_bgr)
    if not faces:
        h, w = image_bgr.shape[:2]
        x0, y0 = int(w * 0.68), int(h * 0.16)
        x1, y1 = int(w * 0.97), int(h * 0.74)
        roi = image_bgr[y0:y1, x0:x1]
        return roi if roi.size else None
    x, y, bw, bh = faces[0]
    px, py = int(bw * pad), int(bh * pad)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(image_bgr.shape[1], x + bw + px)
    y1 = min(image_bgr.shape[0], y + bh + py)
    return image_bgr[y0:y1, x0:x1]


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


def guess_side(tokens: list[OcrToken]) -> str:
    blob = normalize_ar(" ".join(t.text for t in tokens))
    front = sum(1 for h in FRONT_HINTS if h in blob)
    back = sum(1 for h in BACK_HINTS if h in blob)
    if back >= 2 and back >= front:
        return "back"
    if front >= 1 or find_national_id_from_tokens(tokens):
        return "front"
    return "unknown"


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
    """Heuristic: Arabic name line (1–4 words), not address/place."""
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
    if any(k in n for k in ("جمهوريه", "بطاقه", "وزاره", "تحقيق", "شخصيه", "قومي", "اقامه")):
        return False
    words = [w for w in re.split(r"\s+", n) if w]
    if not (1 <= len(words) <= 4):
        return False
    # Reject very short OCR scraps like "قية" / "شر" unless common name length >= 3
    if len(words) == 1 and len(words[0]) < 3:
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


def extract_name_and_address(
    tokens: list[OcrToken], side: str
) -> tuple[str | None, str | None, list[str]]:
    """Egyptian ID front: name = 2 lines, address = 2 lines (clustered by Y).

    Ground truth layout from real cards:
      name L1: محمود
      name L2: عصام عبدالعزيز قطب محمد
      addr L1: ٩ ش عبدالقادر - الحكماء
      addr L2: الزقازيق ثان - الشرقية
    """
    notes: list[str] = []
    if side == "back":
        return None, None, notes

    # Prefer left-of-photo column, but keep tokens that are clearly text
    col = [t for t in tokens if t.cx <= 0.78]
    if len(col) < 3:
        col = list(tokens)

    lines = _cluster_tokens_into_lines(col)
    line_strs: list[str] = []
    line_meta: list[tuple[float, str]] = []  # (cy, text)
    for ln in lines:
        raw = _line_text(ln)
        if not raw:
            continue
        cy = sum(t.cy for t in ln) / len(ln)
        # Skip NID-only lines
        if len(_digit_soup(raw)) >= 10 and _arabic_ratio(raw) < 0.35:
            continue
        if _is_header_noise(raw):
            continue
        text = _merge_line_scraps(raw)
        if not text:
            continue
        line_strs.append(text)
        line_meta.append((cy, text))

    notes.append(f"LINES={len(line_meta)}")

    name_lines: list[str] = []
    addr_lines: list[str] = []
    phase = "seek_name"

    for cy, text in line_meta:
        n = normalize_ar(text)

        # Skip pure label lines (keep going)
        if _is_address_label(text) and len(_strip_address_label_prefix(text)) < 3:
            if phase == "name" and name_lines:
                phase = "address"
            notes.append("LINE: لابل محل الإقامة")
            continue

        if phase == "seek_name":
            if cy < 0.10:
                continue
            if _looks_like_person_name(text) or (
                _arabic_ratio(text) >= 0.7 and len(_digit_soup(text)) == 0 and not _is_strong_address_start(text)
            ):
                # Accept short first-name line like «محمود» / even OCR «مود»
                if len(n) >= 2:
                    name_lines.append(text)
                    phase = "name"
            continue

        if phase == "name":
            if _is_strong_address_start(text) or _is_address_label(text):
                phase = "address"
                rest = _strip_address_label_prefix(text)
                if rest and (_is_strong_address_start(rest) or _looks_like_address(rest) or _digit_soup(rest)):
                    addr_lines.append(_merge_line_scraps(rest))
                continue
            # Second name line (usual): remaining full name
            if len(name_lines) < 2 and (
                _looks_like_person_name(text)
                or (_arabic_ratio(text) >= 0.65 and len(_digit_soup(text)) == 0 and not _looks_like_address(text))
            ):
                name_lines.append(text)
                # After 2 name lines, next content is address
                if len(name_lines) >= 2:
                    phase = "expect_address"
                continue
            # Extra name words rarely on 3rd line — only if still clearly names and high
            if len(name_lines) < 3 and _looks_like_person_name(text) and cy < 0.50:
                name_lines.append(text)
                continue
            # Otherwise treat as address start
            phase = "address"
            if _arabic_ratio(text) >= 0.3 or _digit_soup(text):
                addr_lines.append(_merge_line_scraps(_strip_address_label_prefix(text)))
            continue

        if phase == "expect_address":
            if _is_address_label(text) and len(_strip_address_label_prefix(text)) < 3:
                phase = "address"
                continue
            phase = "address"
            # fall through

        if phase == "address":
            if len(_digit_soup(text)) >= 10 and _arabic_ratio(text) < 0.4:
                break
            rest = _strip_address_label_prefix(text)
            rest = _merge_line_scraps(rest)
            if not rest:
                continue
            # Don't pull name lines into address
            if any(normalize_ar(rest) == normalize_ar(nl) for nl in name_lines):
                continue
            if _looks_like_person_name(rest) and not _looks_like_address(rest) and not _digit_soup(rest) and len(addr_lines) == 0:
                # misplaced name
                if len(name_lines) < 2:
                    name_lines.append(rest)
                continue
            addr_lines.append(rest)
            if len(addr_lines) >= 2:
                phase = "done"
                break

    # Fallback: first two non-header lines = name, next two = address
    if not name_lines and line_meta:
        cand = [t for _, t in line_meta if _arabic_ratio(t) >= 0.55 and len(_digit_soup(t)) < 6]
        name_lines = cand[:2]
        addr_lines = [t for t in cand[2:4] if t not in name_lines]
        notes.append("FALLBACK: أول سطرين اسم / التاليين عنوان")

    if name_lines and not addr_lines:
        # take following lines from meta after last name
        seen_name = False
        for _, text in line_meta:
            if text in name_lines:
                seen_name = True
                continue
            if not seen_name:
                continue
            if _is_address_label(text) and len(_strip_address_label_prefix(text)) < 3:
                continue
            rest = _merge_line_scraps(_strip_address_label_prefix(text))
            if rest and rest not in name_lines:
                addr_lines.append(rest)
            if len(addr_lines) >= 2:
                break
        if addr_lines:
            notes.append("FALLBACK: عنوان بعد سطور الاسم")

    # Final: Egyptian ID typically exactly 2 name lines + 2 address lines
    name_lines = name_lines[:2]
    addr_lines = _clean_address_parts(addr_lines)[:2]

    full_name = " ".join(name_lines) if name_lines else None
    address = " ".join(addr_lines) if addr_lines else None

    if full_name:
        notes.append(f"NAME← {' || '.join(name_lines)}")
    if address:
        notes.append(f"ADDR← {' || '.join(addr_lines)}")

    return full_name, address, notes


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

    # RULE 2+3: sequential OCR mapping for name then address (same lines as raw OCR)
    name, address, seq_notes = extract_name_and_address(tokens, side)
    if name:
        fields["full_name"] = name
        rules_fired.append("SEQ: الاسم من سطور OCR بعد الهيدر مباشرة")
    if address:
        fields["address"] = address
        rules_fired.append("SEQ: العنوان/محل الإقامة بعد الاسم أو بعد لابل محل الإقامة")
    rules_fired.extend(seq_notes)

    # RULE 4: labeled back fields
    job = extract_value_after_label(tokens, ("المهنه", "المهنة", "مهنة"))
    if job:
        fields["job"] = job
        rules_fired.append("LABEL: المهنة → القيمة المجاورة/التالية")

    religion = extract_value_after_label(
        tokens, ("الديانه", "الديانة", "ديانة"), value_hints=RELIGION_VALUES
    )
    if religion:
        fields["religion"] = religion
        rules_fired.append("LABEL/HINT: الديانة")

    marital = extract_value_after_label(
        tokens,
        ("الحاله الاجتماعيه", "الحالة الاجتماعية", "الحاله", "الحالة", "اجتماعية"),
        value_hints=MARITAL_VALUES,
    )
    if marital:
        fields["marital_status"] = marital
        rules_fired.append("LABEL/HINT: الحالة الاجتماعية")

    gender_ocr = extract_value_after_label(
        tokens, ("الجنس",), value_hints=GENDER_VALUES
    )
    if gender_ocr and not fields["gender"]:
        fields["gender"] = gender_ocr
        rules_fired.append("LABEL: الجنس من ظهر البطاقة")
    elif gender_ocr and fields["gender"]:
        # Prefer equation gender; note mismatch
        g_eq = normalize_ar(fields["gender"])
        g_ocr = normalize_ar(gender_ocr)
        if g_eq not in g_ocr and g_ocr not in g_eq:
            rules_fired.append("COND: تعارض جنس OCR مع معادلة الرقم — اعتُمدت المعادلة")

    husband = extract_value_after_label(tokens, ("اسم الزوج", "الزوج", "زوج"))
    if husband and "مهن" not in normalize_ar(husband):
        fields["husband_name"] = husband
        rules_fired.append("LABEL: اسم الزوج")

    # RULE 5: if gender from ID is أنثى and marital missing, try common tokens
    if not fields["marital_status"]:
        for t in tokens:
            for mv in MARITAL_VALUES:
                if normalize_ar(mv) == t.norm_text:
                    fields["marital_status"] = t.text.strip()
                    rules_fired.append("HINT: قيمة حالة اجتماعية مباشرة من OCR")
                    break

    if not fields["religion"]:
        for t in tokens:
            for rv in RELIGION_VALUES:
                if normalize_ar(rv) == t.norm_text:
                    fields["religion"] = t.text.strip()
                    rules_fired.append("HINT: قيمة ديانة مباشرة من OCR")
                    break

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
    ) -> list[OcrToken]:
        h, w = card.shape[:2]
        blocks_a = engine.ocr_bgr(card, enhance=False)
        tokens_a = blocks_to_tokens(blocks_a, width=w, height=h)

        enhanced = enhance_id_card(card)
        blocks_b = engine.ocr_bgr(enhanced, enhance=False)
        tokens_b = blocks_to_tokens(blocks_b, width=w, height=h)

        tokens = merge_token_lists(tokens_a, tokens_b)

        # Extra ROI OCR for bottom strip (national ID) — often missed
        y0, y1 = int(h * 0.70), h
        strip = card[y0:y1, :]
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
                        x0=b.x0 / max(sw, 1),
                        y0=0.70 + (b.y0 / max(sh, 1)) * 0.30,
                        x1=b.x1 / max(sw, 1),
                        y1=0.70 + (b.y1 / max(sh, 1)) * 0.30,
                    )
                )
            tokens = merge_token_lists(tokens, tokens_c)

        # Name zone ROI (left of photo) — Egyptian ID: 2 name lines
        tokens = merge_token_lists(
            tokens,
            self._zone_tokens(engine, card, 0.12, 0.50, 0.0, 0.70, index_base=2000),
        )
        # Address zone ROI — Egyptian ID: 2 address lines under name
        tokens = merge_token_lists(
            tokens,
            self._zone_tokens(engine, card, 0.45, 0.78, 0.0, 0.72, index_base=3000),
        )

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
        scale: float = 2.2,
    ) -> list[OcrToken]:
        h, w = card.shape[:2]
        y0, y1 = int(h * y0r), int(h * y1r)
        x0, x1 = int(w * x0r), int(w * x1r)
        roi = card[y0:y1, x0:x1]
        if not roi.size:
            return []
        big = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        big = enhance_id_card(big)
        blocks = engine.ocr_bgr(big, enhance=False)
        nh, nw = big.shape[:2]
        out: list[OcrToken] = []
        for i, b in enumerate(blocks):
            text = str(b.text or "").strip()
            if not text:
                continue
            out.append(
                OcrToken(
                    index=index_base + i,
                    text=text,
                    score=float(b.score or 0),
                    x0=x0r + (b.x0 / max(nw, 1)) * (x1r - x0r),
                    y0=y0r + (b.y0 / max(nh, 1)) * (y1r - y0r),
                    x1=x0r + (b.x1 / max(nw, 1)) * (x1r - x0r),
                    y1=y0r + (b.y1 / max(nh, 1)) * (y1r - y0r),
                )
            )
        return out

    def _zone_lines(
        self,
        engine: Any,
        card: np.ndarray,
        y0r: float,
        y1r: float,
        x0r: float,
        x1r: float,
    ) -> list[str]:
        toks = self._zone_tokens(engine, card, y0r, y1r, x0r, x1r, index_base=0)
        lines = _cluster_tokens_into_lines(toks, y_thresh=0.06)
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

        engine = ArabicOcrEngine.get()
        tokens = self._ocr_card_passes(
            engine, card, enhance_handwriting=enhance_handwriting
        )

        side = guess_side(tokens)
        fields, decoded, rules = apply_field_rules(tokens, side)

        # Prefer dedicated zone OCR (matches real card: 2 name lines + 2 address lines)
        if side != "back":
            name_zone = self._zone_lines(engine, card, 0.14, 0.48, 0.0, 0.70)
            addr_zone = self._zone_lines(engine, card, 0.48, 0.78, 0.0, 0.72)
            name_zone = [
                ln
                for ln in name_zone
                if not _is_header_noise(ln)
                and not _is_address_label(ln)
                and len(_digit_soup(ln)) < 8
            ]
            addr_zone = [
                _merge_line_scraps(_strip_address_label_prefix(ln))
                for ln in addr_zone
                if not (
                    _is_address_label(ln)
                    and len(_strip_address_label_prefix(ln)) < 3
                )
            ]
            addr_zone = [ln for ln in addr_zone if ln]

            if len(name_zone) >= 2:
                fields["full_name"] = " ".join(name_zone[:2])
                rules.append("ZONE2: الاسم من منطقتين (سطرين)")
            elif len(name_zone) == 1 and (
                not fields.get("full_name")
                or len(name_zone[0]) >= len(str(fields.get("full_name") or ""))
            ):
                # keep clustered full name if longer; else zone
                if not fields.get("full_name"):
                    fields["full_name"] = name_zone[0]
                    rules.append("ZONE2: سطر اسم واحد من المنطقة")

            if len(addr_zone) >= 2:
                fields["address"] = " ".join(addr_zone[:2])
                rules.append("ZONE2: العنوان سطرين من منطقة محل الإقامة")
            elif len(addr_zone) == 1 and not fields.get("address"):
                fields["address"] = addr_zone[0]
                rules.append("ZONE2: سطر عنوان من المنطقة")

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
        crop_pack = build_field_crops(card, tokens, fields, face)
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


def build_field_crops(
    card: np.ndarray,
    tokens: list[OcrToken],
    fields: dict[str, Any],
    face: np.ndarray | None,
) -> dict[str, Any]:
    """Build per-field source crops + annotated card (front or back)."""
    crop_keys = [
        "full_name",
        "address",
        "national_id",
        "job",
        "religion",
        "marital_status",
        "gender",
        "husband_name",
    ]
    regions: dict[str, list[OcrToken]] = {}
    crops: dict[str, str] = {}
    meta: dict[str, Any] = {}

    for key in crop_keys:
        val = fields.get(key)
        toks = tokens_for_value(tokens, val if isinstance(val, str) else None)
        if not toks:
            continue
        regions[key] = toks
        box = union_norm_bbox(toks)
        roi = crop_norm_region(card, box)
        if roi is not None and roi.size:
            crops[key] = _encode_image_b64(roi, quality=90)
            meta[key] = {
                "bbox_norm": [round(x, 4) for x in (box or (0, 0, 0, 0))],
                "token_texts": [t.text for t in toks],
            }

    annotated = annotate_field_regions(card, regions)
    # Draw face box if present
    faces = detect_faces(card)
    if faces:
        x, y, bw, bh = faces[0]
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 180, 220), 2)
        cv2.putText(
            annotated,
            "face",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 220),
            2,
            cv2.LINE_AA,
        )

    out: dict[str, Any] = {
        "annotated": _encode_image_b64(annotated, quality=88),
        "crops": crops,
        "crop_meta": meta,
    }
    if face is not None and face.size:
        out["face"] = _encode_image_b64(face, quality=90)
    return out
