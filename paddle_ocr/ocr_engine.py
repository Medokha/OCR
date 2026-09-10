"""Arabic OCR engine powered by PaddleOCR 3.x (PP-OCRv5 Arabic)."""

from __future__ import annotations

import logging
import os
import ssl
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS


def _relax_ssl_if_needed() -> None:
    """Corporate proxies often break cert verification for model hosts."""
    if os.environ.get("OCR_STRICT_SSL", "").lower() in {"1", "true", "yes"}:
        return
    try:
        ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        import requests

        _orig = requests.Session.request

        def _request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.setdefault("verify", False)
            return _orig(self, method, url, **kwargs)

        requests.Session.request = _request  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001
        logger.debug("Could not relax SSL verification", exc_info=True)


def _prepare_paddle_runtime() -> None:
    """Avoid known CPU/oneDNN + PIR crashes on Windows PaddlePaddle 3.3.x."""
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    _relax_ssl_if_needed()


@dataclass
class TextBlock:
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
    def width(self) -> float:
        return max(self.x1 - self.x0, 1.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "bbox": [self.x0, self.y0, self.x1, self.y1],
        }


@dataclass
class PageResult:
    page_index: int
    text: str
    lines: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class OcrDocumentResult:
    pages: list[PageResult]
    full_text: str
    page_count: int
    line_count: int


class ArabicOcrEngine:
    """Lazy-loaded singleton wrapper around PaddleOCR Arabic pipeline."""

    _instance: ArabicOcrEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._ocr = None
        self._init_error: str | None = None

    @classmethod
    def get(cls) -> ArabicOcrEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ArabicOcrEngine()
        return cls._instance

    def _ensure_ready(self) -> None:
        if self._ocr is not None:
            return
        if self._init_error:
            raise RuntimeError(self._init_error)

        try:
            _prepare_paddle_runtime()
            from paddleocr import PaddleOCR

            det_dir = MODELS_DIR / "PP-OCRv5_server_det"
            rec_dir = MODELS_DIR / "arabic_PP-OCRv5_mobile_rec"
            kwargs: dict[str, Any] = {
                "lang": "ar",
                "ocr_version": "PP-OCRv5",
                "device": "cpu",
                "enable_mkldnn": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            if det_dir.exists() and rec_dir.exists():
                kwargs["text_detection_model_dir"] = str(det_dir)
                kwargs["text_recognition_model_dir"] = str(rec_dir)
                kwargs["text_detection_model_name"] = "PP-OCRv5_server_det"
                kwargs["text_recognition_model_name"] = "arabic_PP-OCRv5_mobile_rec"
                logger.info("Loading local Arabic models from %s", MODELS_DIR)
            else:
                logger.info(
                    "Local models missing; PaddleOCR will download weights on first run."
                )

            self._ocr = PaddleOCR(**kwargs)
            logger.info("PaddleOCR Arabic model ready.")
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"Failed to initialize PaddleOCR: {exc}"
            logger.exception(self._init_error)
            raise RuntimeError(self._init_error) from exc

    def count_pdf_pages(self, pdf_path: str) -> int:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        try:
            return len(pdf)
        finally:
            pdf.close()

    def count_document_pages(self, path: str) -> int:
        ext = Path(path).suffix.lower()
        if ext in PDF_EXTENSIONS:
            return self.count_pdf_pages(path)
        if ext in IMAGE_EXTENSIONS:
            return 1
        raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def _load_image_rgb(image_path: str) -> Any:
        import cv2
        import numpy as np

        data = np.fromfile(image_path, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            # Fallback for odd encodings / paths
            from PIL import Image

            return np.asarray(Image.open(image_path).convert("RGB"))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _enhance_handwriting(image: Any) -> Any:
        """Boost contrast/edges for handwritten ink on forms/scans."""
        import cv2
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim == 2:
            gray = arr
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Downscale very large pages before heavy filters (avoids OOM / process kill).
        h, w = gray.shape[:2]
        max_side = 2200
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # Fast denoise (NlMeans is too heavy on big scans and can crash the server).
        den = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)

        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        contrast = clahe.apply(den)

        blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        sharp = cv2.addWeighted(contrast, 1.4, blur, -0.4, 0)

        binary = cv2.adaptiveThreshold(
            sharp,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        mixed = cv2.addWeighted(sharp, 0.55, binary, 0.45, 0)
        return cv2.cvtColor(mixed, cv2.COLOR_GRAY2RGB)

    def _ocr_numpy(self, image: Any) -> list[TextBlock]:
        assert self._ocr is not None
        raw_results = self._ocr.predict(input=image)
        raw = (raw_results or [None])[0]
        return self._extract_blocks(raw)

    def ocr_bgr(self, image_bgr: Any, *, enhance: bool = False) -> list[TextBlock]:
        """Run OCR on an OpenCV BGR image array."""
        self._ensure_ready()
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(np.asarray(image_bgr), cv2.COLOR_BGR2RGB)
        if enhance:
            rgb = self._enhance_handwriting(rgb)
        return self._ocr_numpy(rgb)

    def iter_pdf_pages(
        self,
        pdf_path: str,
        *,
        enhance_handwriting: bool = False,
        render_scale: float | None = None,
    ):
        """Yield OCR results one PDF page at a time."""
        self._ensure_ready()
        assert self._ocr is not None

        import numpy as np
        import pypdfium2 as pdfium

        # Handwriting: modest DPI bump + single enhanced pass (dual pass was crashing).
        if render_scale is None:
            render_scale = 2.4 if enhance_handwriting else 2.0

        pdf = pdfium.PdfDocument(pdf_path)
        try:
            total = len(pdf)
            for index in range(total):
                page = pdf[index]
                try:
                    bitmap = page.render(scale=float(render_scale))
                    image = np.asarray(bitmap.to_pil().convert("RGB"))
                finally:
                    page.close()

                logger.info(
                    "OCR page %s/%s (scale=%.1f handwriting=%s)",
                    index + 1,
                    total,
                    render_scale,
                    enhance_handwriting,
                )

                if enhance_handwriting:
                    image = self._enhance_handwriting(image)

                blocks = self._ocr_numpy(image)
                lines = [b.text for b in blocks]
                scores = [b.score for b in blocks]
                yield PageResult(
                    page_index=index,
                    text="\n".join(lines).strip(),
                    lines=lines,
                    scores=scores,
                    blocks=blocks,
                )
        finally:
            pdf.close()

    def iter_image(
        self,
        image_path: str,
        *,
        enhance_handwriting: bool = False,
    ):
        """Yield a single-page OCR result from an image file."""
        self._ensure_ready()
        image = self._load_image_rgb(image_path)
        logger.info(
            "OCR image %s (handwriting=%s)",
            Path(image_path).name,
            enhance_handwriting,
        )
        if enhance_handwriting:
            image = self._enhance_handwriting(image)
        blocks = self._ocr_numpy(image)
        lines = [b.text for b in blocks]
        scores = [b.score for b in blocks]
        yield PageResult(
            page_index=0,
            text="\n".join(lines).strip(),
            lines=lines,
            scores=scores,
            blocks=blocks,
        )

    def iter_document(
        self,
        path: str,
        *,
        enhance_handwriting: bool = False,
        render_scale: float | None = None,
    ):
        ext = Path(path).suffix.lower()
        if ext in PDF_EXTENSIONS:
            yield from self.iter_pdf_pages(
                path,
                enhance_handwriting=enhance_handwriting,
                render_scale=render_scale,
            )
            return
        if ext in IMAGE_EXTENSIONS:
            yield from self.iter_image(
                path,
                enhance_handwriting=enhance_handwriting,
            )
            return
        raise ValueError(f"Unsupported file type: {ext}")

    def recognize_pdf(
        self,
        pdf_path: str,
        *,
        enhance_handwriting: bool = False,
        render_scale: float | None = None,
    ) -> OcrDocumentResult:
        return self.recognize_document(
            pdf_path,
            enhance_handwriting=enhance_handwriting,
            render_scale=render_scale,
        )

    def recognize_document(
        self,
        path: str,
        *,
        enhance_handwriting: bool = False,
        render_scale: float | None = None,
    ) -> OcrDocumentResult:
        pages = list(
            self.iter_document(
                path,
                enhance_handwriting=enhance_handwriting,
                render_scale=render_scale,
            )
        )
        full_parts = []
        for page in pages:
            if not page.text:
                continue
            full_parts.append(f"—— الصفحة {page.page_index + 1} ——\n{page.text}")

        full_text = "\n\n".join(full_parts).strip()
        line_count = sum(len(p.lines) for p in pages)
        return OcrDocumentResult(
            pages=pages,
            full_text=full_text,
            page_count=len(pages),
            line_count=line_count,
        )

    @staticmethod
    def _as_dict(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            data = raw
        elif hasattr(raw, "json"):
            data = raw.json
            if callable(data):
                data = data()
        elif hasattr(raw, "res"):
            data = {"res": raw.res}
        else:
            try:
                data = dict(raw)
            except Exception:  # noqa: BLE001
                return {}

        if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
            return data["res"]
        return data if isinstance(data, dict) else {}

    @classmethod
    def _extract_page_index(cls, raw: Any, fallback: int) -> int:
        data = cls._as_dict(raw)
        value = data.get("page_index")
        if value is None:
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @classmethod
    def _extract_blocks(cls, raw: Any) -> list[TextBlock]:
        data = cls._as_dict(raw)
        texts = cls._first_seq(data.get("rec_texts"), data.get("texts"))
        scores_raw = cls._first_seq(data.get("rec_scores"), data.get("scores"))
        boxes = cls._first_seq(
            data.get("rec_boxes"),
            data.get("dt_polys"),
            data.get("rec_polys"),
            data.get("boxes"),
        )

        blocks: list[TextBlock] = []
        for i, item in enumerate(texts):
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            try:
                score = float(scores_raw[i]) if i < len(scores_raw) else 0.0
            except Exception:  # noqa: BLE001
                score = 0.0
            box = boxes[i] if i < len(boxes) else None
            x0, y0, x1, y1 = cls._box_to_xyxy(box, i)
            blocks.append(
                TextBlock(text=text, score=score, x0=x0, y0=y0, x1=x1, y1=y1)
            )

        if len(blocks) == 0 and "ocr_result" in data:
            for i, row in enumerate(data.get("ocr_result") or []):
                if isinstance(row, dict) and row.get("text"):
                    text = str(row["text"]).strip()
                    box = row.get("box")
                    if box is None:
                        box = row.get("bbox")
                    x0, y0, x1, y1 = cls._box_to_xyxy(box, i)
                    try:
                        score = float(row.get("score") or 0.0)
                    except Exception:  # noqa: BLE001
                        score = 0.0
                    blocks.append(
                        TextBlock(
                            text=text,
                            score=score,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                        )
                    )
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    box = row[0]
                    rec = row[1]
                    text = str(rec[0] if isinstance(rec, (list, tuple)) else rec).strip()
                    score = (
                        float(rec[1])
                        if isinstance(rec, (list, tuple)) and len(rec) > 1
                        else 0.0
                    )
                    x0, y0, x1, y1 = cls._box_to_xyxy(box, i)
                    if text:
                        blocks.append(
                            TextBlock(
                                text=text, score=score, x0=x0, y0=y0, x1=x1, y1=y1
                            )
                        )
        return blocks

    @staticmethod
    def _first_seq(*candidates: Any) -> list[Any]:
        """Pick first non-None sequence without bool()-testing numpy arrays."""
        for value in candidates:
            if value is None:
                continue
            try:
                return list(value)
            except TypeError:
                continue
        return []

    @staticmethod
    def _box_to_xyxy(box: Any, fallback_index: int) -> tuple[float, float, float, float]:
        if box is None:
            y = float(fallback_index) * 40.0
            return 0.0, y, 400.0, y + 30.0
        try:
            import numpy as np

            arr = np.asarray(box, dtype=float)
            if arr.size == 0:
                y = float(fallback_index) * 40.0
                return 0.0, y, 400.0, y + 30.0
            if arr.ndim == 1 and arr.size >= 4:
                return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            flat = arr.reshape(-1, 2)
            xs = flat[:, 0]
            ys = flat[:, 1]
            return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        except Exception:  # noqa: BLE001
            y = float(fallback_index) * 40.0
            return 0.0, y, 400.0, y + 30.0

    @classmethod
    def _extract_lines(cls, raw: Any) -> tuple[list[str], list[float]]:
        blocks = cls._extract_blocks(raw)
        return [b.text for b in blocks], [b.score for b in blocks]
