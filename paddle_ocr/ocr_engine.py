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
class PageResult:
    page_index: int
    text: str
    lines: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


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

    def iter_pdf_pages(self, pdf_path: str):
        """Yield OCR results one PDF page at a time."""
        self._ensure_ready()
        assert self._ocr is not None

        import numpy as np
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        try:
            total = len(pdf)
            for index in range(total):
                page = pdf[index]
                try:
                    # ~150 DPI equivalent for readable Arabic text
                    bitmap = page.render(scale=2.0)
                    image = np.asarray(bitmap.to_pil().convert("RGB"))
                finally:
                    page.close()

                logger.info("OCR page %s/%s", index + 1, total)
                raw_results = self._ocr.predict(input=image)
                raw = (raw_results or [None])[0]
                lines, scores = self._extract_lines(raw)
                yield PageResult(
                    page_index=index,
                    text="\n".join(lines).strip(),
                    lines=lines,
                    scores=scores,
                )
        finally:
            pdf.close()

    def recognize_pdf(self, pdf_path: str) -> OcrDocumentResult:
        pages = list(self.iter_pdf_pages(pdf_path))
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
    def _extract_lines(cls, raw: Any) -> tuple[list[str], list[float]]:
        data = cls._as_dict(raw)
        texts = data.get("rec_texts") or data.get("texts") or []
        scores_raw = data.get("rec_scores") or data.get("scores") or []

        lines: list[str] = []
        for item in texts:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                lines.append(text)

        scores: list[float] = []
        try:
            for score in scores_raw:
                scores.append(float(score))
        except Exception:  # noqa: BLE001
            scores = []

        # Fallback for older-style nested results
        if not lines and "ocr_result" in data:
            for row in data["ocr_result"] or []:
                if isinstance(row, dict) and row.get("text"):
                    lines.append(str(row["text"]).strip())
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    lines.append(str(row[1][0] if isinstance(row[1], (list, tuple)) else row[1]).strip())

        return lines, scores
