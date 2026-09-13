"""Secondary Arabic OCR via OnnxTR (FAST det + PARSeq rec), fine-tuned for Arabic docs.

Open-source models (Apache-friendly ONNX Runtime stack):
  - madskills/onnxtr-fast_base-arabic
  - madskills/onnxtr-parseq-arabic

Used as an ensemble with PaddleOCR — never as a hard replacement.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DET_DIR = MODELS_DIR / "onnxtr-fast_base-arabic"
REC_DIR = MODELS_DIR / "onnxtr-parseq-arabic"


class ArabicOnnxTrEngine:
    """Lazy singleton OnnxTR predictor for Arabic print."""

    _instance: ArabicOnnxTrEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._predictor = None
        self._init_error: str | None = None

    @classmethod
    def get(cls) -> ArabicOnnxTrEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ArabicOnnxTrEngine()
        return cls._instance

    @classmethod
    def available(cls) -> bool:
        return DET_DIR.exists() and REC_DIR.exists() and (DET_DIR / "model.onnx").exists()

    def _load_arch(self, folder: Path) -> Any:
        from onnxtr.models import detection, recognition

        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        model_path = str(folder / "model.onnx")
        arch = cfg.pop("arch")
        task = cfg.pop("task")
        for key, value in list(cfg.items()):
            if isinstance(value, list):
                cfg[key] = tuple(value)
        if task == "detection":
            model = detection.__dict__[arch](model_path)
        elif task == "recognition":
            model = recognition.__dict__[arch](
                model_path, input_shape=cfg["input_shape"], vocab=cfg["vocab"]
            )
        else:
            raise RuntimeError(f"Unsupported OnnxTR task: {task}")
        model.cfg = cfg
        return model

    def _ensure_ready(self) -> None:
        if self._predictor is not None:
            return
        if self._init_error:
            raise RuntimeError(self._init_error)
        if not self.available():
            self._init_error = (
                "OnnxTR Arabic models missing under models/onnxtr-*-arabic"
            )
            raise RuntimeError(self._init_error)
        try:
            from onnxtr.models import ocr_predictor

            det = self._load_arch(DET_DIR)
            reco = self._load_arch(REC_DIR)
            self._predictor = ocr_predictor(det_arch=det, reco_arch=reco)
            logger.info("OnnxTR Arabic models ready.")
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"Failed to init OnnxTR: {exc}"
            logger.exception(self._init_error)
            raise RuntimeError(self._init_error) from exc

    def ocr_bgr(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Return word-level boxes as dicts: text, score, x0,y0,x1,y1 (pixel coords)."""
        import cv2

        self._ensure_ready()
        if image_bgr is None or not getattr(image_bgr, "size", 0):
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # onnxtr expects file path or numpy via DocumentFile
        from onnxtr.io import DocumentFile

        # Write is avoided: DocumentFile accepts ndarray list in recent versions
        try:
            doc = DocumentFile.from_images([rgb])
        except Exception:
            # Fallback: temp encode
            ok, buf = cv2.imencode(".png", image_bgr)
            if not ok:
                return []
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(buf.tobytes())
                path = tmp.name
            try:
                doc = DocumentFile.from_images([path])
            finally:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

        result = self._predictor(doc)
        h, w = image_bgr.shape[:2]
        out: list[dict[str, Any]] = []
        for page in getattr(result, "pages", []) or []:
            for block in getattr(page, "blocks", []) or []:
                for line in getattr(block, "lines", []) or []:
                    words = getattr(line, "words", None) or []
                    if words:
                        for word in words:
                            value = str(getattr(word, "value", "") or "").strip()
                            if not value:
                                continue
                            geom = getattr(word, "geometry", None)
                            conf = float(getattr(word, "confidence", 0.0) or 0.0)
                            x0, y0, x1, y1 = _geom_to_xyxy(geom, w, h)
                            out.append(
                                {
                                    "text": value,
                                    "score": conf,
                                    "x0": x0,
                                    "y0": y0,
                                    "x1": x1,
                                    "y1": y1,
                                }
                            )
                    else:
                        value = str(getattr(line, "value", "") or "").strip()
                        if not value:
                            # join words already handled; skip empty
                            continue
                        geom = getattr(line, "geometry", None)
                        conf = float(getattr(line, "confidence", 0.0) or 0.0)
                        x0, y0, x1, y1 = _geom_to_xyxy(geom, w, h)
                        out.append(
                            {
                                "text": value,
                                "score": conf,
                                "x0": x0,
                                "y0": y0,
                                "x1": x1,
                                "y1": y1,
                            }
                        )
        return out


def _geom_to_xyxy(geom: Any, width: int, height: int) -> tuple[float, float, float, float]:
    """OnnxTR geometry is usually ((x0,y0),(x1,y1)) normalized 0..1."""
    if geom is None:
        return 0.0, 0.0, float(width), float(height)
    try:
        (x0, y0), (x1, y1) = geom
        # If already absolute pixels (>1), keep; else scale
        if max(float(x0), float(y0), float(x1), float(y1)) <= 1.5:
            return float(x0) * width, float(y0) * height, float(x1) * width, float(y1) * height
        return float(x0), float(y0), float(x1), float(y1)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, float(width), float(height)
