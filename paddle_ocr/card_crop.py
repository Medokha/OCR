"""Document/ID card cropping via open-source DeepLabV3 ONNX (autocrop_kh).

Model: metythorn/autocrop — autocrop_model_v2.onnx
Trained for ID cards / passports / documents; much more robust than Canny edges
on phone photos with cluttered backgrounds.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "autocrop" / "autocrop_model_v2.onnx"


class CardCropEngine:
    """Lazy singleton around autocrop_kh DeepLabV3 ONNX session."""

    _instance: CardCropEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._session: Any = None
        self._init_error: str | None = None

    @classmethod
    def get(cls) -> CardCropEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CardCropEngine()
        return cls._instance

    @classmethod
    def available(cls) -> bool:
        return MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000

    def _ensure_ready(self) -> None:
        if self._session is not None:
            return
        if self._init_error:
            raise RuntimeError(self._init_error)
        if not self.available():
            self._init_error = f"Autocrop model missing: {MODEL_PATH}"
            raise RuntimeError(self._init_error)
        try:
            from autocrop_kh import load_autocrop_model

            self._session = load_autocrop_model(str(MODEL_PATH), device="cpu")
            logger.info("DeepLabV3 autocrop model ready (%s)", MODEL_PATH.name)
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"Failed to load autocrop model: {exc}"
            logger.exception(self._init_error)
            raise RuntimeError(self._init_error) from exc

    def crop_bgr(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Return perspective-corrected card crop (BGR), or None on failure."""
        if image_bgr is None or not getattr(image_bgr, "size", 0):
            return None
        try:
            self._ensure_ready()
            from autocrop_kh import extract

            # autocrop_kh expects RGB
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            cropped = extract(image_true=rgb, trained_model=self._session, image_size=384)
            if cropped is None:
                return None
            out = np.asarray(cropped)
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            if out.ndim != 3 or out.shape[2] != 3:
                return None
            h, w = out.shape[:2]
            if h < 80 or w < 120:
                return None
            # Enforce landscape ID-1 orientation
            if h > w * 1.05:
                out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001
            logger.debug("DeepLabV3 card crop failed", exc_info=True)
            return None


def crop_id_card(
    image_bgr: np.ndarray,
    *,
    fallback_quad_fn: Any = None,
    warp_fn: Any = None,
) -> tuple[np.ndarray, str]:
    """Best-effort card crop: DeepLabV3 first, then classic quad fallback.

    Returns (card_bgr, method) where method is one of:
      deeplab / contour / raw
    """
    if CardCropEngine.available():
        ml = CardCropEngine.get().crop_bgr(image_bgr)
        if ml is not None and ml.size:
            return ml, "deeplab"

    if fallback_quad_fn is not None and warp_fn is not None:
        try:
            quad = fallback_quad_fn(image_bgr)
            if quad is not None:
                warped = warp_fn(image_bgr, quad)
                if warped is not None and warped.size:
                    return warped, "contour"
        except Exception:  # noqa: BLE001
            logger.debug("contour card crop failed", exc_info=True)

    return image_bgr, "raw"
