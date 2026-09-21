"""Local handwritten Arabic/English OCR via sherif1313/Arabic-English-handwritten-OCR-v3.

Open-source (Apache-2.0) Qwen2.5-VL-3B fine-tune for handwriting.
https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3

Lazy-loaded singleton — only loads when handwriting mode is requested.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LOCAL_DIR = MODELS_DIR / "Arabic-English-handwritten-OCR-v3"
HF_ID = "sherif1313/Arabic-English-handwritten-OCR-v3"

DEFAULT_PROMPT = (
    "استخرج النص المكتوب باليد من هذه الصورة فقط. "
    "أرجع النص كما هو بدون شرح أو تعليقات. "
    "لو مفيش نص واضح اكتب فراغ:"
)

FIELD_PROMPT = (
    "هذه قصّة حقل من استمارة. اسم الحقل: {label}. "
    "اقرأ فقط القيمة المكتوبة بخط اليد داخل الصورة. "
    "أرجع القيمة فقط بدون اسم الحقل وبدون شرح. "
    "لو الخانة فاضية اكتب:"
)


class HandwrittenOcrEngine:
    """Lazy singleton around Arabic-English-handwritten-OCR-v3 (Qwen2.5-VL)."""

    _instance: HandwrittenOcrEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: str | None = None
        self._init_error: str | None = None

    @classmethod
    def get(cls) -> HandwrittenOcrEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = HandwrittenOcrEngine()
        return cls._instance

    @classmethod
    def model_source(cls) -> str:
        if (LOCAL_DIR / "config.json").exists():
            return str(LOCAL_DIR)
        return os.environ.get("HANDWRITTEN_OCR_MODEL", HF_ID)

    @classmethod
    def available(cls) -> bool:
        """True if local weights exist OR HF download is allowed."""
        if (LOCAL_DIR / "config.json").exists():
            return True
        return os.environ.get("OCR_ALLOW_HF_DOWNLOAD", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }

    def _ensure_ready(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if self._init_error:
            raise RuntimeError(self._init_error)
        # Corporate SSL inspection (same as download scripts)
        os.environ.setdefault("CURL_CA_BUNDLE", "")
        os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
        try:
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # noqa: BLE001
            self._init_error = (
                "يحتاج الموديل: torch + transformers (Qwen2.5-VL). "
                f"التفاصيل: {exc}"
            )
            raise RuntimeError(self._init_error) from exc

        source = self.model_source()
        logger.info("Loading handwritten OCR model from %s …", source)
        try:
            use_cuda = torch.cuda.is_available()
            # float16 on CPU cuts RAM vs float32 (~7GB weights vs ~14GB)
            dtype = torch.bfloat16 if use_cuda else torch.float16
            local_only = Path(source).exists()

            logger.info("Loading processor…")
            self._processor = AutoProcessor.from_pretrained(
                source,
                trust_remote_code=True,
                local_files_only=local_only,
            )

            kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "local_files_only": local_only,
            }
            logger.info("Loading weights on %s (%s)…", "cuda" if use_cuda else "cpu", dtype)
            if use_cuda:
                kwargs["device_map"] = "auto"
            else:
                # Keep peak RAM lower: allow accelerate to spill layers to disk if needed
                offload_dir = MODELS_DIR / "_hw_offload"
                offload_dir.mkdir(parents=True, exist_ok=True)
                kwargs["device_map"] = "auto"
                kwargs["max_memory"] = {"cpu": "7GiB"}
                kwargs["offload_folder"] = str(offload_dir)
                kwargs["offload_state_dict"] = True
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source, **kwargs
            )
            self._model.eval()
            self._device = "cuda" if use_cuda else "cpu"
            logger.info("Handwritten OCR ready on %s (%s)", self._device, dtype)
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._processor = None
            self._init_error = f"فشل تحميل Arabic-English-handwritten-OCR-v3: {exc}"
            logger.exception(self._init_error)
            raise RuntimeError(self._init_error) from exc

    @staticmethod
    def _bgr_to_pil(image_bgr: np.ndarray):
        from PIL import Image

        if image_bgr is None or not getattr(image_bgr, "size", 0):
            raise ValueError("empty image")
        rgb = cv2.cvtColor(np.asarray(image_bgr), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # Upscale tiny field crops (common for form cells)
        if max(img.size) < 800:
            img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.Resampling.LANCZOS)
        # Cap huge pages for speed/memory
        mx = max(img.size)
        if mx > 1200:
            s = 1200 / mx
            img = img.resize(
                (max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))),
                Image.Resampling.LANCZOS,
            )
        return img

    def recognize_bgr(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Read handwriting from an OpenCV BGR crop/page."""
        self._ensure_ready()
        assert self._model is not None and self._processor is not None

        import torch

        pil = self._bgr_to_pil(image_bgr)
        text_prompt = (prompt or DEFAULT_PROMPT).strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[pil],
            padding=True,
            return_tensors="pt",
        )
        # Move tensors to model device
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self._processor.tokenizer.eos_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        out = self._processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return (out or "").strip()

    def recognize_field_bgr(
        self,
        image_bgr: np.ndarray,
        *,
        label: str = "",
    ) -> str:
        """Read a single form-field crop with a field-aware prompt."""
        prompt = FIELD_PROMPT.format(label=label or "القيمة")
        raw = self.recognize_bgr(image_bgr, prompt=prompt, max_new_tokens=128)
        # Strip common model chatter
        cleaned = raw.strip().strip("\"'`")
        for junk in (
            "لا يوجد نص",
            "لا يوجد",
            "فارغ",
            "empty",
            "none",
            "n/a",
        ):
            if cleaned.casefold() == junk or cleaned == ":" or cleaned == "：":
                return ""
        return cleaned
