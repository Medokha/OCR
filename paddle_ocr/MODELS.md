# Models Used in This OCR System

This project runs **fully local / open-source** (no cloud OCR APIs).  
It combines several specialized models: text OCR, document crop, and face crop — then applies Egyptian ID rules on top.

---

## Overview

| Stage | Model | Role |
|---|---|---|
| **Primary Arabic OCR** | PaddleOCR PP-OCRv5 (det + Arabic rec) | Detect and read Arabic/Latin text & digits |
| **Secondary Arabic OCR** | OnnxTR (FAST det + PARSeq rec) | Ensemble / fallback for hard regions |
| **ID card crop** | DeepLabV3 ONNX (`autocrop_model_v2`) | Find and warp the card from a phone photo |
| **Face crop** | OpenCV YuNet | Crop the portrait on the ID front |
| **Fallback geometry** | OpenCV (contours / deskew) | If DeepLab crop fails |
| **Field logic** | Rules + national-ID equations | Map OCR text → name, NID, DOB, job, expiry… |

---

## 1. Primary OCR — PaddleOCR PP-OCRv5 (Arabic)

**Stack:** [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) + [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 3.x  

**Local weights** (under `paddle_ocr/models/`):

| Model | Path | Job |
|---|---|---|
| **PP-OCRv5_server_det** | `models/PP-OCRv5_server_det` | Text **detection** (finds word/line boxes) |
| **arabic_PP-OCRv5_mobile_rec** | `models/arabic_PP-OCRv5_mobile_rec` | Text **recognition** (reads Arabic script + Indic digits) |

**Why this pair**
- Server detector is stronger on dense ID layouts and faint print.
- Arabic mobile recognizer is tuned for Arabic letters and Eastern Arabic numerals (`٠١٢…٩`) used on Egyptian IDs.
- Used as the **main engine** for PDF OCR and Egyptian ID front/back reading.

**Where it runs in code:** `ocr_engine.py` → `ArabicOcrEngine`

---

## 2. Secondary OCR — OnnxTR (Arabic)

**Stack:** [OnnxTR](https://github.com/mindee/onnxtr) + ONNX Runtime  

**Local weights:**

| Model | Path | Job |
|---|---|---|
| **onnxtr-fast_base-arabic** | `models/onnxtr-fast_base-arabic` | Text detection (FAST) |
| **onnxtr-parseq-arabic** | `models/onnxtr-parseq-arabic` | Text recognition (PARSeq) |

Sources (community fine-tunes for Arabic documents):
- `madskills/onnxtr-fast_base-arabic`
- `madskills/onnxtr-parseq-arabic`

**Role**
- Secondary ensemble — especially helpful on back-side bands and noisy crops.
- Kept **separate** from Paddle tokens so noisy words do not overwrite strong primary results.
- Optional: skip with env `OCR_SKIP_ONNXTR=1`.

**Where it runs:** `arabic_onnxtr.py` → `ArabicOnnxTrEngine`

---

## 3. Card crop — DeepLabV3 Autocrop

**Stack:** [autocrop-kh](https://pypi.org/project/autocrop-kh/) + ONNX Runtime  

**Local weight:**

| Model | Path | Job |
|---|---|---|
| **autocrop_model_v2.onnx** | `models/autocrop/autocrop_model_v2.onnx` | Segment ID/passport/document and perspective-correct the card |

Based on **DeepLabV3**-style document segmentation (metythorn/autocrop lineage).

**Role**
- Primary way to crop the physical card from cluttered phone photos.
- Fallback: OpenCV contour quad + warp if the ONNX model is missing or fails.

**Where it runs:** `card_crop.py` → `CardCropEngine` / `crop_id_card()`

---

## 4. Face crop — OpenCV YuNet

**Stack:** OpenCV DNN  

**Local weight:**

| Model | Path | Job |
|---|---|---|
| **face_detection_yunet_2023mar.onnx** | `models/face_detection_yunet_2023mar.onnx` | Detect the portrait face on the ID front |

Downloaded from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (YuNet 2023).

**Role**
- Crops the personal photo shown in the ID UI.
- Layout zones for birth date still prefer fixed card geometry (YuNet box alone is often chin-only).

**Where it runs:** `egyptian_id.py` → `ensure_yunet_model()` / `crop_face()`

---

## 5. What is *not* a neural model (but critical)

After OCR, Egyptian ID fields are filled with **deterministic logic**:

- **Layout zones** (normalized boxes for name, address, NID, DOB, job, status, expiry)
- **National ID equations** (14 digits → birth date, gender, governorate + checksum)
- **Back-side parsers** (profession, religion, marital status, expiry “سارية حتى”)
- **OpenCV enhance / deskew** for low-contrast scans

This is why DOB can still fill correctly even when the printed date under the eagle is too faint to OCR.

---

## Pipeline (Egyptian ID)

```text
Photo
  → DeepLabV3 card crop (or contour fallback)
  → deskew / mild enhance
  → PaddleOCR (primary)  +  OnnxTR (secondary, optional)
  → YuNet face crop (front)
  → Zone OCR + rules + NID equations
  → Structured fields + field preview crops
```

---

## Runtime notes

- All inference is intended for **CPU** by default.
- First run may download missing weights (`download_models.py` / YuNet auto-download).
- Windows tip: Paddle may need PIR/MKLDNN flags disabled (handled in `ocr_engine.py`).
- Suggested app ports: PDF UI `/`, Egyptian ID UI `/id` (often `8001` if `8000` is blocked).

---

## Dependencies (from `requirements.txt`)

```text
paddlepaddle, paddleocr     → primary OCR
onnxruntime, onnxtr         → secondary OCR
autocrop-kh                 → DeepLab card crop
opencv-python (via stack)   → YuNet, geometry, enhance
fastapi + uvicorn           → web API / UI
```

---

## Summary for stakeholders

This system is an **open-source, on-premise Arabic OCR stack**:

1. **PaddleOCR PP-OCRv5 Arabic** — main reader  
2. **OnnxTR Arabic** — secondary reader  
3. **DeepLabV3 autocrop** — card localization  
4. **YuNet** — face crop  
5. **Domain rules** — Egyptian national ID field mapping  

No third-party cloud vision API is required for OCR.
