"""FastAPI app: Arabic PDF OCR with PaddleOCR v3 UI."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

# Must be set before paddle/paddleocr import (via ocr_engine).
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from egyptian_id import EgyptianIdExtractor, ensure_yunet_model
from field_extractor import FieldExtractor
from mapping_store import apply_mapping, load_mapping, save_mapping
from ocr_engine import (
    PDF_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    ArabicOcrEngine,
    PageResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Arabic OCR — PaddleOCR v3", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

MAX_UPLOAD_MB = 40
field_extractor = FieldExtractor()
id_extractor = EgyptianIdExtractor()
IMAGE_ONLY = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _page_payload(page: PageResult) -> dict:
    return {
        "page_index": page.page_index,
        "page_number": page.page_index + 1,
        "text": page.text,
        "line_count": len(page.lines),
        "avg_score": (
            round(sum(page.scores) / len(page.scores), 4) if page.scores else None
        ),
    }


def _file_extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


def _validate_upload(file: UploadFile, content: bytes) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="لم يتم اختيار ملف.")
    ext = _file_extension(file.filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="يُقبل PDF أو صورة (PNG / JPG / WEBP / BMP / TIFF).",
        )
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"حجم الملف أكبر من {MAX_UPLOAD_MB} ميجابايت.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="الملف فارغ.")
    return ext


def _parse_bool_flag(raw: str | None) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _parse_field_labels(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    text = str(raw).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, str) and data.strip():
            return [data.strip()]
    except json.JSONDecodeError:
        pass
    parts = re.split(r"[\n,،;؛|]+", text)
    return [p.strip() for p in parts if p.strip()]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"max_upload_mb": MAX_UPLOAD_MB},
    )


@app.get("/map", response_class=HTMLResponse)
async def mapping_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "map.html", {})


@app.get("/id", response_class=HTMLResponse)
async def egyptian_id_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "id.html",
        {"max_upload_mb": MAX_UPLOAD_MB},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "engine": "PaddleOCR", "lang": "ar"}


@app.post("/api/id/ocr")
async def ocr_egyptian_id(
    file: UploadFile = File(...),
    enhance_handwriting: str = Form(default="0"),
    side: str = Form(default="auto"),
) -> JSONResponse:
    content = await file.read()
    if not file.filename:
        raise HTTPException(status_code=400, detail="لم يتم اختيار ملف.")
    ext = _file_extension(file.filename)
    if ext not in IMAGE_ONLY:
        raise HTTPException(
            status_code=400,
            detail="ارفع صورة البطاقة (PNG / JPG / WEBP / BMP / TIFF).",
        )
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"حجم الملف أكبر من {MAX_UPLOAD_MB} ميجابايت.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="الملف فارغ.")

    side_norm = (side or "auto").strip().lower()
    if side_norm not in {"auto", "front", "back"}:
        raise HTTPException(
            status_code=400,
            detail="قيمة side يجب أن تكون auto أو front أو back.",
        )

    handwriting = _parse_bool_flag(enhance_handwriting)
    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"id_{job_id}_", dir=UPLOAD_DIR))
    input_path = work_dir / f"card{ext}"

    try:
        ensure_yunet_model()
        input_path.write_bytes(content)
        logger.info(
            "Egyptian ID OCR: %s (%.2f MB) handwriting=%s side=%s",
            file.filename,
            size_mb,
            handwriting,
            side_norm,
        )
        result = id_extractor.process_image_path(
            input_path,
            enhance_handwriting=handwriting,
            forced_side=None if side_norm == "auto" else side_norm,
        )
        payload = result.to_dict()
        payload["ok"] = True
        payload["filename"] = file.filename
        payload["forced_side"] = side_norm
        payload["pipeline"] = "v2026-09-13b"
        # Surface crop method for UI verification
        crop_rule = next(
            (r for r in (payload.get("rules") or []) if str(r).startswith("CROP:")),
            None,
        )
        payload["crop_method"] = (
            str(crop_rule).split(":", 1)[-1].strip() if crop_rule else "unknown"
        )
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Egyptian ID OCR failed")
        raise HTTPException(
            status_code=500,
            detail=f"فشل قراءة البطاقة: {exc}",
        ) from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/api/mapping")
async def get_mapping() -> JSONResponse:
    return JSONResponse(load_mapping())


@app.post("/api/mapping")
async def post_mapping(request: Request) -> JSONResponse:
    body = await request.json()
    mapping = body.get("mapping")
    if mapping is None and isinstance(body, list):
        mapping = body
    if not isinstance(mapping, list):
        raise HTTPException(status_code=400, detail="mapping يجب أن يكون list.")
    saved = save_mapping(mapping, meta={"source": body.get("source") if isinstance(body, dict) else None})
    return JSONResponse({"ok": True, "mapping_count": len(saved.get("mapping") or []), **saved})


@app.post("/api/mapping/apply")
async def post_mapping_apply(request: Request) -> JSONResponse:
    body = await request.json()
    ocr_list = body.get("ocr_list") or []
    mapping = body.get("mapping")
    applied = apply_mapping(ocr_list, mapping=mapping)
    return JSONResponse({"ok": True, **applied})


@app.post("/api/ocr")
async def ocr_document(
    file: UploadFile = File(...),
    fields: str = Form(default=""),
    enhance_handwriting: str = Form(default="0"),
) -> JSONResponse:
    content = await file.read()
    ext = _validate_upload(file, content)
    labels = _parse_field_labels(fields)
    handwriting = _parse_bool_flag(enhance_handwriting)

    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"ocr_{job_id}_", dir=UPLOAD_DIR))
    input_path = work_dir / f"input{ext}"

    try:
        input_path.write_bytes(content)
        logger.info(
            "OCR start: %s (%.2f MB) fields=%s handwriting=%s",
            file.filename,
            len(content) / (1024 * 1024),
            labels,
            handwriting,
        )
        result = ArabicOcrEngine.get().recognize_document(
            str(input_path),
            enhance_handwriting=handwriting,
        )
        pages_payload = [_page_payload(p) for p in result.pages]
        extraction = field_extractor.extract_from_pages(
            [
                {
                    "page_number": p.page_index + 1,
                    "lines": p.lines,
                    "text": p.text,
                    "blocks": [b.to_dict() for b in p.blocks],
                }
                for p in result.pages
            ],
            labels=labels,
        )
        extraction_dict = extraction.to_dict()
        mapped = apply_mapping(extraction_dict.get("ocr_list") or [])
        return JSONResponse(
            {
                "ok": True,
                "filename": file.filename,
                "page_count": result.page_count,
                "line_count": result.line_count,
                "full_text": result.full_text,
                "pages": pages_payload,
                "requested_fields": labels,
                "extraction": extraction_dict,
                "mapped": mapped,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("OCR failed")
        raise HTTPException(
            status_code=500,
            detail=f"فشل استخراج النص: {exc}",
        ) from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/ocr/stream")
async def ocr_document_stream(
    file: UploadFile = File(...),
    fields: str = Form(default=""),
    enhance_handwriting: str = Form(default="0"),
) -> StreamingResponse:
    """Stream OCR page-by-page; extract values for user-entered field labels."""
    content = await file.read()
    ext = _validate_upload(file, content)
    labels = _parse_field_labels(fields)
    handwriting = _parse_bool_flag(enhance_handwriting)

    filename = file.filename or f"document{ext}"
    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"ocr_{job_id}_", dir=UPLOAD_DIR))
    input_path = work_dir / f"input{ext}"
    input_path.write_bytes(content)
    size_mb = len(content) / (1024 * 1024)
    logger.info(
        "OCR stream start: %s (%.2f MB) fields=%s handwriting=%s",
        filename,
        size_mb,
        labels,
        handwriting,
    )

    def event_stream():
        engine = ArabicOcrEngine.get()
        pages_done = 0
        line_count = 0
        accumulated_pages: list[dict] = []
        try:
            total = engine.count_document_pages(str(input_path))
            yield json.dumps(
                {
                    "type": "start",
                    "filename": filename,
                    "page_count": total,
                    "requested_fields": labels,
                    "enhance_handwriting": handwriting,
                    "kind": "pdf" if ext in PDF_EXTENSIONS else "image",
                },
                ensure_ascii=False,
            ) + "\n"

            for page in engine.iter_document(
                str(input_path),
                enhance_handwriting=handwriting,
            ):
                pages_done += 1
                line_count += len(page.lines)
                accumulated_pages.append(
                    {
                        "page_number": page.page_index + 1,
                        "lines": page.lines,
                        "text": page.text,
                        "blocks": [b.to_dict() for b in page.blocks],
                    }
                )
                extraction = field_extractor.extract_from_pages(
                    accumulated_pages, labels=labels
                )
                extraction_dict = extraction.to_dict()
                mapped = apply_mapping(extraction_dict.get("ocr_list") or [])
                yield json.dumps(
                    {
                        "type": "page",
                        "filename": filename,
                        "page_count": total,
                        "pages_done": pages_done,
                        "line_count": line_count,
                        "page": _page_payload(page),
                        "extraction": extraction_dict,
                        "mapped": mapped,
                    },
                    ensure_ascii=False,
                ) + "\n"

            final_extraction = field_extractor.extract_from_pages(
                accumulated_pages, labels=labels
            )
            final_dict = final_extraction.to_dict()
            final_mapped = apply_mapping(final_dict.get("ocr_list") or [])
            yield json.dumps(
                {
                    "type": "done",
                    "filename": filename,
                    "page_count": total,
                    "pages_done": pages_done,
                    "line_count": line_count,
                    "extraction": final_dict,
                    "mapped": final_mapped,
                },
                ensure_ascii=False,
            ) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("OCR stream failed")
            yield json.dumps(
                {"type": "error", "detail": f"فشل استخراج النص: {exc}"},
                ensure_ascii=False,
            ) + "\n"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
