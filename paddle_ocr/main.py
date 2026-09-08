"""FastAPI app: Arabic PDF OCR with PaddleOCR v3 UI."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

# Must be set before paddle/paddleocr import (via ocr_engine).
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ocr_engine import ArabicOcrEngine, PageResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Arabic OCR — PaddleOCR v3", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

MAX_UPLOAD_MB = 40


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


def _validate_pdf_upload(file: UploadFile, content: bytes) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="لم يتم اختيار ملف.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="يُقبل ملف PDF فقط.")
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"حجم الملف أكبر من {MAX_UPLOAD_MB} ميجابايت.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="الملف فارغ.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"max_upload_mb": MAX_UPLOAD_MB},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "engine": "PaddleOCR", "lang": "ar"}


@app.post("/api/ocr")
async def ocr_pdf(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    _validate_pdf_upload(file, content)

    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"ocr_{job_id}_", dir=UPLOAD_DIR))
    pdf_path = work_dir / "input.pdf"

    try:
        pdf_path.write_bytes(content)
        logger.info("OCR start: %s (%.2f MB)", file.filename, len(content) / (1024 * 1024))
        result = ArabicOcrEngine.get().recognize_pdf(str(pdf_path))
        logger.info(
            "OCR done: %s pages=%s lines=%s",
            file.filename,
            result.page_count,
            result.line_count,
        )
        return JSONResponse(
            {
                "ok": True,
                "filename": file.filename,
                "page_count": result.page_count,
                "line_count": result.line_count,
                "full_text": result.full_text,
                "pages": [_page_payload(p) for p in result.pages],
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
async def ocr_pdf_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """Stream OCR results page-by-page as NDJSON."""
    content = await file.read()
    _validate_pdf_upload(file, content)

    filename = file.filename or "document.pdf"
    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"ocr_{job_id}_", dir=UPLOAD_DIR))
    pdf_path = work_dir / "input.pdf"
    pdf_path.write_bytes(content)
    size_mb = len(content) / (1024 * 1024)
    logger.info("OCR stream start: %s (%.2f MB)", filename, size_mb)

    def event_stream():
        engine = ArabicOcrEngine.get()
        pages_done = 0
        line_count = 0
        try:
            total = engine.count_pdf_pages(str(pdf_path))
            yield json.dumps(
                {
                    "type": "start",
                    "filename": filename,
                    "page_count": total,
                },
                ensure_ascii=False,
            ) + "\n"

            for page in engine.iter_pdf_pages(str(pdf_path)):
                pages_done += 1
                line_count += len(page.lines)
                payload = {
                    "type": "page",
                    "filename": filename,
                    "page_count": total,
                    "pages_done": pages_done,
                    "line_count": line_count,
                    "page": _page_payload(page),
                }
                yield json.dumps(payload, ensure_ascii=False) + "\n"

            yield json.dumps(
                {
                    "type": "done",
                    "filename": filename,
                    "page_count": total,
                    "pages_done": pages_done,
                    "line_count": line_count,
                },
                ensure_ascii=False,
            ) + "\n"
            logger.info(
                "OCR stream done: %s pages=%s lines=%s",
                filename,
                pages_done,
                line_count,
            )
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
