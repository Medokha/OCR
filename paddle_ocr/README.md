# Arabic PDF OCR — PaddleOCR v3

تطبيق ويب لاستخراج النص العربي من ملفات PDF باستخدام [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (الإصدار 3.x / نموذج عربي PP-OCRv5).

## المتطلبات

- Python 3.10+
- Windows / Linux / macOS
- اتصال إنترنت لأول تشغيل (تحميل أوزان النموذج)

## التثبيت

```powershell
cd paddle_ocr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python download_models.py
```

> على Windows CPU غالباً يكفي `paddlepaddle` من PyPI. إن فشل التثبيت راجع [دليل تثبيت PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick).
>
> سكربت `download_models.py` يحمّل نموذج الكشف `PP-OCRv5_server_det` ونموذج التعرف العربي `arabic_PP-OCRv5_mobile_rec` محلياً (مفيد عند مشاكل SSL/البروكسي).
## التشغيل

```powershell
cd paddle_ocr
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000
```

افتح المتصفح على: http://127.0.0.1:8000

## الاستخدام

1. اضغط **اختر ملف PDF**
2. اضغط **ابدأ القراءة**
3. اعرض النص كاملاً أو حسب الصفحة، ثم انسخ أو حمّل TXT

## API

`POST /api/ocr` — multipart form field: `file` (PDF)

`GET /health` — فحص الخدمة
