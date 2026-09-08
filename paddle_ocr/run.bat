@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set FLAGS_enable_pir_api=0
set FLAGS_enable_pir_in_executor=0
set FLAGS_use_mkldnn=0
echo Starting Arabic OCR at http://127.0.0.1:8000
uvicorn main:app --host 127.0.0.1 --port 8000
