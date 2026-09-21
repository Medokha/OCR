"""Download sherif1313/Arabic-English-handwritten-OCR-v3 into models/.

Uses unverified SSL (same pattern as download_models.py) for corporate proxies.
Streams large shards to disk with progress.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

HF_ID = "sherif1313/Arabic-English-handwritten-OCR-v3"
ROOT = Path(__file__).resolve().parent / "models"
TARGET = ROOT / "Arabic-English-handwritten-OCR-v3"
CTX = ssl._create_unverified_context()

SKIP_SUFFIXES = (
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
)


def api_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "paddle-ocr-setup"})
    with urllib.request.urlopen(req, context=CTX, timeout=120) as resp:
        return resp.read()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest.relative_to(TARGET)}", flush=True)
        return

    req = urllib.request.Request(url, headers={"User-Agent": "paddle-ocr-setup"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {dest.name} ...", flush=True)
    with urllib.request.urlopen(req, context=CTX, timeout=600) as resp:
        total = resp.headers.get("Content-Length")
        total_n = int(total) if total and total.isdigit() else None
        done = 0
        chunk = 1024 * 1024
        with tmp.open("wb") as out:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if total_n:
                    pct = 100.0 * done / total_n
                    print(
                        f"\r  {dest.name}: {done / 1e6:.1f}/{total_n / 1e6:.1f} MB ({pct:.0f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  {dest.name}: {done / 1e6:.1f} MB", end="", flush=True)
    print(flush=True)
    tmp.replace(dest)
    print(f"saved {dest.relative_to(TARGET)} ({dest.stat().st_size:,} bytes)", flush=True)


def list_files(repo_id: str) -> list[str]:
    api = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=1"
    entries = json.loads(api_get(api).decode("utf-8"))
    files: list[str] = []
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e.get("path") or ""
        low = path.lower()
        if any(low.endswith(s) for s in SKIP_SUFFIXES):
            continue
        if low.startswith(".") or "/." in low:
            continue
        files.append(path)
    return files


def main() -> None:
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Listing {HF_ID} ...", flush=True)
    files = list_files(HF_ID)
    if not files:
        raise SystemExit(f"No files listed for {HF_ID}")
    # Prefer config/tokenizer first, then weight shards
    files.sort(key=lambda p: (0 if not p.endswith((".safetensors", ".bin", ".pt")) else 1, p))
    print(f"{len(files)} files -> {TARGET}", flush=True)
    for rel in files:
        url = f"https://huggingface.co/{HF_ID}/resolve/main/{rel}"
        try:
            download_file(url, TARGET / rel)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {rel}: {exc}", file=sys.stderr, flush=True)
            raise
    print("Done.", flush=True)
    print("Form UI: enable handwriting checkbox on /form", flush=True)


if __name__ == "__main__":
    main()
