"""Download PaddleOCR Arabic models (works around corporate SSL inspection)."""

from __future__ import annotations

import json
import ssl
import tarfile
import urllib.request
import zipfile
from pathlib import Path

MODELS = [
    "PP-OCRv5_server_det",
    "arabic_PP-OCRv5_mobile_rec",
]

ROOT = Path(__file__).resolve().parent / "models"
CTX = ssl._create_unverified_context()


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "paddle-ocr-setup"})
    with urllib.request.urlopen(req, context=CTX, timeout=120) as resp:
        return resp.read()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip existing {dest.name}")
        return
    print(f"downloading {url}")
    data = get(url)
    dest.write_bytes(data)
    print(f"saved {dest} ({len(data)} bytes)")


def download_hf_repo(model_name: str) -> Path:
    target = ROOT / model_name
    api = f"https://huggingface.co/api/models/PaddlePaddle/{model_name}/tree/main"
    entries = json.loads(get(api).decode("utf-8"))
    files = [e["path"] for e in entries if e.get("type") == "file"]
    if not files:
        raise RuntimeError(f"No files listed for {model_name}")

    for rel in files:
        url = f"https://huggingface.co/PaddlePaddle/{model_name}/resolve/main/{rel}"
        download_file(url, target / rel)

    return target


def try_bos_tarball(model_name: str) -> Path | None:
    """Fallback: Baidu BOS official inference tarball."""
    url = (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
        f"paddle3.0.0/{model_name}_infer.tar"
    )
    tar_path = ROOT / f"{model_name}_infer.tar"
    try:
        download_file(url, tar_path)
    except Exception as exc:  # noqa: BLE001
        print(f"BOS failed for {model_name}: {exc}")
        return None

    out = ROOT / model_name
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(ROOT)
    # Sometimes extracts to model_name/ or model_name_infer/
    candidates = [ROOT / model_name, ROOT / f"{model_name}_infer"]
    for cand in candidates:
        if cand.exists():
            return cand
    return out


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        try:
            path = download_hf_repo(name)
            print(f"OK HuggingFace -> {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"HuggingFace failed for {name}: {exc}")
            path = try_bos_tarball(name)
            if path is None:
                raise SystemExit(f"Could not download {name}")
            print(f"OK BOS -> {path}")

    print("Done. Models are in:", ROOT)


if __name__ == "__main__":
    main()
