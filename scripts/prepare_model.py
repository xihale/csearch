#!/usr/bin/env python
"""Materialize quantized EmbeddingGemma 2 ONNX weights into models/.

Downloads (or copies from the HF cache) the int4-GPU and int8-CPU exports of
onnx-community/embeddinggemma-2-ONNX as real files - required because the HF
cache's symlink layout breaks ONNX external-data path resolution.

Usage: python scripts/prepare_model.py [--dest models] [--variants q4f16 int8]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = "onnx-community/embeddinggemma-2-ONNX"
SHARED = [
    "tokenizer.json", "tokenizer_config.json", "config.json",
    "config_sentence_transformers.json",
]
VISION_SHARED = [
    "preprocessor_config.json", "processor_config.json",
]
WEIGHTS = {
    "q4f16": [
        "onnx/model_q4f16.onnx", "onnx/model_q4f16.onnx_data",
    ],
    "int8": [
        "onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data",
    ],
}
VISION_WEIGHTS = {
    "q4f16": [
        "onnx/vision_encoder_q4f16.onnx", "onnx/vision_encoder_q4f16.onnx_data",
    ],
    "int8": [
        "onnx/vision_encoder_quantized.onnx", "onnx/vision_encoder_quantized.onnx_data",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"),
                    help="destination root (default: <repo>/models)")
    ap.add_argument("--variants", nargs="+", default=["q4f16", "int8"],
                    choices=list(WEIGHTS))
    ap.add_argument("--vision", action="store_true",
                    help="experimental: also fetch vision_encoder ONNX weights (+93MB q4f16) for image indexing")
    args = ap.parse_args()

    dest = Path(args.dest) / "embeddinggemma-2-ONNX"
    dest.mkdir(parents=True, exist_ok=True)

    required_patterns = set(SHARED)
    for v in args.variants:
        required_patterns.update(WEIGHTS[v])
        if args.vision:
            required_patterns.update(VISION_WEIGHTS[v])
    if args.vision:
        required_patterns.update(VISION_SHARED)
    patterns = sorted(required_patterns)
    env = dict(os.environ)
    # the corporate proxy breaks python's certifi; use the system CA bundle,
    # and bypass possibly stale mirrors
    ca = "/etc/ssl/certs/ca-certificates.crt"
    if Path(ca).exists():
        env.setdefault("SSL_CERT_FILE", ca)
        env.setdefault("REQUESTS_CA_BUNDLE", ca)
    env["HF_ENDPOINT"] = "https://huggingface.co"

    code = (
        "from huggingface_hub import snapshot_download;"
        f"print(snapshot_download({REPO!r}, local_dir={str(dest)!r}, allow_patterns={patterns!r}))"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)

    # flatten onnx/* to the layout Embedder expects (weights stay under onnx/)
    for variant in args.variants:
        for f in WEIGHTS[variant]:
            if not (dest / f).exists():
                print(f"missing after download: {f}", file=sys.stderr)
                return 1
        if args.vision:
            for f in VISION_WEIGHTS[variant]:
                if not (dest / f).exists():
                    print(f"missing after download: {f}", file=sys.stderr)
                    return 1
    print(f"model ready: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
