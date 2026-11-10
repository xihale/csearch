"""Quantized EmbeddingGemma 2 inference for code search.

Primary backend is a raw ONNX Runtime session over the official int4/int8
exports (onnx-community/embeddinggemma-2-ONNX):
  * GPU: onnx/model_q4f16.onnx (int4 weights, fp16 activations, ~157MB)
  * CPU: onnx/model_quantized.onnx (int8 dynamic, ~314MB)
The graph takes input_ids + attention_mask plus three required multimodal
feature inputs that must be zero-length arrays for text-only use, and outputs
a mean-pooled sentence_embedding that we L2-normalize.

A "torch" backend (SentenceTransformer bf16, text-only load) is kept as the
quality reference and fallback for machines without onnxruntime.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

import csearch  # noqa: F401 - offline-first + CUDA allocator in __init__.py

MODEL_ID = "onnx-community/embeddinggemma-2-ONNX"
# weights live next to the package, so the CLI/MCP server work from any cwd
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / MODEL_ID.split("/")[-1]
NATIVE_DIM = 768
SUPPORTED_DIMS = (128, 256, 512, 768)
QUERY_PROMPT = "task: code retrieval | query: "  # CodeRetrieval task prefix
MM_INPUTS = ("image_features", "video_features", "audio_features")

# variant -> weights file under <model_dir>/onnx/; the execution provider is
# always auto-selected (CUDA when available: q4f16=79 ch/s, int8=38 ch/s on CUDA
# vs int8=3 ch/s on CPU, so CPU EP is strictly a fallback)
ORT_VARIANTS = {
    "q4f16": "model_q4f16.onnx",
    "int8": "model_quantized.onnx",
}
VISION_VARIANTS = {
    "q4f16": "vision_encoder_q4f16.onnx",
    "int8": "vision_encoder_quantized.onnx",
}


def doc_text(title: str, code: str) -> str:
    """Asymmetric retrieval format for corpus chunks (Google model card)."""
    return f"title: {title} | text: {code}"


def doc_image_text(title: str) -> str:
    """Asymmetric multimodal format for image/visual corpus items."""
    return f"title: {title} | text: <|image|>"


class Embedder:
    """Encode code chunks and natural-language queries into a shared space."""

    def __init__(
        self,
        model_dir: str | os.PathLike,
        backend: str = "ort",  # "ort" | "torch"
        variant: str | None = None,  # ort only: "cuda" | "cpu"; None = auto
        dim: int = NATIVE_DIM,
        batch_size: int = 16,
        max_length: int = 1024,
        cpu: bool = False,  # ort only: force CPUExecutionProvider (e.g. GPU busy)
    ):
        if dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {dim}")
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.backend = backend
        self.dim = dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.variant = variant
        self.cpu = cpu

        if backend == "ort":
            self._tok, self._sess = self._load_ort()
        elif backend == "torch":
            self._st = self._load_torch()
        else:
            raise ValueError(f"backend must be 'ort' or 'torch', got {backend!r}")

    # ---- loading -------------------------------------------------------

    def _pick_variant(self) -> tuple[str, str]:
        """Resolve (weights path, provider). Provider: CUDA when available."""
        import onnxruntime as ort

        has_cuda = not self.cpu and "CUDAExecutionProvider" in ort.get_available_providers()
        if self.variant:
            if self.variant not in ORT_VARIANTS:
                raise ValueError(f"variant must be one of {list(ORT_VARIANTS)}")
            name = ORT_VARIANTS[self.variant]
        else:
            name = ORT_VARIANTS["q4f16" if has_cuda else "int8"]
        path = self.model_dir / "onnx" / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run scripts/prepare_model.py to fetch quantized "
                f"weights, or pass --model <dir>"
            )
        return str(path), "CUDAExecutionProvider" if has_cuda else "CPUExecutionProvider"

    def _pick_vision_variant(self) -> tuple[str | None, str]:
        """Resolve vision encoder ONNX path and provider matching _pick_variant."""
        import onnxruntime as ort

        has_cuda = not self.cpu and "CUDAExecutionProvider" in ort.get_available_providers()
        if self.variant:
            if self.variant not in VISION_VARIANTS:
                raise ValueError(f"variant must be one of {list(VISION_VARIANTS)}")
            name = VISION_VARIANTS[self.variant]
        else:
            name = VISION_VARIANTS["q4f16" if has_cuda else "int8"]
        path = self.model_dir / "onnx" / name
        provider = "CUDAExecutionProvider" if has_cuda else "CPUExecutionProvider"
        if not path.exists():
            return None, provider
        return str(path), provider

    def _load_ort(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        path, provider = self._pick_variant()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if provider == "CUDAExecutionProvider":
            # Grow on demand instead of doubling: with the default (kNextPowerOfTwo)
            # a resident server reserves ~4GB for a ~160MB model and starves the
            # other sessions that share the GPU.
            providers = [(provider, {"arena_extend_strategy": "kSameAsRequested"})]
        else:
            providers = [provider]
        sess = ort.InferenceSession(path, so, providers=providers)
        tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._weights, self._provider = Path(path).name, provider
        return tok, sess

    def _load_torch(self):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        # text-only load: skip vision/audio encoders (271M of the 740M params)
        return SentenceTransformer(
            "google/embeddinggemma-2",
            device=device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"vision_config": None, "audio_config": None},
        )

    # ---- encoding ------------------------------------------------------

    def encode_docs(self, titles: list[str], texts: list[str],
                    progress: Callable[[int, int], None] | None = None,
                    dim: int | None = None) -> np.ndarray:
        target_dim = self.dim if dim is None else dim
        if target_dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {target_dim}")
        payloads = [doc_text(t, c) for t, c in zip(titles, texts)]
        return self._encode(payloads, is_query=False, progress=progress, dim=target_dim)

    def encode_queries(self, queries: list[str], dim: int | None = None) -> np.ndarray:
        target_dim = self.dim if dim is None else dim
        if target_dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {target_dim}")
        return self._encode(queries, is_query=True, dim=target_dim)

    def encode_images(
        self,
        titles: list[str],
        images: list,
        progress: Callable[[int, int], None] | None = None,
        dim: int | None = None,
    ) -> np.ndarray:
        target_dim = self.dim if dim is None else dim
        if target_dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {target_dim}")
        if not images:
            return np.empty((0, target_dim), dtype=np.float32)

        if self.backend == "ort":
            vision_path, provider = self._pick_vision_variant()
            if vision_path is not None and (self.model_dir / "processor_config.json").exists():
                import onnxruntime as ort
                from PIL import Image
                from transformers import AutoProcessor

                so = ort.SessionOptions()
                so.log_severity_level = 3
                if provider == "CUDAExecutionProvider":
                    providers = [(provider, {"arena_extend_strategy": "kSameAsRequested"})]
                else:
                    providers = [provider]

                vision_sess = None
                proc = None
                try:
                    vision_sess = ort.InferenceSession(vision_path, so, providers=providers)
                    proc = AutoProcessor.from_pretrained(self.model_dir)
                    n = len(images)
                    out = np.empty((n, NATIVE_DIM), dtype=np.float32)
                    for i, (title, img_item) in enumerate(zip(titles, images)):
                        if isinstance(img_item, (str, Path)):
                            with Image.open(img_item) as raw_img:
                                pil_img = raw_img.convert("RGB")
                        else:
                            pil_img = img_item.convert("RGB")

                        enc = proc(images=[[pil_img]], text=[doc_image_text(title)], return_tensors="np")
                        img_feats = vision_sess.run(
                            ["image_features"],
                            {
                                "pixel_values": enc["pixel_values"].astype(np.float32, copy=False),
                                "pixel_position_ids": enc["image_position_ids"].astype(np.int64, copy=False),
                            },
                        )[0]
                        emb = self._sess.run(
                            ["sentence_embedding"],
                            {
                                "input_ids": enc["input_ids"].astype(np.int64, copy=False),
                                "attention_mask": enc["attention_mask"].astype(np.int64, copy=False),
                                "image_features": img_feats.astype(np.float32, copy=False),
                                "video_features": np.zeros((0, 512), np.float32),
                                "audio_features": np.zeros((0, 512), np.float32),
                            },
                        )[0]
                        out[i] = emb[0]
                        if progress is not None:
                            progress(i + 1, n)
                    return _mrl_normalize(out, target_dim)
                finally:
                    if vision_sess is not None:
                        del vision_sess
                    if proc is not None:
                        del proc

        return self.encode_docs(
            titles,
            [f"[image {t}]" for t in titles],
            progress=progress,
            dim=target_dim,
        )

    def _encode(self, payloads: list[str], is_query: bool,
                progress: Callable[[int, int], None] | None = None,
                dim: int | None = None) -> np.ndarray:
        target_dim = self.dim if dim is None else dim
        if not payloads:
            return np.empty((0, target_dim), dtype=np.float32)
        if self.backend == "torch":
            return self._encode_torch(payloads, is_query, show_bar=progress is not None, dim=target_dim)
        return self._encode_ort(payloads, QUERY_PROMPT if is_query else None, progress, dim=target_dim)

    def _encode_ort(self, payloads: list[str], prefix: str | None,
                    progress: Callable[[int, int], None] | None = None,
                    dim: int | None = None) -> np.ndarray:
        target_dim = self.dim if dim is None else dim
        if prefix:
            payloads = [prefix + p for p in payloads]
        empty = {name: np.zeros((0, 512), np.float32) for name in MM_INPUTS}
        n = len(payloads)
        out = np.empty((n, NATIVE_DIM), dtype=np.float32)
        if n > self.batch_size:
            # Longest batch first avoids growing CUDA arena allocations and OOM.
            # Bound the length prepass to one batch; retain only lengths.
            lengths = []
            for i in range(0, n, self.batch_size):
                tokenized = self._tok(
                    payloads[i : i + self.batch_size], truncation=True,
                    max_length=self.max_length,
                )
                lengths.extend(len(ids) for ids in tokenized["input_ids"])
                del tokenized
            order = np.argsort(lengths, kind="stable")[::-1]
        else:
            order = np.arange(n)
        done = 0
        for i in range(0, n, self.batch_size):
            idx = order[i : i + self.batch_size]
            enc = self._tok(
                [payloads[j] for j in idx], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="np",
            )
            emb = self._sess.run(
                ["sentence_embedding"],
                {
                    "input_ids": enc["input_ids"].astype(np.int64, copy=False),
                    "attention_mask": enc["attention_mask"].astype(np.int64, copy=False),
                    **empty,
                },
            )[0]
            out[idx] = emb
            done += len(idx)
            if progress is not None:
                progress(done, n)
        return _mrl_normalize(out, target_dim)

    def _encode_torch(self, payloads: list[str], is_query: bool,
                      show_bar: bool = False,
                      dim: int | None = None) -> np.ndarray:
        target_dim = self.dim if dim is None else dim
        kwargs = dict(
            batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=show_bar,
            truncate_dim=target_dim,
        )
        if is_query:
            kwargs["prompt_name"] = "CodeRetrieval"
        return self._st.encode(payloads, **kwargs).astype(np.float32, copy=False)

    @property
    def info(self) -> str:
        if self.backend == "ort":
            return f"ort/{Path(self._weights).stem}/{self._provider.removesuffix('ExecutionProvider')}"
        return "torch/sentence-transformers-bf16"


def _mrl_normalize(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation to `dim` columns + L2 re-normalization."""
    emb = np.ascontiguousarray(embeddings[:, :dim], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    emb /= norms
    return emb
