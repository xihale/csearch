import os

# offline-first + fragmentation-free CUDA allocator before any HF/torch import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
