"""ASR module for ZAIC-LifeMate using ChunkFormer CTC Large Vietnamese."""

from .chunkformer_vie import ChunkFormerASR, LifeMateASR, normalize_vietnamese_text

__all__ = [
    "ChunkFormerASR",
    "LifeMateASR",
    "normalize_vietnamese_text",
]

