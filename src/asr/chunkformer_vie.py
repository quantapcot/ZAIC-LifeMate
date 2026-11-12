"""ChunkFormer CTC Large Vietnamese ASR Integration for ZAIC-LifeMate.

Provides an offline-ready, GPU-accelerated Vietnamese Automatic Speech Recognition (ASR)
interface using the ChunkFormer architecture (khanhld/chunkformer-ctc-large-vie).
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
import warnings
from pathlib import Path
from typing import List, Optional, Union

import torch

# Filter common benign warnings from pydub / distributed redirects on Windows
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

try:
    from chunkformer import ChunkFormerModel
except ImportError as exc:
    raise ImportError(
        "chunkformer package is required. "
        "Please install it in the lifemate-chunkformer conda environment."
    ) from exc

logger = logging.getLogger("ZAIC-LifeMate.ASR")


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text for evaluation (WER/CER) and uniform downstream consumption.

    Steps:
      1. Normalize Unicode to NFC.
      2. Convert to lowercase.
      3. Replace punctuation and symbols with whitespace.
      4. Collapse consecutive whitespaces and strip.
    """
    if not text:
        return ""
    # 1. NFC normalization
    text = unicodedata.normalize("NFC", text)
    # 2. Lowercase
    text = text.lower()
    # 3. Replace punctuation / special characters with space
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # 4. Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ChunkFormerASR:
    """Vietnamese Speech Recognition interface using ChunkFormer CTC Large.

    Attributes:
        model_path (str): Path to local checkpoint directory.
        device (torch.device): Inference device (cuda:0 or cpu).
        last_latency_seconds (float): Latency of the most recent transcribe call.
        last_peak_vram_mb (float): Peak allocated VRAM in MB during the most recent call.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/pretrained/chunkformer-ctc-large-vie",
        device: Optional[Union[str, torch.device]] = None,
        chunk_size: int = 64,
        left_context_size: int = 128,
        right_context_size: int = 128,
        total_batch_duration: int = 1800,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"ChunkFormer model directory does not exist: {self.model_path}"
            )

        # Device selection & CPU fallback
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self.chunk_size = chunk_size
        self.left_context_size = left_context_size
        self.right_context_size = right_context_size
        self.total_batch_duration = total_batch_duration

        self.last_latency_seconds: float = 0.0
        self.last_peak_vram_mb: float = 0.0

        logger.info("Loading ChunkFormer model from %s to %s", self.model_path, self.device)
        load_t0 = time.perf_counter()
        self.model = ChunkFormerModel.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.load_time_seconds = time.perf_counter() - load_t0
        logger.info("ChunkFormer model loaded in %.3f s", self.load_time_seconds)

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """Transcribe a single audio file to Vietnamese text.

        Args:
            audio_path: Path to the .wav audio file.

        Returns:
            Transcribed Vietnamese text string.
        """
        results = self.batch_transcribe([audio_path])
        return results[0] if results else ""

    def batch_transcribe(self, audio_paths: List[Union[str, Path]]) -> List[str]:
        """Transcribe a batch of audio files.

        Args:
            audio_paths: List of file paths to audio files.

        Returns:
            List of transcribed strings corresponding to audio_paths.
        """
        if not audio_paths:
            return []

        resolved_paths: List[str] = []
        for p in audio_paths:
            resolved = str(Path(p).resolve())
            if not os.path.isfile(resolved):
                raise FileNotFoundError(f"Audio file not found: {p} (resolved: {resolved})")
            resolved_paths.append(resolved)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            raw_transcripts = self.model.batch_decode(
                audio_paths=resolved_paths,
                chunk_size=self.chunk_size,
                left_context_size=self.left_context_size,
                right_context_size=self.right_context_size,
                total_batch_duration=self.total_batch_duration,
            )
        elapsed = time.perf_counter() - start_time

        self.last_latency_seconds = elapsed
        if self.device.type == "cuda":
            self.last_peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        else:
            self.last_peak_vram_mb = 0.0

        return [t.strip() if isinstance(t, str) else "" for t in raw_transcripts]


# Alias for backward compatibility
LifeMateASR = ChunkFormerASR

