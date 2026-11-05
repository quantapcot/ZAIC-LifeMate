"""Offline Vietnamese ASR using a locally exported PhoWhisper checkpoint."""
from pathlib import Path
from time import perf_counter
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class LifeMateASR:
    def __init__(self, model_path, device="cuda"):
        self.path = Path(model_path).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.path}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass device='cpu' to use CPU")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(str(self.path), local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(self.path), local_files_only=True, torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.last_latency_seconds = None

    def transcribe(self, audio_path: str) -> str:
        start = perf_counter()
        self.last_latency_seconds = None
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        except (RuntimeError, OSError) as exc:
            raise ValueError(f"Cannot read audio file: {path}") from exc
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError("Audio must be nonempty and contain finite samples")
        audio = audio.mean(axis=1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        # Explicit chunks prevent the processor silently truncating audio over 30 seconds.
        texts = []
        with torch.inference_mode():
            for offset in range(0, len(audio), 30 * 16000):
                inputs = self.processor(audio[offset:offset + 30 * 16000], sampling_rate=16000,
                                        return_tensors="pt", return_attention_mask=True)
                features = inputs.input_features.to(device=self.device, dtype=self.dtype)
                mask = inputs.attention_mask.to(self.device)
                tokens = self.model.generate(features, attention_mask=mask, language="vi",
                                             task="transcribe", do_sample=False,
                                             num_beams=1, max_new_tokens=256)
                texts.append(self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_latency_seconds = perf_counter() - start
        return " ".join(text for text in texts if text)
