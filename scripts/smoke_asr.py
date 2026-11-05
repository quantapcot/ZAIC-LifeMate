"""Run from the repository root: python -m scripts.smoke_asr AUDIO.wav."""
import argparse
import torch
from src.asr import LifeMateASR

parser = argparse.ArgumentParser()
parser.add_argument("audio")
parser.add_argument("--model", default="models/pretrained/phowhisper-large-zaloai")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
asr = LifeMateASR(args.model, args.device)
if asr.device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(asr.device)
print(asr.transcribe(args.audio))
print(f"latency_seconds={asr.last_latency_seconds:.3f}")
if asr.device.type == "cuda":
    print(f"peak_vram_bytes={torch.cuda.max_memory_allocated(asr.device)}")
