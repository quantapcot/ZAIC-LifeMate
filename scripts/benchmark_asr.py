"""Local aggregate benchmark; never writes transcripts or predictions."""
import argparse
import json
from pathlib import Path
import unicodedata

import numpy as np
import torch
from src.asr import LifeMateASR


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--model", default="models/pretrained/phowhisper-large-zaloai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=10, choices=range(10, 51))
    args = parser.parse_args()
    root = Path(args.train_dir).resolve()
    asr = LifeMateASR(args.model, args.device)
    if asr.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(asr.device)
    word_errors = char_errors = word_total = char_total = 0
    latencies = []
    with (root / "train.jsonl").open(encoding="utf-8") as data:
        for line in data:
            if len(latencies) >= args.limit:
                break
            item = json.loads(line)
            path = (root / item["audio"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Audio path escapes train directory")
            reference = normalize(item["audio_text"])
            prediction = normalize(asr.transcribe(str(path)))
            word_errors += distance(reference.split(), prediction.split())
            char_errors += distance(reference, prediction)
            word_total += len(reference.split())
            char_total += len(reference)
            latencies.append(asr.last_latency_seconds)
    if len(latencies) != args.limit or not word_total or not char_total:
        raise ValueError("Insufficient samples or empty references")
    print(f"samples={len(latencies)} WER={word_errors / word_total:.4f} CER={char_errors / char_total:.4f}")
    print(f"mean_latency_s={np.mean(latencies):.3f} p95_latency_s={np.percentile(latencies, 95):.3f}")
    if asr.device.type == "cuda":
        print(f"peak_vram_bytes={torch.cuda.max_memory_allocated(asr.device)}")


if __name__ == "__main__":
    main()
