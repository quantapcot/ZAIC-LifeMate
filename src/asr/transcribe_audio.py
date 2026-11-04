#!/usr/bin/env python3
"""Batch-transcribe the challenge WAV files with MLX Whisper.

The script reads PCM WAVs directly so it does not depend on ffmpeg.  Output is
written incrementally as JSONL and can be resumed safely. This is reusable ASR
logic; orchestration entry points live in ``scripts/``.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
from mlx_whisper import transcribe


def load_pcm_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit mono PCM: {path}")
        sample_rate = wav.getframerate()
        if sample_rate != 16_000:
            raise ValueError(f"Expected 16 kHz audio, got {sample_rate}: {path}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def existing_paths(output: Path) -> set[str]:
    done: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["path"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="WAV paths or directories")
    parser.add_argument("--path-list", type=Path, help="Text file with one WAV path per line")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", default="mlx-community/whisper-large-v3-turbo"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--initial-prompt")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only process <sample_id>/<sample_id>.wav, excluding history audio",
    )
    args = parser.parse_args()

    wavs: list[Path] = []
    raw_paths = list(args.paths)
    if args.path_list:
        raw_paths.extend(
            line.strip()
            for line in args.path_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not raw_paths:
        parser.error("provide WAV paths/directories or --path-list")
    for raw in raw_paths:
        path = Path(raw)
        wavs.extend(sorted(path.rglob("*.wav")) if path.is_dir() else [path])
    if args.main_only:
        wavs = [path for path in wavs if path.stem == path.parent.name]
    if args.limit is not None:
        wavs = wavs[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_paths(args.output)
    pending = [path for path in wavs if str(path) not in done]
    print(f"{len(pending)} pending / {len(wavs)} total", flush=True)

    started = time.monotonic()
    with args.output.open("a", encoding="utf-8") as fout:
        for index, path in enumerate(pending, 1):
            decode_options = {"beam_size": args.beam_size} if args.beam_size else {}
            result = transcribe(
                load_pcm_wav(path),
                path_or_hf_repo=args.model,
                language="vi",
                task="transcribe",
                temperature=0.0,
                verbose=None,
                condition_on_previous_text=False,
                initial_prompt=args.initial_prompt,
                **decode_options,
            )
            record = {
                "path": str(path),
                "text": result["text"].strip(),
                "language": result["language"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            elapsed = time.monotonic() - started
            rate = elapsed / index
            print(
                f"[{index}/{len(pending)}] {path}: {record['text']} "
                f"({rate:.2f}s/file)",
                flush=True,
            )


if __name__ == "__main__":
    main()
