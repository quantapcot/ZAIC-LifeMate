#!/usr/bin/env python3
"""Select low-confidence ASR rows or merge a prompted second pass.

This module contains reusable ASR selection logic; use the wrapper in
``scripts/`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--alternate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--min-improvement", type=float, default=0.015)
    args = parser.parse_args()

    train = rows(Path("train/train.jsonl"))
    base = rows(args.base)
    alternate = rows(args.alternate) if args.alternate else []
    all_text = [row["audio_text"] for row in train] + [row["text"] for row in base + alternate]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(all_text)
    train_matrix = matrix[: len(train)]
    candidate_matrix = matrix[len(train) :]
    scores = linear_kernel(candidate_matrix, train_matrix).max(axis=1)

    base_scores = {row["path"]: float(scores[index]) for index, row in enumerate(base)}
    if not alternate:
        selected = [row["path"] for row in base if base_scores[row["path"]] < args.threshold]
        args.output.write_text("".join(path + "\n" for path in selected), encoding="utf-8")
        print(f"selected {len(selected)}/{len(base)} below {args.threshold}")
        return

    offset = len(base)
    alt_scores = {
        row["path"]: float(scores[offset + index]) for index, row in enumerate(alternate)
    }
    alt_by_path = {row["path"]: row for row in alternate}
    merged = []
    replaced = 0
    for row in base:
        alt = alt_by_path.get(row["path"])
        if alt and alt_scores[row["path"]] >= base_scores[row["path"]] + args.min_improvement:
            merged.append(alt)
            replaced += 1
        else:
            merged.append(row)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    print(f"merged {len(merged)} rows; replaced {replaced} with prompted ASR")


if __name__ == "__main__":
    main()
