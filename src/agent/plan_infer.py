#!/usr/bin/env python3
"""One-pass local planner for ZAC LifeMate, optimized for batch inference."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from .agent_infer import ROOT, Retriever, done_ids, history_text, load_transcripts, read_jsonl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-OptiQ-4bit"

APP_KEYWORDS = {
    "smart_home": ("đèn", "quạt", "điều hòa", "điều hoà", "máy lạnh", "rèm", "ổ cắm"),
    "alarm": ("báo thức",),
    "reminders": ("nhắc",),
    "calendar": ("lịch", "cuộc hẹn", "họp", "cà phê", "đá bóng"),
    "notes": ("ghi chú",),
    "tasks": ("việc cần làm", "danh sách việc"),
    "contacts": ("danh bạ", "liên hệ", "số điện thoại"),
    "phone": ("gọi",),
    "sms": ("sms", "nhắn tin"),
    "chat_ott": ("zalo", "chat", "gửi vị trí"),
    "email": ("email", "thư điện tử"),
    "banking": ("ngân hàng", "tài khoản", "tiết kiệm", "chuyển khoản"),
    "ewallet": ("ví", "nạp điện thoại", "mã qr"),
    "bill_payment": ("hóa đơn", "hoá đơn", "tiền điện", "tiền nước", "tiền mạng", "truyền hình"),
    "stocks": ("cổ phiếu", "chứng khoán"),
    "weather": ("mưa", "thời tiết", "ngoài trời", "nắng"),
    "gold_price": ("vàng", "sjc"),
    "fuel_price": ("xăng", "dầu"),
    "translator": ("dịch",),
    "maps": ("bản đồ", "chỉ đường", "bao xa", "địa điểm"),
    "food_delivery": ("đặt món", "đồ ăn", "quán", "bún", "cơm", "phở"),
    "grocery": ("siêu thị", "đi chợ", "thực phẩm"),
    "recipes": ("công thức", "nấu"),
    "ride_hailing": ("đặt xe", "taxi", "ô tô", "xe bốn chỗ", "xe 4 chỗ", "xe 7 chỗ"),
    "public_transit": ("xe buýt", "metro", "thẻ xe"),
    "parking": ("đỗ xe", "đậu xe"),
    "flights": ("chuyến bay", "máy bay", "vé đi", "giờ bay", "check-in"),
    "train_bus_tickets": ("vé tàu", "xe khách"),
    "ecommerce": ("mua online", "đơn hàng"),
    "shopping_list": ("danh sách mua",),
    "music": ("nhạc", "bài hát", "playlist"),
    "tv": ("kênh", "tivi", "tv", "ghi hình"),
    "event_tickets": ("vé xem", "vé sự kiện", "ca nhạc", "triển lãm"),
    "habit_tracker": ("thói quen", "chuỗi ngày"),
    "pantry": ("tủ bếp", "còn trong bếp"),
}


SYSTEM = """Bạn là agent LifeMate trong ZAC 2026. Nhiệm vụ là dự đoán CHÍNH XÁC toàn bộ chuỗi tool_calls cho yêu cầu cuối.

Bạn được cung cấp state ban đầu để suy luận offline. Tuy vậy, output vẫn phải chứa các READ calls cần thiết như một agent thật (tìm contact/beneficiary, list hóa đơn/lịch/báo thức, kiểm tra số dư/giá/thời tiết/trạng thái thiết bị...) trước các WRITE calls. Không được bỏ read call chỉ vì đã nhìn thấy state.

Quy tắc:
- Transcript ASR có thể sai âm/tên; sửa theo context, state và ví dụ gần nhất.
- Chỉ xử lý yêu cầu cuối. Lịch sử đã được áp dụng vào state.
- Tôn trọng nếu/thì và chỉ chạy nhánh đúng dựa trên state/read result.
- Dùng đúng tool đã cài, đúng schema, đúng ID trong state hoặc từ kết quả read.
- Nếu bắt buộc thiếu dữ liệu không thể lấy qua tool, dùng sys.ask_missing_info. Nếu yêu cầu ngoài khả năng app thì [].
- Thời gian tương đối tính từ current_time/timezone. Chuỗi phải theo đúng thứ tự hành động.
- Output duy nhất: <calls>[{{"name":"app.tool","arguments":{{...}}}}, ...]</calls>. Không giải thích.

Ví dụ train gần nhất:
{examples}
"""


def compact_tools(catalog: dict[str, Any], allowed: set[str]) -> list[dict[str, Any]]:
    out = []
    for app in catalog["apps"]:
        for tool in app.get("tools", []):
            if tool["name"] in allowed:
                out.append(
                    {
                        "name": tool["name"],
                        "kind": tool.get("kind"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters"),
                    }
                )
    return out


def format_examples(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(rows, 1):
        blocks.append(
            f"VÍ DỤ {index}\n"
            f"current_time={row['context']['current_time']}\n"
            f"installed_tools={json.dumps(sorted(name for app in row['context']['installed_apps'].values() for name in app.get('tools', [])), ensure_ascii=False)}\n"
            f"recent_history={json.dumps(row.get('history', [])[-4:], ensure_ascii=False)}\n"
            f"request={row['audio_text']}\n"
            f"calls={json.dumps(row['label']['tool_calls'], ensure_ascii=False)}\n"
            f"results={json.dumps(row['label']['tool_results'], ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


def parse_calls(text: str) -> list[dict[str, Any]]:
    if "<calls>" in text and "</calls>" in text:
        text = text.split("<calls>", 1)[1].split("</calls>", 1)[0]
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "[":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, list):
                    return value
            except json.JSONDecodeError:
                pass
    return []


def keyword_apps(text: str) -> set[str]:
    folded = text.casefold()
    return {
        app
        for app, keywords in APP_KEYWORDS.items()
        if any(keyword in folded for keyword in keywords)
    }


class AppClassifier:
    def __init__(self, rows: list[dict[str, Any]]):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform([row["audio_text"] for row in rows])
        labels = [
            sorted({call["name"].split(".", 1)[0] for call in row["label"]["tool_calls"]})
            for row in rows
        ]
        self.encoder = MultiLabelBinarizer()
        targets = self.encoder.fit_transform(labels)
        self.model = OneVsRestClassifier(
            LogisticRegression(C=5.0, max_iter=500, class_weight="balanced"),
            n_jobs=1,
        ).fit(matrix, targets)

    def predict(self, text: str, top_k: int = 6) -> set[str]:
        probs = self.model.predict_proba(self.vectorizer.transform([text]))[0]
        order = probs.argsort()[::-1]
        chosen = [index for index in order[:top_k] if probs[index] >= 0.04]
        return {str(self.encoder.classes_[index]) for index in chosen}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "public_test"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--candidate-neighbors", type=int, default=12)
    args = parser.parse_args()

    split_dir = ROOT / args.split
    items = read_jsonl(split_dir / f"{args.split}.jsonl")
    if args.ids:
        wanted = set(args.ids)
        items = [item for item in items if item["id"] in wanted]
    if args.limit is not None:
        items = items[: args.limit]

    train = read_jsonl(ROOT / "train/train.jsonl")
    retriever = Retriever(train)
    app_classifier = AppClassifier(train)
    transcripts = load_transcripts(args.transcripts)
    catalog = json.loads((split_dir / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))

    sys.path.insert(0, str(split_dir))
    ToolEnv = importlib.import_module("zaic_tools").ToolEnv
    from mlx_lm import generate, load

    model, tokenizer = load(args.model)
    completed = done_ids(args.output)
    pending = [item for item in items if item["id"] not in completed]
    print(f"{len(pending)} pending / {len(items)} selected", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open("a", encoding="utf-8") as fout:
        for item_index, item in enumerate(pending, 1):
            request = item.get("audio_text") or transcripts.get(item["audio"], "")
            if not request:
                raise ValueError(f"Missing transcript for {item['id']}")
            neighbors = retriever.get(
                request,
                item["id"] if args.split == "train" else None,
                k=max(args.examples, args.candidate_neighbors),
            )
            examples = neighbors[: args.examples]
            allowed = {
                name
                for app in item["context"]["installed_apps"].values()
                for name in app.get("tools", [])
            } | {"sys.ask_missing_info"}
            candidate_apps = {
                call["name"].split(".", 1)[0]
                for neighbor in neighbors[: args.candidate_neighbors]
                for call in neighbor["label"]["tool_calls"]
            } | keyword_apps(request) | app_classifier.predict(request) | {"sys"}
            for turn in item.get("history", []):
                for call in turn.get("tool_calls", []):
                    candidate_apps.add(call["name"].split(".", 1)[0])
            if len(item["context"]["installed_apps"]) <= 4:
                candidate_apps.update(item["context"]["installed_apps"])
            candidate_allowed = {
                name for name in allowed if name.split(".", 1)[0] in candidate_apps
            }
            if not candidate_allowed:
                candidate_allowed = allowed
            state_path = split_dir / f"{args.split}_environment" / item["id"] / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            relevant_state = {
                key: value
                for key, value in state.items()
                if key != "_meta" and key in candidate_apps
            }
            compact_context = {
                key: value
                for key, value in item["context"].items()
                if key != "installed_apps"
            }
            prompt_data = {
                "context": compact_context,
                "history": history_text(item, transcripts, max_turns=6),
                "request_asr": request,
                "candidate_tool_schemas": compact_tools(catalog, candidate_allowed),
                "all_installed_tool_names": sorted(allowed),
                "initial_state": relevant_state,
            }
            messages = [
                {"role": "system", "content": SYSTEM.format(examples=format_examples(examples))},
                {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            raw = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
            proposed = parse_calls(raw)
            env = ToolEnv(args.split, item["id"], root=ROOT)
            invalid = []
            for call in proposed:
                if not isinstance(call, dict):
                    invalid.append({"call": call, "error": "not an object"})
                    continue
                name = call.get("name")
                arguments = call.get("arguments") or {}
                validation = env.validate(name, arguments)
                if not validation["ok"]:
                    invalid.append({"call": call, "error": validation["error"]})
                    continue
                env.call(name, arguments)

            row: dict[str, Any] = {
                "id": item["id"],
                "tool_calls": env.get_tool_calls(),
                "state_hash": env.get_state_hash(),
                "_request": request,
                "_raw": raw,
                "_invalid": invalid,
            }
            if args.split == "train":
                gold_env = ToolEnv(args.split, item["id"], root=ROOT)
                for call in item["label"]["tool_calls"]:
                    gold_env.call(call["name"], call["arguments"])
                row["_exact_calls"] = row["tool_calls"] == item["label"]["tool_calls"]
                row["_state_match"] = row["state_hash"] == gold_env.get_state_hash()

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            elapsed = time.monotonic() - started
            print(
                f"[{item_index}/{len(pending)}] {item['id']} calls={len(row['tool_calls'])} "
                f"invalid={len(invalid)} exact={row.get('_exact_calls')} "
                f"state={row.get('_state_match')} {elapsed / item_index:.1f}s/item",
                flush=True,
            )


if __name__ == "__main__":
    main()
