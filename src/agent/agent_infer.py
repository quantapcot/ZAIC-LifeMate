#!/usr/bin/env python3
"""Run a local Qwen tool-using agent against the ZAC LifeMate simulator."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Repository root. Runtime data lives in ``train/`` and ``public_test/`` at
# this level, while this module intentionally stays under ``src/agent/``.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "mlx-community/Qwen3.5-2B-4bit"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_transcripts(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    for row in read_jsonl(path):
        audio_path = row["path"]
        out[audio_path] = row["text"]
        marker = "audio/"
        if marker in audio_path:
            out[audio_path[audio_path.index(marker) :]] = row["text"]
    return out


def catalog_tools(catalog: dict[str, Any], allowed: set[str]) -> list[dict[str, Any]]:
    tools = []
    for app in catalog["apps"]:
        for spec in app.get("tools", []):
            if spec["name"] not in allowed:
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec.get("description", ""),
                        "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
    return tools


def history_text(
    item: dict[str, Any], transcripts: dict[str, str], max_turns: int | None = None
) -> str:
    if not item.get("history"):
        return "(không có)"
    chunks = []
    turns = item["history"][-max_turns:] if max_turns else item["history"]
    for turn in turns:
        if turn["role"] == "user":
            text = turn.get("audio_text") or transcripts.get(turn.get("audio", ""))
            text = text or "(audio chưa có transcript)"
            chunks.append(f"Người dùng: {text}")
        else:
            chunks.append(f"Trợ lý: {turn.get('text', '')}")
            if turn.get("tool_calls"):
                chunks.append("Tool calls đã chạy: " + json.dumps(turn["tool_calls"], ensure_ascii=False))
            if turn.get("tool_results"):
                chunks.append("Kết quả đã có: " + json.dumps(turn["tool_results"], ensure_ascii=False))
    return "\n".join(chunks)


class Retriever:
    def __init__(self, train: list[dict[str, Any]]):
        self.train = train
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform([x["audio_text"] for x in train])

    def get(self, text: str, exclude_id: str | None, k: int = 2) -> list[dict[str, Any]]:
        query = self.vectorizer.transform([text])
        scores = linear_kernel(query, self.matrix)[0]
        ranked = scores.argsort()[::-1]
        out = []
        for index in ranked:
            row = self.train[int(index)]
            if row["id"] == exclude_id:
                continue
            out.append(row)
            if len(out) == k:
                break
        return out


def example_text(examples: list[dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(examples, 1):
        blocks.append(
            f"VÍ DỤ {index}\n"
            f"Thời điểm: {row['context']['current_time']}\n"
            f"Yêu cầu: {row['audio_text']}\n"
            f"Chuỗi đúng: {json.dumps(row['label']['tool_calls'], ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


SYSTEM_BASE = """Bạn là agent LifeMate cho bộ ZAC 2026. Hãy hoàn thành yêu cầu CUỐI của người dùng bằng các function được cấp.

Quy tắc bắt buộc:
- Audio được ASR nên có thể sai âm, đặc biệt tên người/app; hãy suy luận theo ngữ cảnh và dữ liệu tool.
- Chỉ xử lý yêu cầu cuối; lịch sử đã được thực thi vào state và chỉ dùng để hiểu tham chiếu.
- Dùng tool đọc khi cần tìm ID, hóa đơn, số dư, lịch, dự báo, thiết bị hoặc lựa chọn cụ thể. Không bịa ID.
- Thực hiện luôn tool ghi khi đủ dữ liệu. Nếu thiếu tham số bắt buộc và không thể lấy bằng tool, gọi sys.ask_missing_info.
- Tôn trọng điều kiện nếu/thì: gọi tool đọc điều kiện trước, chỉ thực hiện nhánh đúng theo kết quả.
- Hiểu thời gian tương đối theo current_time và timezone. Giữ đúng thứ tự các hành động.
- Sau khi toàn bộ yêu cầu đã hoàn tất, trả lời ngắn gọn bằng tiếng Việt và KHÔNG gọi thêm tool.
- Mỗi lượt chỉ nên gọi MỘT function để quan sát kết quả trước khi quyết định bước tiếp.

Các ví dụ gần nhất từ tập train (có thể khác dữ liệu/điều kiện hiện tại):
{examples}
"""


CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_value(raw: str, schema: dict[str, Any]) -> Any:
    raw = html.unescape(raw.strip())
    expected = schema.get("type")
    try:
        value = json.loads(raw)
        if expected == "string" and not isinstance(value, str):
            return str(value)
        return value
    except json.JSONDecodeError:
        if expected == "integer":
            return int(float(raw.replace(",", "")))
        if expected == "number":
            return float(raw.replace(",", ""))
        if expected == "boolean":
            return raw.casefold() in {"true", "1", "yes", "có"}
        return raw.strip('"')


def parse_first_call(output: str, specs: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    match = CALL_RE.search(output)
    if not match:
        return None
    name, body = match.groups()
    properties = ((specs.get(name) or {}).get("parameters") or {}).get("properties") or {}
    arguments = {}
    for param, raw in PARAM_RE.findall(body):
        arguments[param] = parse_value(raw, properties.get(param) or {})
    return {"name": name, "arguments": arguments}


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["id"] for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "public_test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, default=9)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    split_dir = ROOT / args.split
    items = read_jsonl(split_dir / f"{args.split}.jsonl")
    if args.ids:
        wanted = set(args.ids)
        items = [x for x in items if x["id"] in wanted]
    if args.limit is not None:
        items = items[: args.limit]

    train = read_jsonl(ROOT / "train/train.jsonl")
    retriever = Retriever(train)
    transcripts = load_transcripts(args.transcripts)
    catalog = json.loads((split_dir / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))
    all_specs = {t["name"]: t for app in catalog["apps"] for t in app.get("tools", [])}

    sys.path.insert(0, str(split_dir))
    ToolEnv = importlib.import_module("zaic_tools").ToolEnv
    from mlx_lm import generate, load
    import mlx.core as mx

    mx.set_cache_limit(128 * 1024 * 1024)

    model, tokenizer = load(args.model)
    completed = done_ids(args.output)
    pending = [x for x in items if x["id"] not in completed]
    print(f"{len(pending)} pending / {len(items)} selected", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with args.output.open("a", encoding="utf-8") as fout:
        for item_index, item in enumerate(pending, 1):
            if args.split == "train":
                request = item["audio_text"]
            else:
                full_path = str(split_dir / item["audio"])
                request = transcripts.get(full_path) or transcripts.get(item["audio"], "")
                if not request:
                    raise ValueError(f"Missing transcript for {item['id']}: {full_path}")

            neighbors = retriever.get(request, item["id"] if args.split == "train" else None, k=4)
            examples = neighbors[:2]
            system = SYSTEM_BASE.format(examples=example_text(examples))
            compact_context = {k:v for k,v in item['context'].items() if k != 'installed_apps'}
            user = (
                "BỐI CẢNH HIỆN TẠI:\n"
                + json.dumps(compact_context, ensure_ascii=False)
                + "\n\nLỊCH SỬ ĐÃ XẢY RA:\n"
                + history_text(item, transcripts)
                + "\n\nYÊU CẦU CUỐI (transcript ASR):\n"
                + request
            )
            allowed = {
                tool
                for app in item["context"]["installed_apps"].values()
                for tool in app.get("tools", [])
            } | {"sys.ask_missing_info"}
            candidate_apps = set(list(item['context']['installed_apps'])[:3])
            candidate_apps.add('sys')
            candidate_apps.update(c['name'].split('.')[0] for neighbor in neighbors for c in neighbor['label']['tool_calls'])
            candidate_apps.update(c['name'].split('.')[0] for turn in item.get('history',[]) for c in turn.get('tool_calls',[]))
            selected_allowed = {name for name in allowed if name.split('.')[0] in candidate_apps}
            tools = catalog_tools(catalog, selected_allowed)
            specs = {name: all_specs[name] for name in allowed if name in all_specs}
            session_dir = ROOT / "outputs" / (
                "work_sessions" if args.split == "public_test" else "eval_sessions"
            )
            session_dir.mkdir(exist_ok=True)
            session_path = session_dir / (item['id'] + '.pickle')
            if session_path.exists():
                with session_path.open('rb') as stream:
                    env = pickle.load(stream)
            else:
                env = ToolEnv(args.split, item["id"], root=ROOT)
                with session_path.open('wb') as stream:
                    pickle.dump(env, stream)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            raw_outputs = []
            read_cache: dict[str, dict[str, Any]] = {}
            for previous in env.log:
                call = {'name':previous['name'],'arguments':previous['arguments']}
                messages.append({'role':'assistant','content':'','tool_calls':[{'type':'function','function':call}]})
                messages.append({'role':'tool','content':json.dumps(previous['result'],ensure_ascii=False)})
                if env.kind(previous['name']) == 'read':
                    read_cache[json.dumps(call,sort_keys=True,ensure_ascii=False)] = previous['result']

            for _step in range(len(env.get_tool_calls()), args.max_steps):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=args.thinking,
                )
                output = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    verbose=False,
                )
                raw_outputs.append(output)
                print(f"{item['id']} step={_step} output={output[:700]}",flush=True)
                call = parse_first_call(output, specs)
                if call is None:
                    break

                validation = env.validate(call["name"], call["arguments"])
                if not validation["ok"]:
                    result = validation
                else:
                    cache_key = json.dumps(call, sort_keys=True, ensure_ascii=False)
                    is_read = env.kind(call["name"]) == "read"
                    if is_read and cache_key in read_cache:
                        result = read_cache[cache_key]
                    else:
                        result = env.call(call["name"], call["arguments"])
                        with session_path.open('wb') as stream:
                            pickle.dump(env, stream)
                        if is_read:
                            read_cache[cache_key] = result

                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"type": "function", "function": call}
                        ],
                    }
                )
                messages.append(
                    {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}
                )

            row: dict[str, Any] = {
                "id": item["id"],
                "tool_calls": env.get_tool_calls(),
                "state_hash": env.get_state_hash(),
                "_request": request,
                "_raw_outputs": raw_outputs,
            }
            if args.split == "train":
                gold_env = ToolEnv(args.split, item["id"], root=ROOT)
                for gold_call in item["label"]["tool_calls"]:
                    gold_env.call(gold_call["name"], gold_call["arguments"])
                row["_exact_calls"] = row["tool_calls"] == item["label"]["tool_calls"]
                row["_state_match"] = row["state_hash"] == gold_env.get_state_hash()

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            elapsed = time.monotonic() - started
            print(
                f"[{item_index}/{len(pending)}] {item['id']} calls={len(row['tool_calls'])} "
                f"exact={row.get('_exact_calls')} state={row.get('_state_match')} "
                f"{elapsed / item_index:.1f}s/item",
                flush=True,
            )


if __name__ == "__main__":
    main()
