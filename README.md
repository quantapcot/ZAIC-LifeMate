# ZAIC LifeMate

Pipeline thử nghiệm cho bài toán LifeMate: nhận dạng tiếng nói tiếng Việt,
lập kế hoạch gọi tool, thực thi Agent và kiểm tra submission.

## Cấu trúc dự án

| Thư mục | Công dụng | Ví dụ |
| --- | --- | --- |
| `src/` | Mã nguồn chính của hệ thống | Các module Python |
| `src/asr/` | Nhận dạng giọng nói tiếng Việt và xử lý transcript | `transcribe_audio.py`, `select_low_asr.py` |
| `src/agent/` | Logic Agent: hiểu yêu cầu, chọn tool, thực thi và xử lý kết quả | `agent_infer.py`, `plan_infer.py`, `run_calls.py` |
| `src/training/` | Code huấn luyện/fine-tune mô hình | Thêm `train_lora.py`, `prepare_sft.py` tại đây |
| `src/evaluation/` | Logic kiểm tra, tính điểm và chuẩn hóa submission | `finalize_submission.py`, `export_result.py` |
| `configs/` | Cấu hình model và các quá trình train/inference | YAML/JSON config |
| `scripts/` | Các entry point mỏng để chạy pipeline | `run_agent.py`, `transcribe_audio.py` |
| `train/` | Dữ liệu train do Ban tổ chức cung cấp, chỉ lưu local | JSONL, audio, simulator tools |
| `public_test/` | Dữ liệu public test dùng để tạo submission, chỉ lưu local | JSONL, audio, simulator tools |
| `models/pretrained/` | Model gốc tải về trước khi fine-tune | Model ASR/LLM |
| `models/checkpoints/` | Kết quả model/checkpoint sau từng lần train | LoRA adapter, checkpoint |
| `outputs/` | Artifact tạo ra khi chạy hệ thống | Transcript, session, submission |
| `outputs/benchmarks/` | Kết quả đánh giá và so sánh model | Metrics CSV/JSON |
| `tests/` | Kiểm thử các chức năng | Unit/integration tests |
| `docs/` | Tài liệu nghiên cứu và báo cáo kỹ thuật | Paper notes, experiment reports |

### Các cặp thư mục dễ nhầm

- `src/training/` chứa **code dùng để train**; `models/checkpoints/` chứa
  **model hoặc adapter sinh ra sau khi train**.
- `src/evaluation/` chứa **code tính/kiểm tra điểm**;
  `outputs/benchmarks/` chứa **kết quả đánh giá**.
- `src/agent/` chứa **logic thực thi Agent**; `scripts/` chỉ chứa các chương
  trình entry point gọi module để chạy pipeline.

## Dữ liệu local

Dữ liệu do Ban tổ chức cung cấp không được commit lên Git. Sau khi clone repo,
đặt dữ liệu vào đúng hai thư mục sau (giữ lại cấu trúc gốc):

```text
train/
  train.jsonl
  train_tools_catalog.json
  audio/
  train_environment/
  zaic_tools/

public_test/
  public_test.jsonl
  public_test_tools_catalog.json
  audio/
  public_test_environment/
  zaic_tools/
```

`.gitignore` giữ lại `.gitkeep` để Git lưu cấu trúc thư mục, nhưng loại dữ liệu,
model, checkpoint và output tạm chạy local. Kết quả tổng hợp nhỏ trong
`outputs/benchmarks/` có thể được commit để so sánh thí nghiệm.

## Cài đặt

Yêu cầu Python 3.10+ và macOS Apple Silicon cho MLX.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Chạy pipeline

Các lệnh dưới đây chạy từ thư mục gốc của repo.

```bash
# 1. Tạo transcript cho public test
python scripts/transcribe_audio.py public_test/audio \
  --main-only \
  --output outputs/transcripts/public.jsonl

# 2. Chạy one-pass planner
python scripts/run_plan.py \
  --split public_test \
  --transcripts outputs/transcripts/public.jsonl \
  --output outputs/predictions.jsonl

# 3. Kiểm tra và tạo submission cuối
python scripts/finalize_submission.py \
  outputs/predictions.jsonl \
  outputs/submission.jsonl
```

`scripts/run_agent.py` chạy Agent lặp nhiều bước; `scripts/run_calls.py` và
`scripts/run_reviewed_solver.py` phục vụ luồng review/thực thi tool thủ công.
