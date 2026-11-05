# ASR PhoWhisper ZaloAI

Checkpoint: `kzap201/phowhisper-large-zaloai`, tag `v2`, commit
`576f676a5b4a9705bd7440060dabfd4ed012dd01`. Tải về
`models/pretrained/phowhisper-large-zaloai/` bằng `hf download` với `--revision v2`.
Không đưa model hoặc dữ liệu cuộc thi vào Git.

```powershell
hf download kzap201/phowhisper-large-zaloai --revision v2 --local-dir models/pretrained/phowhisper-large-zaloai
```

Checkpoint được xuất với Transformers 5.17.0; 4.57.6 không load được tokenizer.
Trên Windows, tạo môi trường riêng kế thừa PyTorch CUDA từ `lifemate`:

```powershell
conda activate lifemate
python -m venv --system-site-packages .venv-asr
.\.venv-asr\Scripts\python.exe -m pip install -r requirements-asr.txt
```

Kiểm tra PyTorch CUDA trong môi trường mới trước khi chạy. Không cài PyTorch CPU
đè lên môi trường hiện có. Sau khi tải hoàn tất, chạy với file WAV tự có:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv-asr\Scripts\python.exe -m scripts.smoke_asr path/to/audio.wav
.\.venv-asr\Scripts\python.exe -m scripts.benchmark_asr --limit 10
```

Truyền `--device cpu` nếu không có CUDA. Benchmark chỉ in metrics tổng hợp;
không ghi transcript hay dự đoán. Chọn 10–50 dòng đầu trong train, do đó không
phải tập đánh giá độc lập và có thể trùng dữ liệu huấn luyện. WER/CER dùng NFC,
chữ thường, thay dấu câu bằng khoảng trắng, gộp khoảng trắng; CER tính cả khoảng
trắng. Latency bao gồm đọc và tiền xử lý audio cùng inference, không bao gồm load
model; peak VRAM là bộ nhớ PyTorch cấp phát, bao gồm model. File dài được chia
thành các đoạn 30 giây không chồng lấn, có thể mất từ tại ranh giới.

```python
from src.asr import LifeMateASR
asr = LifeMateASR('models/pretrained/phowhisper-large-zaloai')
text = asr.transcribe('sample.wav')
print(text, asr.last_latency_seconds)
```

Model card ghi base PhoWhisper-large theo BSD-3-Clause và checkpoint được
fine-tune trên LifeMate. Nhóm cần xác minh quyền công bố, nguồn dữ liệu và quy
định của cuộc thi trước khi dùng hoặc phân phối checkpoint.

## Kiểm thử local ngày 2026-11-06

Trên RTX 5060 Laptop 8 GB, PyTorch 2.11.0+cu128 và Transformers 5.17.0,
checkpoint v2 đã load và chạy với `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
Mười dòng train đầu cho WER 1,48%, CER 0,96%, latency trung bình 0,865 giây,
p95 1,475 giây và peak PyTorch allocated 3.361.916.928 bytes. Các mẫu này có
thể nằm trong tập fine-tune, vì vậy đây không phải kết quả đánh giá độc lập.
