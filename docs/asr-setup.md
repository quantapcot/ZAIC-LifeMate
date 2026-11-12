# Hướng dẫn Thiết lập và Sử dụng ChunkFormer CTC Large Vietnamese ASR

## 1. Thông tin Model

- **Model ID:** `khanhld/chunkformer-ctc-large-vie`
- **Hugging Face Hub:** https://huggingface.co/khanld/chunkformer-ctc-large-vie
- **GitHub:** https://github.com/khanld/chunkformer
- **Kiến trúc:** ChunkFormer + CTC (Conformer / Chunk-based CTC for streaming and non-streaming ASR)
- **Checkpoint Commit SHA:** `311fc03558a895dc2b32957f2fb4236c7fb1455b`
- **Thư mục lưu weights local:** `models/pretrained/chunkformer-ctc-large-vie/`
- **Các tệp trọng số chính:**
  - `config.yaml`
  - `global_cmvn`
  - `vocab.txt`
  - `pytorch_model.bin` (~1.4 GB)

---

## 2. Thiết lập Môi trường Conda (`lifemate-chunkformer`)

Môi trường khuyến nghị: **Python 3.11** trên Windows 11 với NVIDIA GPU (CUDA 12.8 / 12.9).

### Bước 2.1: Tạo môi trường Conda mới
```powershell
conda create -n lifemate-chunkformer python=3.11 -y
conda activate lifemate-chunkformer
```

### Bước 2.2: Cài đặt PyTorch hỗ trợ CUDA
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Kiểm tra CUDA:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"
```

### Bước 2.3: Cài đặt ChunkFormer không cần DeepSpeed (Bypass Windows Build Issue)
Gói `chunkformer 1.2.2` trên PyPI khai báo phụ thuộc `deepspeed>=0.14.0`, vốn yêu cầu môi trường Linux/C++ compiler phức tạp. Đối với việc inference ASR, DeepSpeed là không cần thiết.

Thực hiện cài đặt `--no-deps`, sau đó cài các runtime dependencies:
```powershell
pip install chunkformer==1.2.2 --no-deps
pip install -r requirements-asr.txt
```

---

## 3. Tải Checkpoint Offline

Tải toàn bộ checkpoint về thư mục dự án `models/pretrained/chunkformer-ctc-large-vie`:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('khanhld/chunkformer-ctc-large-vie', local_dir='models/pretrained/chunkformer-ctc-large-vie')"
```

---

## 4. Chế độ Chạy Offline (Zero Network)

Sau khi đã tải checkpoint về `models/pretrained/chunkformer-ctc-large-vie/`, model có thể hoạt động hoàn toàn offline không cần kết nối mạng:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

---

## 5. Kiểm thử nhanh (Smoke Test)

### Chạy trên GPU (mặc định):
```powershell
python -m scripts.smoke_chunkformer --audio "train/audio/train_0001/train_0001.wav"
```

### Chạy trên CPU fallback:
```powershell
python -m scripts.smoke_chunkformer --audio "train/audio/train_0001/train_0001.wav" --device cpu
```

---

## 6. Chạy Benchmark Đánh giá

Chạy benchmark trên tập dữ liệu train để đo WER, CER, Độ trễ (Mean/P95), Real-Time Factor (RTF) và Peak VRAM:

```powershell
# Đánh giá 10 mẫu:
python -m scripts.benchmark_asr --limit 10 --output-json outputs/asr_benchmark_chunkformer_10.json

# Đánh giá 30 mẫu:
python -m scripts.benchmark_asr --limit 30 --output-json outputs/asr_benchmark_chunkformer_30.json
```

---

## 7. Hướng dẫn Tích hợp vào Action Model

Dành cho các thành viên phát triển Agent / Tool Calling:

```python
from src.asr import ChunkFormerASR

# Khởi tạo (tự động nhận GPU nếu có, tự fallback CPU nếu không có)
asr = ChunkFormerASR(
    model_path="models/pretrained/chunkformer-ctc-large-vie",
    device="cuda",  # hoặc "cpu"
)

# Nhận diện đơn tệp:
transcript = asr.transcribe("path/to/user_voice.wav")
print("Transcribed:", transcript)
print(f"Latency: {asr.last_latency_seconds:.3f} s")
print(f"Peak VRAM: {asr.last_peak_vram_mb:.2f} MB")

# Nhận diện hàng loạt (batch):
transcripts = asr.batch_transcribe(["audio1.wav", "audio2.wav"])
```

---

## 8. Kết quả Thực nghiệm Thực tế (NVIDIA RTX 5060 Laptop GPU, 8GB VRAM)

| Chỉ số | PhoWhisper Large ZaloAI (Cũ) | ChunkFormer CTC Large (Mới) | Đánh giá & Cải thiện |
| :--- | :--- | :--- | :--- |
| **Kiến trúc** | Whisper Seq2Seq (Encoder-Decoder) | ChunkFormer CTC | Non-autoregressive, streamable |
| **Model Load Time** | ~4.5 s | **~2.0 s** | Nhanh hơn 2.2x |
| **Peak VRAM** | 3,361 MB (~3.36 GB) | **619 - 659 MB (~0.64 GB)** | **Tiết kiệm >80% VRAM**, cực kỳ an toàn cho GPU 8GB |
| **Mean Latency / mẫu** | 0.865 s | **0.100 s - 0.154 s** | **Nhanh hơn 5.6x - 8.6x** |
| **P95 Latency** | 1.475 s | **0.098 s - 0.455 s** | Ổn định, không bị nghẽn |
| **Real-Time Factor (RTF)** | ~0.12 | **0.015 - 0.022** | **Xử lý nhanh hơn thời gian thực 45x - 67x** |
| **Offline Inference** | Hỗ trợ | **Hỗ trợ đầy đủ** (`HF_HUB_OFFLINE=1`) | Hoàn toàn offline |
