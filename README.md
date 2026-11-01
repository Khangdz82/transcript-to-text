# Transcript to Text (TTT)

Dự án mã nguồn mở 100% chuyên dụng bóc tách giọng nói thành văn bản (Speech-to-Text / Audio & Video Transcription) siêu chuẩn, bảo mật cục bộ, hỗ trợ mọi định dạng video/audio và link YouTube/URL, xuất đa định dạng (TXT, SRT, VTT, JSON, CSV, TSV, Markdown).

---

## 🌟 Điểm nổi bật

- **100% Open Source:** Hoàn toàn miễn phí, không phụ thuộc API trả phí (OpenAI API key không cần thiết), chạy offline bảo mật tuyệt đối trên máy tính cá nhân.
- **Đầu vào đa dạng:**
  - File cục bộ: `.mp4`, `.mkv`, `.avi`, `.mov`, `.flv`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`,...
  - Link trực tuyến: YouTube, Vimeo và mọi nền tảng được `yt-dlp` hỗ trợ.
- **Độ chính xác siêu chuẩn:**
  - Sử dụng backend **Faster-Whisper** (CTranslate2) tối ưu tốc độ nhanh gấp 4 lần so với Whisper gốc.
  - Tích hợp bộ lọc **Silero VAD** (Voice Activity Detection) loại bỏ khoảng lặng, tiếng thở, nhạc nền, chống hiện tượng lặp từ (hallucination).
  - Tự động nhận diện ngôn ngữ (hỗ trợ Tiếng Việt, Tiếng Anh, Tiếng Trung, Tiếng Nhật,...) hoặc ép ngôn ngữ cố định.
  - Trích xuất mốc thời gian chi tiết đến từng từ (**Word-level timestamps**).
- **Xuất 7 định dạng phổ biến:**
  - `TXT`: Văn bản thuần (hỗ trợ kèm/không kèm timestamp).
  - `SRT`: Phụ đề SubRip chuẩn cho phần mềm dựng phim (Premiere, CapCut, DaVinci).
  - `VTT`: Phụ đề WebVTT chuẩn web/video player.
  - `JSON`: Dữ liệu chi tiết chứa segment, từ, độ tin cậy, logprob.
  - `CSV` / `TSV`: Dạng bảng cho xử lý dữ liệu và phân tích.
  - `MD`: Markdown ghi chú phân đoạn theo mốc thời gian (Obsidian, Notion).
- **Giao diện & Tiện ích:**
  - **CLI tiện lợi:** Lệnh `ttt` hoặc `transcript-to-text` với thanh tiến trình trực quan.
  - **Web UI:** Giao diện web hiện đại, kéo thả file hoặc dán link YouTube, tải phụ đề 1-click.
  - **Python API:** Thư viện Python sạch sẽ, dễ dàng nhúng vào dự án khác.

---

## 🚀 Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.9+
- FFmpeg (đã có trong PATH hệ thống)

### 2. Cài đặt thư viện
```bash
cd C:\Users\Admin\Downloads\transcript_to_text

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt

# Cài đặt gói lệnh dòng lệnh (CLI)
pip install -e .
```

---

## 📖 Hướng dẫn sử dụng

### 1. Dòng lệnh (CLI)

#### Bóc tách video / audio cục bộ:
```bash
# Bóc tách cơ bản (tự động nhận ngôn ngữ, xuất txt, srt, vtt, json)
ttt video.mp4

# Chọn mô hình 'large-v3-turbo' siêu chuẩn, xuất toàn bộ định dạng (-f all)
ttt video.mp4 -m large-v3-turbo -f all -o ./output

# Chỉ định Tiếng Việt (-l vi), lưu vào thư mục transcripts/
ttt "bai_giang.mp4" -l vi -o ./transcripts
```

#### Bóc tách từ link YouTube:
```bash
ttt "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -m base -f srt,txt
```

#### Các tùy chọn nâng cao:
| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `source` | Đường dẫn file video/audio hoặc link YouTube | *(Bắt buộc)* |
| `-o, --output-dir` | Thư mục lưu file kết quả | Thư mục hiện tại |
| `-m, --model` | Mô hình (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3`) | `base` |
| `-l, --language` | Mã ngôn ngữ (`vi`, `en`, `auto`,...) | `auto` |
| `-f, --format` | Danh sách định dạng (`txt,srt,vtt,json,csv,tsv,md` hoặc `all`) | `txt,srt,vtt,json` |
| `--device` | Thiết bị tính toán (`auto`, `cuda`, `cpu`) | `auto` |
| `--compute-type` | Kiểu dữ liệu (`auto`, `int8`, `float16`, `float32`) | `auto` |
| `--vad / --no-vad` | Bật/tắt lọc khoảng lặng Silero VAD | `True` |
| `--word-timestamps` | Trích xuất mốc thời gian từng từ | `True` |
| `--timestamps-in-txt` | Chèn mốc thời gian vào file .txt | `False` |

---

### 2. Giao diện Web (Web UI)

Khởi động giao diện web:
```bash
ttt web --port 8000
```
Mở trình duyệt truy cập: `http://127.0.0.1:8000`

Giao diện hỗ trợ:
- Dán link YouTube / URL video trực tiếp.
- Kéo thả file MP4, MKV, MP3, WAV từ máy tính.
- Chọn mô hình Whisper và ngôn ngữ.
- Xem trực tiếp nội dung transcript theo từng mốc thời gian.
- Tải về ngay 7 định dạng phụ đề và sao chép văn bản chỉ với 1 click.

---

### 3. Lập trình Python (Library API)

```python
from transcript_to_text import transcribe_source

# Bóc tách 1 dòng lệnh duy nhất:
result, files = transcribe_source(
    source="video.mp4",        # hoặc link YouTube "https://..."
    model_size="large-v3-turbo",
    language="vi",
    formats=["txt", "srt", "vtt", "json", "md"],
    output_dir="./output"
)

print(f"Ngôn ngữ nhận diện: {result.language}")
print(f"Thời lượng: {result.duration} giây")
print(f"Toàn bộ nội dung: {result.text}")
print(f"Các file đã tạo: {files}")
```

---

## 🧪 Kiểm thử (Testing)

Chạy bộ kiểm thử tự động:
```bash
pytest tests/ -v
```

---

## 📄 Bản quyền

Phát hành theo giấy phép mã nguồn mở [MIT License](LICENSE).
