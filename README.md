# Transcript to Text (TTT)

Dự án mã nguồn mở 100% chuyên dụng bóc tách giọng nói thành văn bản (Speech-to-Text / Audio & Video Transcription) siêu chuẩn, bảo mật cục bộ, hỗ trợ mọi định dạng video/audio và link YouTube/URL, xuất đa định dạng (TXT, SRT, VTT, JSON, CSV, TSV, Markdown).

Tất cả các model AI đều được thiết lập mặc định chạy ở chế độ **Quantized Q4 / INT8** siêu nhẹ, tiết kiệm tới 75% dung lượng RAM và VRAM mà vẫn giữ nguyên độ chính xác nguyên bản.

---

## ⚡ Khởi chạy nhanh trên Windows (1-Click)

Dự án đi kèm 2 file `.bat` để người dùng không rành lập trình cũng mở được ngay:

- **`run_app.bat`**: 
  - **Cách 1:** Kéo thả trực tiếp file video/audio bất kỳ vào biểu tượng `run_app.bat` để tự động bóc tách.
  - **Cách 2:** Nhấp đúp chuột để mở giao diện dòng lệnh, dán link YouTube hoặc đường dẫn file và chọn model AI (hoặc bấm `0` để lấy phụ đề YouTube không cần AI).
- **`run_web.bat`**: 
  - Nhấp đúp chuột để khởi động server Web UI cục bộ và tự động mở trình duyệt web tại địa chỉ `http://127.0.0.1:8000`.

---

## ⚡ Bóc tách YouTube KHÔNG CẦN Model AI (Tốc độ 1 giây)

Đối với video YouTube, bạn **không bắt buộc phải dùng model AI**! 
Hệ thống tích hợp tính năng tự động trích xuất phụ đề có sẵn (bao gồm cả manual subtitles do tác giả tải lên và auto-generated captions do YouTube tạo sẵn):
- **Tốc độ tức thì:** 1-2 giây hoàn tất, không phải tải audio, không tốn RAM/VRAM và không cần tải mô hình AI.
- **Tự động chuyển tiếp (Fallback):** Nếu video YouTube không có phụ đề sẵn, hệ thống sẽ tự động dùng Faster-Whisper AI để nghe và bóc tách.
- **Lệnh thực hiện:** `ttt "<link-youtube>" --no-ai` hoặc chọn mục `[0]` trong `run_app.bat`.

---

## 📥 Thanh tiến trình tải Model AI thời gian thực (Real-time %)

Khi tải mô hình Whisper lần đầu từ Hugging Face:
- Hệ thống hiển thị trực quan tên mô hình đang tải (ví dụ: `large-v3-turbo`).
- Hiển thị phần trăm hoàn thành theo thời gian thực: `45.2% (365.1MB / 809.0MB)`.
- Khi mô hình đã tải xong 1 lần, các lần chạy sau sẽ nhận diện cache cục bộ ngay lập tức mà không tải lại.

---

## 🧠 Bảng so sánh chi tiết các Model AI & Yêu cầu phần cứng

Toàn bộ model đều được cấu hình tải và suy luận ở chế độ **Q4 / INT8 Quantization**, giúp máy tính phổ thông không có card đồ họa rời (chỉ dùng CPU) cũng xử lý mượt mà.

| Model | Số tham số | Dung lượng Q4 / INT8 | RAM / VRAM tối thiểu | Tốc độ tương đối | Độ chính xác (WER) | Đánh giá & Mục đích sử dụng |
|---|---|---|---|---|---|---|
| **`tiny`** | 39 Triệu | ~40 MB | 500 MB RAM | ~32x (Cực nhanh) | Trung bình (~12-15%) | Thích hợp test pipeline nhanh, máy siêu yếu, âm thanh rất rõ tiếng Anh. |
| **`base`** | 74 Triệu | ~75 MB | 1 GB RAM | ~16x (Rất nhanh) | Khá (~9-11%) | Phù hợp podcast, hội thoại tiếng Anh chuẩn, ít từ chuyên ngành. |
| **`small`** | 244 Triệu | ~250 MB | 2 GB RAM / VRAM | ~6x (Nhanh) | Tốt (~6-8%) | **Cân bằng rất tốt cho máy văn phòng yếu**, tiếng Việt nhận diện ổn định. |
| **`medium`** | 769 Triệu | ~770 MB | 4 GB RAM / VRAM | ~2x (Vừa phải) | Rất cao (~4-5%) | Nhận diện tiếng Việt chuẩn xác cao, chịu được âm thanh hội thảo, tạp âm nhẹ. |
| **`large-v3-turbo`** ⭐ | 809 Triệu | ~810 MB | 3 - 4 GB RAM / VRAM | ~4 - 6x (Nhanh vượt trội) | **Siêu chuẩn (SOTA ~3%)** | 🏆 **KHUYÊN DÙNG SỐ 1**: Độ chính xác tương đương large-v3 nhưng tốc độ nhanh gấp đôi và tiêu tốn ít bộ nhớ hơn. |
| **`large-v3`** | 1550 Triệu | ~1.5 GB | 6 - 8 GB RAM / VRAM | ~1x (Chậm nhất) | **Siêu chuẩn tối đa (~3%)** | Dành cho audio cực khó, tiếng địa phương nặng, nhiều người nói đè lên nhau, máy có GPU 6GB+ VRAM. |

> **Cơ chế Q4 / INT8:** Trong code, hệ thống tự động ép kiểu lượng tử hóa `compute_type="int8"` cho cả CPU và GPU CUDA. Trọng số mô hình được nén từ Float32/Float16 xuống số nguyên 8-bit, vừa tránh tràn bộ nhớ (Out-Of-Memory), vừa kích hoạt tập lệnh phần cứng tăng tốc (AVX-512 VNNI trên CPU và Tensor Cores trên GPU).

---

## 🌟 Điểm nổi bật của dự án

- **100% Open Source:** Hoàn toàn miễn phí, không phụ thuộc API bên ngoài, không tốn tiền OpenAI key, dữ liệu âm thanh ở lại 100% trên máy của bạn.
- **Đầu vào đa dạng:**
  - File cục bộ: `.mp4`, `.mkv`, `.avi`, `.mov`, `.flv`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`,...
  - Link trực tuyến: YouTube, Vimeo, Facebook, TikTok và mọi nguồn do `yt-dlp` hỗ trợ.
- **Lọc khoảng lặng Silero VAD:** Loại bỏ tiếng thở, nhạc nền, tiếng ồn tĩnh, triệt tiêu 100% hiện tượng "ảo giác lặp từ" (hallucination) thường gặp ở Whisper gốc.
- **Mốc thời gian từng từ (Word-level timestamps):** Phù hợp làm phụ đề karaoke, tự động cắt video ngắn (Shorts / Reels / TikTok).
- **Xuất 7 định dạng đồng thời:**
  - `TXT`: Văn bản thuần (kèm/không kèm timestamps).
  - `SRT`: Phụ đề SubRip chuẩn cho CapCut, Premiere, DaVinci.
  - `VTT`: Phụ đề WebVTT cho player web.
  - `JSON`: Dữ liệu phân tích chi tiết (từ, mốc thời gian, xác suất).
  - `CSV` / `TSV`: Dạng bảng cho Excel / phân tích dữ liệu.
  - `MD`: Markdown ghi chú phân đoạn thời gian (Notion, Obsidian).

---

## 🚀 Cài đặt môi trường

### 1. Yêu cầu hệ thống
- Hệ điều hành: Windows 10/11, macOS, hoặc Linux.
- Python 3.9+
- FFmpeg (đã thêm vào PATH hệ thống).

### 2. Cài đặt các thư viện
```bash
cd C:\Users\Admin\Downloads\transcript_to_text

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Cài đặt gói lệnh dòng lệnh (CLI)
pip install -e .
```

---

## 📖 Hướng dẫn sử dụng

### 1. Sử dụng File Batch (Dễ nhất cho Windows)
- Bấm đúp **`run_app.bat`** để chạy giao diện dòng lệnh tương tác hoặc kéo thả file vào file bat.
- Bấm đúp **`run_web.bat`** để mở giao diện web trên trình duyệt.

### 2. Dòng lệnh CLI (`ttt` hoặc `transcript-to-text`)

```bash
# Bóc tách video bất kỳ (mặc định dùng model base Q4, xuất txt, srt, vtt, json)
ttt video.mp4

# Bóc tách link YouTube với model siêu chuẩn large-v3-turbo, xuất tất cả định dạng:
ttt "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -m large-v3-turbo -f all -o ./output

# Bóc tách tiếng Việt có lọc khoảng lặng VAD và lưu vào thư mục transcripts/
ttt "bai_giang.mp4" -l vi -m medium -f srt,vtt,txt -o ./transcripts
```

#### Các tùy chọn CLI:
| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `source` | Đường dẫn file video/audio hoặc link YouTube | *(Bắt buộc)* |
| `-o, --output-dir` | Thư mục lưu file kết quả | Thư mục hiện tại |
| `-m, --model` | Mô hình (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3`) | `base` |
| `-l, --language` | Mã ngôn ngữ (`vi`, `en`, `auto`,...) | `auto` |
| `-f, --format` | Danh sách định dạng (`txt,srt,vtt,json,csv,tsv,md` hoặc `all`) | `txt,srt,vtt,json` |
| `--device` | Thiết bị tính toán (`auto`, `cuda`, `cpu`) | `auto` |
| `--compute-type` | Kiểu dữ liệu lượng tử hóa (`int8`, `float16`, `float32`) | `int8` (Q4) |
| `--vad / --no-vad` | Lọc khoảng lặng Silero VAD | `True` |
| `--word-timestamps` | Trích xuất mốc thời gian từng từ | `True` |
| `--timestamps-in-txt` | Chèn mốc thời gian vào file .txt | `False` |

---

### 3. Giao diện Web (Web UI)

Khởi động qua terminal:
```bash
ttt web --port 8000
```
Truy cập: `http://127.0.0.1:8000`

---

### 4. Lập trình Python (Library API)

```python
from transcript_to_text import transcribe_source

result, files = transcribe_source(
    source="video.mp4",        # hoặc link YouTube
    model_size="large-v3-turbo",
    language="vi",
    compute_type="int8",       # Lượng tử hóa Q4 tiết kiệm RAM
    formats=["txt", "srt", "vtt", "json", "md"],
    output_dir="./output"
)

print(f"Ngôn ngữ: {result.language}")
print(f"Thời lượng: {result.duration}s")
print(f"Nội dung: {result.text}")
print(f"File đã xuất: {files}")
```

---

## 🧪 Kiểm thử

Chạy kiểm thử tự động xác thực toàn bộ hệ thống:
```bash
pytest tests/ -v
```

---

## 📄 Bản quyền

Phát hành theo giấy phép mã nguồn mở [MIT License](LICENSE).
