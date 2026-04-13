# Data Pipeline — FIT HCMUS RAG Chatbot

Mục tiêu: chuẩn hóa pipeline theo module rõ ràng để dễ bảo trì và tái lập kết quả.

---

## Tổng quan luồng dữ liệu

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NGUỒN DỮ LIỆU                                   │
│                                                                         │
│  ┌─────────────────────────┐   ┌─────────────────────────────────────┐  │
│  │   Static .txt files     │   │   FIT HCMUS Website (PDF Links)     │  │
│  │   Database/*.txt        │   │   fit.hcmus.edu.vn/Default.aspx     │  │
│  │                         │   │   tabid=97 (CTĐT, Quyết định, ...)  │  │
│  │  • CTĐT 5 ngành K2023   │   └──────────────┬──────────────────────┘  │
│  │  • Đề cương môn học     │                  │                         │
│  │  • Điều kiện tốt nghiệp │                  ▼                         │
│  │  • Quy định đào tạo     │   ┌─────────────────────────────────────┐  │
│  │  • Quy định ngoại ngữ   │   │     crawl_fit_pdfs.py               │  │
│  │  • Liên thông ĐH–ThS    │   │                                     │  │
│  └────────────┬────────────┘   │  1. Crawl 36 trang CTĐT theo tabid  │  │
│               │                │  2. Thu thập link PDF                │  │
│               │                │  3. Tải PDF → Data/.tmp_pdfs/        │  │
│               │                └──────────────┬──────────────────────┘  │
└───────────────┼──────────────────────────────┼─────────────────────────┘
                │                              │
                │                              ▼
                │              ┌───────────────────────────────────────────┐
                │              │         OCR PIPELINE (llm_ocr_pdf.py)     │
                │              │                                           │
                │              │  BƯỚC 1 — Scan Detection (PyMuPDF)        │
                │              │    • Đọc text layer từ PDF                │
                │              │    • Nếu > 30% trang trống → scan PDF    │
                │              │                                           │
                │              │  BƯỚC 2A — PaddleOCR (PP-Structure)      │
                │              │    • Layout analysis: phân biệt bảng/text │
                │              │    • OCR thô, confidence scoring          │
                │              │                                           │
                │              │  BƯỚC 2B — Qwen2.5-VL 7B (local GPU)     │
                │              │    • Vùng BẢNG → Markdown table OCR       │
                │              │    • Vùng TEXT → tiếng Việt dấu đầy đủ   │
                │              │                                           │
                │              │  Fallback thay thế (nếu không có GPU):   │
                │              │    • paddle-only  → PaddleOCR thuần      │
                │              │    • gemini       → Gemini Vision API     │
                │              │    • gpt4o        → GPT-4o Vision API     │
                │              │    • tesseract    → pytesseract (cũ)      │
                │              └──────────────┬────────────────────────────┘
                │                             │
                │                             ▼
                │              ┌───────────────────────────────────────────┐
                │              │     Database/pdf_crawled/*.txt            │
                │              │  (PDF scan → text đã OCR, UTF-8)         │
                │              └──────────────┬────────────────────────────┘
                │                             │
                └─────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ETL PIPELINE (run_pipeline.py)                     │
│                                                                         │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────────────┐ │
│  │ Loaders        │    │ Splitters      │    │ Embeddings             │ │
│  │ (loaders.py)   │───►│ (splitters.py) │───►│ (embeddings.py)        │ │
│  │                │    │                │    │                        │ │
│  │ • Đọc *.txt    │    │ outline mode:  │    │ paraphrase-            │ │
│  │   từ Database/ │    │  • Split theo  │    │ multilingual-          │ │
│  │ • Chuẩn hóa    │    │    heading #   │    │ mpnet-base-v2          │ │
│  │   metadata     │    │  • 1000 chars  │    │ (768-dim)              │ │
│  │   source field │    │  • 200 overlap │    │                        │ │
│  │                │    │                │    │ HuggingFace            │ │
│  │                │    │ recursive mode:│    │ Inference API          │ │
│  │                │    │  • Recursive   │    │                        │ │
│  │                │    │    char split  │    │                        │ │
│  └────────────────┘    └────────────────┘    └──────────┬─────────────┘ │
│                                                         │               │
│  ┌────────────────────────────────────────────────────  ▼  ───────────┐ │
│  │                  Vector Store (vector_store.py)                    │ │
│  │                                                                    │ │
│  │   Qdrant Cloud / Local                                             │ │
│  │   Collection: ITUS_mpnet_1000v1                                    │ │
│  │   Upsert with metadata: source, he_dao_tao, nganh, nam            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
Data/
├── crawl_fit_pdfs.py        # Crawler: FIT website → tải PDF → OCR → .txt
├── llm_ocr_pdf.py           # OCR pipeline: PaddleOCR + Qwen2.5-VL / Gemini
├── run_pipeline.py          # CLI: ETL .txt → Qdrant
├── .env.example             # Template biến môi trường
├── .env                     # Biến môi trường (không commit)
│
├── pipeline/                # Modules ETL
│   ├── config.py            # Quản lý env + validate input
│   ├── loaders.py           # Nạp .txt, chuẩn hóa metadata source
│   ├── splitters.py         # Chunking: outline / recursive
│   ├── embeddings.py        # HuggingFace embedding client
│   ├── vector_store.py      # Qdrant upsert + retrieval
│   └── pipeline.py          # Orchestration: ingest → chunk → embed → index
│
├── Database/                # Dữ liệu đã xử lý (input cho ETL)
│   ├── *.txt                # Static documents (CTĐT, đề cương, quy định...)
│   └── pdf_crawled/         # PDF từ FIT website đã OCR → .txt
│
└── .tmp_pdfs/               # PDF tạm (tự động xóa sau khi OCR xong)
```

---

## Cách chạy từng bước

### Bước 1 — Crawl PDF từ FIT HCMUS + OCR

```bash
cd Data

# Dùng Qwen2.5-VL (cần GPU NVIDIA ≥ 8GB VRAM)
python crawl_fit_pdfs.py --ocr-model qwen

# Chỉ PaddleOCR (CPU, không cần GPU)  
python crawl_fit_pdfs.py --ocr-model paddle-only

# Dùng Gemini Vision API (cần GEMINI_API_KEY)
python crawl_fit_pdfs.py --ocr-model gemini

# Xem preview, chưa tải
python crawl_fit_pdfs.py --dry-run

# Chỉ crawl năm 2023, 2024
python crawl_fit_pdfs.py --ocr-model qwen --years 2023 2024

# Bỏ qua OCR (chỉ lấy PDF có text layer)
python crawl_fit_pdfs.py --skip-ocr
```

Output: `Database/pdf_crawled/*.txt`  
Format tên file: `{he_dao_tao}__{nganh}__{nam}__{loai_tai_lieu}.txt`

### Bước 1b — OCR riêng PDF đã tải (tùy chọn)

```bash
# OCR toàn bộ file trong .tmp_pdfs
python llm_ocr_pdf.py --input-dir .tmp_pdfs --output-dir Database/pdf_crawled --model qwen --skip-existing

# OCR một file cụ thể
python llm_ocr_pdf.py --input .tmp_pdfs/abc123.pdf --model gemini
```

### Bước 2 — Build vector index (ETL → Qdrant)

```bash
# Cài package
pip install -r requirements.txt

# Tạo .env từ template
cp .env.example .env
# → Điền: QDRANT_URL, QDRANT_API_KEY, HUGGINGFACE_API_KEY

# Chạy pipeline mặc định
python run_pipeline.py

# Tùy chỉnh tham số
python run_pipeline.py \
  --collection ITUS_mpnet_1000v2 \
  --chunk-strategy outline \
  --chunk-size 1000 \
  --chunk-overlap 200

# Kèm crawl web FIT HCMUS
python run_pipeline.py \
  --enable-web-crawl \
  --web-start-urls "https://www.fit.hcmus.edu.vn/vn/Default.aspx?tabid=36" \
  --web-allowed-domains "www.fit.hcmus.edu.vn,courses.fit.hcmus.edu.vn" \
  --web-max-pages 80
```

---

## Biến môi trường

| Biến | Bắt buộc | Mô tả |
|------|----------|-------|
| `QDRANT_URL` | ✅ | URL Qdrant Cloud hoặc local |
| `QDRANT_API_KEY` | ✅ | API key Qdrant |
| `HUGGINGFACE_API_KEY` | ✅ | Inference API để tạo embedding |
| `GEMINI_API_KEY` | Nếu dùng Gemini OCR | Google AI Studio API key |
| `OPENAI_API_KEY` | Nếu dùng GPT-4o OCR | OpenAI API key |
| `QDRANT_COLLECTION_NAME` | ❌ | Mặc định: `ITUS_mpnet_1000v1` |
| `EMBEDDING_MODEL_NAME` | ❌ | Mặc định: `paraphrase-multilingual-mpnet-base-v2` |
| `CHUNK_STRATEGY` | ❌ | `outline` hoặc `recursive` (mặc định: `outline`) |
| `CHUNK_SIZE` | ❌ | Mặc định: `1000` |
| `CHUNK_OVERLAP` | ❌ | Mặc định: `200` |
| `QDRANT_FORCE_RECREATE` | ❌ | `true` để xóa và tạo lại collection |

---

## Cài đặt dependencies OCR

```bash
# Qwen2.5-VL pipeline (cần GPU)
pip install paddlepaddle-gpu paddleocr transformers accelerate qwen-vl-utils torch torchvision

# Chỉ PaddleOCR (CPU)
pip install paddlepaddle paddleocr pymupdf pillow

# Gemini Vision API
pip install google-genai

# GPT-4o Vision API
pip install openai

# Legacy: pytesseract
pip install pytesseract pillow
# + cài Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Lưu ý

- Notebook `Vector_Database.ipynb` và `Data_Analysis.ipynb` được giữ lại để tham khảo.
- Luồng vận hành chính nên chạy từ `run_pipeline.py` để quản lý phiên bản dễ dàng hơn.
- PDF tạm trong `.tmp_pdfs/` được giữ lại sau khi crawl để tránh tải lại. Xóa bằng: `rm -rf Data/.tmp_pdfs`
