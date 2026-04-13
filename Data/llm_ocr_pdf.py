from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
import unicodedata
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.0-flash"
OPENAI_MODEL = "gpt-4o"
QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# PaddleOCR layout labels
LABEL_TABLE = "table"
LABEL_TEXT = "text"
LABEL_TITLE = "title"
LABEL_FIGURE = "figure"
LABEL_LIST = "list"

OCR_PROMPT_QWEN = """Bạn là chuyên gia OCR tiếng Việt chuyên xử lý tài liệu đại học.
Hãy trích xuất TOÀN BỘ văn bản từ vùng tài liệu này.

Yêu cầu:
- Giữ nguyên dấu thanh tiếng Việt (à, á, ả, ã, ạ, ă, â, đ, ê, ô, ơ, ư, ...)
- Nếu là BẢNG: giữ cấu trúc dùng | phân cách cột, mỗi dòng bảng một hàng
- Giữ nguyên mã môn học, tên môn học, số tín chỉ, cột điểm
- KHÔNG thêm, KHÔNG bỏ, KHÔNG diễn giải — chỉ sao chép chính xác
- Nếu không đọc được từ nào, ghi [?]

Trả về ONLY văn bản đã trích xuất, không giải thích."""

OCR_PROMPT_TABLE_QWEN = """Bạn là chuyên gia OCR tiếng Việt. Đây là VÙNG BẢNG trong tài liệu đại học.

Yêu cầu:
- Trích xuất bảng dạng Markdown: dùng | phân cách cột, --- cho header
- Giữ nguyên: mã môn học, tên môn học, số tín chỉ, điểm, học kỳ
- Các cột thường gặp: STT | Mã MH | Tên môn | TC | LT | TH | Điều kiện tiên quyết
- KHÔNG gộp, KHÔNG bỏ dòng — trích xuất CHÍNH XÁC từng dòng trong bảng
- Nếu không đọc được, ghi [?]

Trả về ONLY bảng Markdown, không giải thích."""

OCR_PROMPT_GEMINI = """Bạn là chuyên gia OCR tiếng Việt. Hãy trích xuất TOÀN BỘ văn bản từ ảnh/trang PDF này.

Yêu cầu:
- Giữ nguyên dấu thanh tiếng Việt (à, á, ả, ã, ạ, ă, â, đ, ê, ô, ơ, ư, ...)
- Giữ cấu trúc bảng biểu (dùng | để phân cách cột nếu có bảng)
- Giữ xuống dòng tự nhiên của tài liệu
- Giữ nguyên số học phần, mã môn, tên môn học
- KHÔNG thêm, KHÔNG bỏ, KHÔNG diễn giải — chỉ sao chép chính xác văn bản
- Nếu không đọc được từ nào, ghi [?] tại chỗ đó

Trả về ONLY văn bản đã trích xuất, không cần giải thích."""

REQUEST_DELAY = 1.0
MAX_RETRIES = 3
RENDER_DPI = 300


# ---------------------------------------------------------------------------
# Lazy model loaders (singleton)
# ---------------------------------------------------------------------------

_paddle_ocr = None
_qwen_model = None
_qwen_processor = None


def _get_paddle_ocr():
    """Khởi tạo PaddleOCR (lazy, singleton)."""
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu PaddleOCR: pip install paddlepaddle paddleocr"
        ) from exc

    _paddle_ocr = PaddleOCR(
        use_angle_cls=True,
        lang="vi",
        use_gpu=True,
        show_log=False,
        type="structure",
    )
    return _paddle_ocr


def _get_qwen_model():
    """Khởi tạo Qwen2.5-VL model + processor (lazy, singleton)."""
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return _qwen_model, _qwen_processor

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện Qwen: pip install transformers accelerate qwen-vl-utils torch torchvision"
        ) from exc

    print(f"  [Qwen] Loading {QWEN_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  [WARN] Không tìm thấy GPU — Qwen sẽ chạy rất chậm trên CPU")

    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    _qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    print(f"  [Qwen] Model loaded on {device}")
    return _qwen_model, _qwen_processor


# ---------------------------------------------------------------------------
# Bước 1: PaddleOCR — Layout analysis + OCR thô
# ---------------------------------------------------------------------------

def paddle_layout_analysis(page_image_bytes: bytes, verbose: bool = True) -> list[dict]:
    """
    Phân tích layout trang PDF bằng PaddleOCR.

    Returns list of regions:
      [{"type": "table"|"text"|"title"|..., "bbox": [x1,y1,x2,y2],
        "text": "raw ocr text", "confidence": float}]
    """
    ocr = _get_paddle_ocr()
    from PIL import Image
    import numpy as np

    img = Image.open(io.BytesIO(page_image_bytes)).convert("RGB")
    img_np = np.array(img)

    result = ocr.ocr(img_np, cls=True)

    regions = []
    if not result or not result[0]:
        return regions

    for line in result[0]:
        if not line:
            continue
        # PaddleOCR format: [points, (text, confidence)]
        points, (text, conf) = line
        # Bounding box from polygon points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

        regions.append({
            "type": LABEL_TEXT,
            "bbox": bbox,
            "text": text,
            "confidence": conf,
        })

    if verbose and regions:
        avg_conf = sum(r["confidence"] for r in regions) / len(regions)
        print(f"    [Paddle] {len(regions)} vùng text, avg confidence: {avg_conf:.2%}")

    return regions


def paddle_ocr_full_page(page_image_bytes: bytes, verbose: bool = True) -> str:
    """PaddleOCR toàn trang — trả về text thô gộp từ tất cả regions."""
    regions = paddle_layout_analysis(page_image_bytes, verbose)
    if not regions:
        return ""

    # Sort by y-position (top to bottom), then x-position (left to right)
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    lines = []
    for r in regions:
        lines.append(r["text"])

    return "\n".join(lines)


def paddle_detect_tables(page_image_bytes: bytes, verbose: bool = True) -> list[dict]:
    """
    Phát hiện vùng bảng bằng PaddleOCR PP-Structure layout analysis.

    Returns list of table regions with bounding boxes.
    """
    try:
        from paddleocr import PPStructure
    except ImportError:
        return []

    global _paddle_structure
    if "_paddle_structure" not in globals() or _paddle_structure is None:
        _paddle_structure = PPStructure(
            table=True,
            ocr=True,
            show_log=False,
            lang="vi",
        )

    from PIL import Image
    import numpy as np

    img = Image.open(io.BytesIO(page_image_bytes)).convert("RGB")
    img_np = np.array(img)

    result = _paddle_structure(img_np)

    tables = []
    texts = []
    for region in result:
        region_type = region.get("type", "").lower()
        bbox = region.get("bbox", [])
        if region_type == "table":
            tables.append({
                "type": LABEL_TABLE,
                "bbox": bbox,
                "html": region.get("res", {}).get("html", ""),
                "text": region.get("res", {}).get("text", ""),
            })
        elif region_type in ("text", "title"):
            text_content = ""
            res = region.get("res", [])
            if isinstance(res, list):
                text_content = "\n".join(item.get("text", "") for item in res if isinstance(item, dict))
            elif isinstance(res, dict):
                text_content = res.get("text", "")
            texts.append({
                "type": region_type,
                "bbox": bbox,
                "text": text_content,
            })

    if verbose:
        print(f"    [PPStructure] {len(tables)} bảng, {len(texts)} vùng text")

    return tables + texts


# ---------------------------------------------------------------------------
# Bước 2: Qwen2.5-VL — Trích xuất thông minh
# ---------------------------------------------------------------------------

def qwen_ocr_region(
    page_image_bytes: bytes,
    region_type: str = "text",
    bbox: Optional[list] = None,
    verbose: bool = True,
) -> str:
    """
    Dùng Qwen2.5-VL để OCR một vùng ảnh.
    Nếu bbox != None → crop vùng đó trước khi OCR.
    """
    import torch
    from PIL import Image

    model, processor = _get_qwen_model()

    img = Image.open(io.BytesIO(page_image_bytes)).convert("RGB")

    # Crop region nếu có bbox
    if bbox:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        img = img.crop((x1, y1, x2, y2))

    # Chọn prompt phù hợp
    prompt = OCR_PROMPT_TABLE_QWEN if region_type == LABEL_TABLE else OCR_PROMPT_QWEN

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input],
        images=[img],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=False,
        )

    # Decode only new tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text.strip()


def qwen_ocr_full_page(page_image_bytes: bytes, verbose: bool = True) -> str:
    """Qwen2.5-VL OCR toàn trang (không layout analysis)."""
    return qwen_ocr_region(page_image_bytes, region_type="text", bbox=None, verbose=verbose)


# ---------------------------------------------------------------------------
# Pipeline 2 bước: PaddleOCR → Qwen2.5-VL
# ---------------------------------------------------------------------------

def ocr_pipeline(
    pdf_path: Path,
    verbose: bool = True,
    use_qwen: bool = True,
) -> str:
    """
    Pipeline 2 bước:
      1. PaddleOCR: layout analysis + OCR thô
      2. Qwen2.5-VL: trích xuất thông minh cho vùng bảng / text phức tạp

    Nếu PaddleOCR đủ tốt (confidence cao, không có bảng) → dùng text thô.
    Nếu có bảng hoặc confidence thấp → gửi vùng đó qua Qwen.
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Thiếu PyMuPDF: pip install pymupdf") from exc

    from PIL import Image

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    all_texts: list[str] = []

    if verbose:
        print(f"  [Pipeline] {total_pages} trang | PaddleOCR → {'Qwen2.5-VL' if use_qwen else 'text thô'}")

    for page_num in range(total_pages):
        page = doc[page_num]

        if verbose:
            print(f"  [Trang {page_num + 1}/{total_pages}]")

        # Thử text layer trước
        text_layer = page.get_text("text").strip()
        if text_layer and len(text_layer) > 50:
            if verbose:
                print(f"    [Text layer] {len(text_layer)} chars — dùng text layer")
            all_texts.append(text_layer)
            continue

        # Render trang thành ảnh
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # Bước 1: PaddleOCR — phát hiện vùng
        try:
            structure_regions = paddle_detect_tables(img_bytes, verbose)
        except Exception as e:
            if verbose:
                print(f"    [WARN] PPStructure lỗi: {e} — fallback PaddleOCR cơ bản")
            structure_regions = []

        has_tables = any(r["type"] == LABEL_TABLE for r in structure_regions)

        if structure_regions and use_qwen and has_tables:
            # Có bảng → dùng Qwen cho vùng bảng, PaddleOCR cho text
            if verbose:
                print(f"    [Strategy] Phát hiện bảng → Qwen2.5-VL cho bảng")

            # Sort by y-position
            structure_regions.sort(key=lambda r: r["bbox"][1] if r.get("bbox") else 0)

            page_parts = []
            for region in structure_regions:
                if region["type"] == LABEL_TABLE:
                    # Bảng → Qwen OCR vùng bảng
                    try:
                        table_text = qwen_ocr_region(
                            img_bytes,
                            region_type=LABEL_TABLE,
                            bbox=region.get("bbox"),
                            verbose=verbose,
                        )
                        page_parts.append(table_text)
                    except Exception as e:
                        if verbose:
                            print(f"    [WARN] Qwen bảng lỗi: {e} — dùng text thô")
                        page_parts.append(region.get("text", ""))
                else:
                    # Text region → dùng text từ PPStructure
                    page_parts.append(region.get("text", ""))

            all_texts.append("\n\n".join(part for part in page_parts if part))

        elif use_qwen:
            # Không có bảng nhưng trang scan → Qwen OCR toàn trang
            if verbose:
                print(f"    [Strategy] Trang scan không bảng → Qwen2.5-VL toàn trang")
            try:
                page_text = qwen_ocr_full_page(img_bytes, verbose)
                all_texts.append(page_text)
            except Exception as e:
                if verbose:
                    print(f"    [WARN] Qwen lỗi: {e} — fallback PaddleOCR")
                try:
                    page_text = paddle_ocr_full_page(img_bytes, verbose)
                    all_texts.append(page_text)
                except Exception as e2:
                    if verbose:
                        print(f"    [WARN] PaddleOCR lỗi: {e2} — trang bị bỏ qua")
                    all_texts.append(f"[Trang {page_num + 1}: OCR không khả dụng — cài PaddleOCR hoặc Qwen]")
        else:
            # paddle-only mode
            try:
                page_text = paddle_ocr_full_page(img_bytes, verbose)
                all_texts.append(page_text)
            except Exception as e:
                if verbose:
                    print(f"    [WARN] PaddleOCR lỗi: {e}")
                all_texts.append(f"[Trang {page_num + 1}: PaddleOCR không khả dụng]")

    doc.close()
    combined = "\n\n".join(all_texts)
    if verbose:
        print(f"  [Pipeline] Tổng: {len(combined):,} ký tự")
    return combined


# ---------------------------------------------------------------------------
# Gemini OCR fallback (PDF upload trực tiếp)
# ---------------------------------------------------------------------------

def ocr_pdf_gemini(pdf_path: Path, verbose: bool = True) -> str:
    """OCR toàn bộ PDF bằng Gemini — upload file trực tiếp."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("[ERROR] Thiếu thư viện: pip install google-genai")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] Cần GEMINI_API_KEY hoặc GOOGLE_API_KEY trong biến môi trường")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    if verbose:
        print(f"  [Gemini] Uploading {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)...")

    for attempt in range(MAX_RETRIES):
        try:
            uploaded = client.files.upload(
                file=str(pdf_path),
                config=types.UploadFileConfig(mime_type="application/pdf"),
            )
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 2
                print(f"  [RETRY] Upload lỗi: {e} — thử lại sau {wait}s")
                time.sleep(wait)
            else:
                print(f"  [ERROR] Upload thất bại: {e}")
                return ""

    if verbose:
        print(f"  [Gemini] OCR đang xử lý...")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(parts=[
                        types.Part(file_data=types.FileData(
                            file_uri=uploaded.uri,
                            mime_type="application/pdf",
                        )),
                        types.Part(text=OCR_PROMPT_GEMINI),
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=32768,
                ),
            )
            text = response.text or ""
            if verbose:
                print(f"  [Gemini] Trích xuất được {len(text):,} ký tự")
            return text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 3
                print(f"  [RETRY] Generate lỗi: {e} — thử lại sau {wait}s")
                time.sleep(wait)
            else:
                print(f"  [ERROR] OCR thất bại: {e}")
                return ""
    return ""


# ---------------------------------------------------------------------------
# GPT-4o OCR fallback (convert PDF → ảnh → vision API)
# ---------------------------------------------------------------------------

def ocr_pdf_gpt4o(pdf_path: Path, verbose: bool = True) -> str:
    """OCR PDF bằng GPT-4o — convert từng trang thành ảnh."""
    try:
        import fitz
    except ImportError:
        print("[ERROR] Thiếu PyMuPDF: pip install pymupdf")
        sys.exit(1)

    try:
        import openai
    except ImportError:
        print("[ERROR] Thiếu: pip install openai pillow")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] Cần OPENAI_API_KEY trong biến môi trường")
        sys.exit(1)

    oclient = openai.OpenAI(api_key=api_key)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    all_texts: list[str] = []

    if verbose:
        print(f"  [GPT-4o] Xử lý {total_pages} trang...")

    for page_num in range(total_pages):
        page = doc[page_num]

        if verbose:
            print(f"  [GPT-4o] Trang {page_num + 1}/{total_pages}...", end="\r")

        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64_img = base64.b64encode(img_bytes).decode()

        for attempt in range(MAX_RETRIES):
            try:
                resp = oclient.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_img}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": OCR_PROMPT_GEMINI},
                        ],
                    }],
                    max_tokens=4096,
                    temperature=0.0,
                )
                page_text = resp.choices[0].message.content or ""
                all_texts.append(page_text)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt * 2
                    print(f"\n  [RETRY] Trang {page_num + 1} lỗi: {e} — thử lại sau {wait}s")
                    time.sleep(wait)
                else:
                    print(f"\n  [ERROR] Trang {page_num + 1} thất bại: {e}")
                    all_texts.append(f"[Trang {page_num + 1}: OCR thất bại]")

        time.sleep(REQUEST_DELAY)

    doc.close()
    combined = "\n\n".join(all_texts)
    if verbose:
        print(f"\n  [GPT-4o] Tổng: {len(combined):,} ký tự từ {total_pages} trang")
    return combined


# ---------------------------------------------------------------------------
# Hybrid: text layer → pipeline/fallback
# ---------------------------------------------------------------------------

def extract_with_llm_fallback(
    pdf_path: Path,
    model: str = "qwen",
    verbose: bool = True,
) -> str:
    """
    Thông minh: dùng PyMuPDF lấy text layer trước.
    Nếu > 30% trang scan → dùng pipeline OCR.
    """
    try:
        import fitz
    except ImportError:
        return _dispatch_ocr(pdf_path, model, verbose)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    text_pages = []
    scan_page_count = 0

    for page_num in range(total_pages):
        page_text = doc[page_num].get_text("text").strip()
        if page_text and len(page_text) > 30:
            text_pages.append(page_text)
        else:
            scan_page_count += 1
            text_pages.append(None)

    doc.close()

    scan_ratio = scan_page_count / max(total_pages, 1)

    if verbose:
        print(f"  [Analyze] {total_pages} trang | "
              f"{scan_page_count} trang scan ({scan_ratio:.0%})")

    if scan_ratio > 0.3:
        if verbose:
            print(f"  [Decide] Phần lớn là scan → dùng OCR pipeline")
        return _dispatch_ocr(pdf_path, model, verbose)

    result_parts = []
    for i, text in enumerate(text_pages):
        if text:
            result_parts.append(text)
        else:
            result_parts.append(f"[Trang {i + 1}: ảnh scan — chạy lại với --force-ocr để OCR]")

    combined = "\n\n".join(result_parts)
    if verbose:
        print(f"  [Text layer] Trích xuất {len(combined):,} ký tự")
    return combined


def _dispatch_ocr(pdf_path: Path, model: str, verbose: bool) -> str:
    """Dispatch OCR theo model được chọn."""
    if model == "qwen":
        return ocr_pipeline(pdf_path, verbose, use_qwen=True)
    elif model == "paddle-only":
        return ocr_pipeline(pdf_path, verbose, use_qwen=False)
    elif model == "gemini":
        return ocr_pdf_gemini(pdf_path, verbose)
    elif model == "gpt4o":
        return ocr_pdf_gpt4o(pdf_path, verbose)
    else:
        raise ValueError(f"Model không hỗ trợ: {model}. Dùng 'qwen', 'paddle-only', 'gemini', hoặc 'gpt4o'")


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize Unicode và cleanup phổ biến."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.replace("\u00a0", " ")
    return text.strip()


# ---------------------------------------------------------------------------
# Process single PDF
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    output_path: Path,
    model: str = "qwen",
    force_ocr: bool = False,
    verbose: bool = True,
) -> bool:
    """Xử lý 1 file PDF → .txt. Trả về True nếu thành công."""
    if output_path.exists():
        if verbose:
            print(f"  [SKIP] Đã tồn tại: {output_path.name}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  PDF: {pdf_path.name}")
        print(f"  OUT: {output_path.name}")
        print(f"  Model: {model.upper()}")

    if force_ocr:
        text = _dispatch_ocr(pdf_path, model, verbose)
    else:
        text = extract_with_llm_fallback(pdf_path, model, verbose)

    if not text or len(text.strip()) < 20:
        print(f"  [WARN] Không trích xuất được text (quá ngắn)")
        return False

    cleaned = clean_text(text)
    output_path.write_text(cleaned, encoding="utf-8")
    print(f"  ✓ Saved {len(cleaned):,} chars → {output_path.name}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR Pipeline: PaddleOCR + Qwen2.5-VL — Trích xuất text từ PDF scan tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Đường dẫn file PDF đơn lẻ")
    group.add_argument("--input-dir", type=Path, help="Thư mục chứa nhiều PDF")

    parser.add_argument(
        "--output", type=Path,
        help="Đường dẫn file .txt output (chỉ dùng với --input)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "Database" / "pdf_crawled",
        help="Thư mục output khi dùng --input-dir (mặc định: Database/pdf_crawled)",
    )
    parser.add_argument(
        "--model", choices=["qwen", "paddle-only", "gemini", "gpt4o"],
        default="qwen",
        help="OCR model (mặc định: qwen = PaddleOCR + Qwen2.5-VL)",
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="Bỏ qua text layer, luôn dùng OCR pipeline",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Bỏ qua file .txt đã tồn tại",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Ít output hơn",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # Single file mode
    if args.input:
        if not args.input.exists():
            print(f"[ERROR] File không tồn tại: {args.input}")
            sys.exit(1)
        output = args.output or args.input.with_suffix(".txt")
        success = process_pdf(
            args.input, output,
            model=args.model,
            force_ocr=args.force_ocr,
            verbose=verbose,
        )
        sys.exit(0 if success else 1)

    # Directory mode
    input_dir: Path = args.input_dir
    if not input_dir.exists():
        print(f"[ERROR] Thư mục không tồn tại: {input_dir}")
        sys.exit(1)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] Không tìm thấy file .pdf trong: {input_dir}")
        sys.exit(0)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  OCR Pipeline — {len(pdf_files)} file PDF")
    print(f"  Model : {args.model.upper()}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
        output_path = output_dir / (pdf_path.stem + ".txt")

        if args.skip_existing and output_path.exists():
            if verbose:
                print(f"  [SKIP] Đã có: {output_path.name}")
            skip_count += 1
            continue

        ok = process_pdf(
            pdf_path, output_path,
            model=args.model,
            force_ocr=args.force_ocr,
            verbose=verbose,
        )
        if ok:
            success_count += 1
        else:
            error_count += 1

        if i < len(pdf_files):
            time.sleep(REQUEST_DELAY)

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ")
    print(f"  ✓ Thành công : {success_count}")
    print(f"  ⟳ Bỏ qua    : {skip_count}")
    print(f"  ✗ Lỗi        : {error_count}")
    print(f"  Output       : {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
