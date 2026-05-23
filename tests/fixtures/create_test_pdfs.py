#!/usr/bin/env python3
"""Generate synthetic test PDFs for smoke tests."""
from pathlib import Path


def create_pdf(path: Path, content: str) -> None:
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), content, fontsize=12)
        doc.save(str(path))
        doc.close()
    except ImportError:
        # Fallback: create minimal valid PDF manually
        _create_minimal_pdf(path, content)


def _create_minimal_pdf(path: Path, content: str) -> None:
    content_stream = f"BT /F1 12 Tf 50 700 Td ({content[:200]}) Tj ET"
    content_bytes = content_stream.encode()
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 << /Type /Font "
        b"/Subtype /Type1 /BaseFont /Helvetica >> >> >> >> endobj\n"
        b"4 0 obj << /Length " + str(len(content_bytes)).encode() + b" >>\n"
        b"stream\n" + content_bytes + b"\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000058 00000 n\n0000000115 00000 n\n0000000266 00000 n\n"
        b"trailer << /Size 5 /Root 1 0 R >>\nstartxref\n"
        + str(400 + len(content_bytes)).encode() + b"\n%%EOF\n"
    )
    path.write_bytes(pdf)


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent
    pdf_dir = fixtures_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    create_pdf(
        pdf_dir / "test_doc_ja.pdf",
        "テスト文書\n作成日：2026年1月15日\n売上合計：1,234,567円\n項目：商品A, 商品B, 商品C"
    )
    create_pdf(
        pdf_dir / "test_doc_vi.pdf",
        "Tài liệu thử nghiệm\nNgày tạo: 15/01/2026\nCác phòng ban: Phòng Kế toán, Phòng Kỹ thuật, Phòng Nhân sự"
    )
    print(f"Created test PDFs in {pdf_dir}")
