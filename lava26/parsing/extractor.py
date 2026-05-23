import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    page_num: int        # 0-indexed (PyMuPDF native)
    page_num_1: int      # 1-indexed (submission format)
    text: str
    image_path: Optional[Path] = None
    width: int = 0
    height: int = 0


def extract_page_text(pdf_path: Path, page_num: int) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        text = page.get_text("text")
        doc.close()
        return text.strip()
    except Exception as e:
        logger.warning("PyMuPDF text extraction failed p%d: %s", page_num, e)
        return _fallback_text(pdf_path, page_num)


def _fallback_text(pdf_path: Path, page_num: int) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        if page_num < len(reader.pages):
            return reader.pages[page_num].extract_text() or ""
    except Exception as e:
        logger.warning("pypdf fallback also failed: %s", e)
    return ""


def render_page_image(
    pdf_path: Path,
    page_num: int,
    dpi: int = 150,
    max_long_side: int = 1568,
):
    try:
        from PIL import Image
        import fitz
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # Downsize to max_long_side if needed
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_long_side:
            scale = max_long_side / long_side
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        return img
    except Exception as e:
        logger.error("render_page_image failed p%d: %s", page_num, e)
        return None


def extract_all_pages(pdf_path: Path, cfg: Any, image_cache_dir: Optional[Path] = None) -> List[PageData]:
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)
        doc.close()
    except Exception:
        logger.warning("Cannot open %s, trying pypdf count", pdf_path)
        try:
            from pypdf import PdfReader
            n_pages = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            n_pages = 0

    pages = []
    dpi = getattr(cfg.parsing, "dpi", 150)
    max_long = getattr(cfg.parsing, "max_image_long_side", 1568)

    for i in range(n_pages):
        text = extract_page_text(pdf_path, i)
        img = render_page_image(pdf_path, i, dpi=dpi, max_long_side=max_long)

        img_path = None
        if img is not None and image_cache_dir is not None:
            image_cache_dir.mkdir(parents=True, exist_ok=True)
            img_path = image_cache_dir / f"page_{i:04d}.jpg"
            img.save(img_path, "JPEG", quality=90)

        pd = PageData(
            page_num=i,
            page_num_1=i + 1,
            text=text,
            image_path=img_path,
            width=img.width if img else 0,
            height=img.height if img else 0,
        )
        pages.append(pd)
        logger.debug("Extracted page %d/%d from %s", i + 1, n_pages, pdf_path.name)

    return pages
