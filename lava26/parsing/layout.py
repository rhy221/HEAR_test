import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = "region"  # figure | table | text | region


def _mineru_available() -> bool:
    return importlib.util.find_spec("magic_pdf") is not None


def _get_bboxes_mineru(pdf_path: Path, page_num: int, device: str) -> List[BBox]:
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.model_block_type import ModelBlockTypeEnum

    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(str(pdf_path))

    # MinerU analyzes entire document; we filter to requested page
    model_list = doc_analyze(pdf_bytes, ocr=False, device=device)

    bboxes = []
    if page_num < len(model_list):
        page_info = model_list[page_num]
        for block in page_info.get("layout_dets", []):
            cat = block.get("category_id", -1)
            poly = block.get("poly", [])
            if len(poly) >= 8:
                xs = poly[0::2]
                ys = poly[1::2]
                label = "figure" if cat == 3 else "table" if cat == 5 else "region"
                bboxes.append(BBox(min(xs), min(ys), max(xs), max(ys), label))
    return bboxes


def _get_bboxes_pymupdf(pdf_path: Path, page_num: int) -> List[BBox]:
    import fitz
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    drawings = page.get_drawings()
    doc.close()

    bboxes = []
    for d in drawings:
        rect = d.get("rect")
        if rect and rect.width > 30 and rect.height > 30:
            bboxes.append(BBox(rect.x0, rect.y0, rect.x1, rect.y1, "region"))

    # Merge overlapping bboxes heuristically
    bboxes = _merge_overlapping(bboxes, iou_threshold=0.3)
    return bboxes


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a.x1 - a.x0) * (a.y1 - a.y0)
    area_b = (b.x1 - b.x0) * (b.y1 - b.y0)
    return inter / (area_a + area_b - inter)


def _merge_overlapping(boxes: List[BBox], iou_threshold: float) -> List[BBox]:
    merged = []
    used = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        group = [b]
        for j, b2 in enumerate(boxes[i + 1:], i + 1):
            if not used[j] and _iou(b, b2) > iou_threshold:
                group.append(b2)
                used[j] = True
        x0 = min(g.x0 for g in group)
        y0 = min(g.y0 for g in group)
        x1 = max(g.x1 for g in group)
        y1 = max(g.y1 for g in group)
        merged.append(BBox(x0, y0, x1, y1, group[0].label))
    return merged


def get_layout_bboxes(pdf_path: Path, page_num: int, cfg: Any) -> List[BBox]:
    use_mineru = getattr(cfg.parsing, "use_mineru", True)
    fallback_on_error = getattr(cfg.parsing, "mineru_fallback_on_error", True)
    device = getattr(cfg.parsing, "mineru_device", "cuda")

    if use_mineru and _mineru_available():
        try:
            return _get_bboxes_mineru(pdf_path, page_num, device)
        except Exception as e:
            if fallback_on_error:
                logger.warning("MinerU failed, using PyMuPDF fallback: %s", e)
            else:
                raise
    elif use_mineru and not _mineru_available():
        logger.info("MinerU not installed, using PyMuPDF heuristic fallback")

    try:
        return _get_bboxes_pymupdf(pdf_path, page_num)
    except Exception as e:
        logger.warning("PyMuPDF bbox extraction failed: %s", e)
        return []
