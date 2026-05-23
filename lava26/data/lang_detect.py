import re
from typing import Optional


_LANGID_AVAILABLE = False
try:
    import langid
    langid.set_languages(["ja", "vi"])
    _LANGID_AVAILABLE = True
except ImportError:
    pass

_JA_PATTERN = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")
_VI_PATTERN = re.compile(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]")


def detect_language(text: str, default: str = "ja") -> str:
    if not text or not text.strip():
        return default

    if _LANGID_AVAILABLE:
        try:
            lang, _ = langid.classify(text)
            if lang in ("ja", "vi"):
                return lang
        except Exception:
            pass

    # Character heuristic fallback
    ja_count = len(_JA_PATTERN.findall(text))
    vi_count = len(_VI_PATTERN.findall(text))

    if ja_count > vi_count:
        return "ja"
    elif vi_count > 0:
        return "vi"
    return default


def resolve_language(question_meta: dict, text: str, cfg_language: any) -> str:
    source = getattr(cfg_language, "source", "metadata_or_detect")
    default = getattr(cfg_language, "default", "ja")

    if source == "metadata_only":
        return question_meta.get("language") or default

    if source == "detect_only":
        return detect_language(text, default)

    # metadata_or_detect
    lang = question_meta.get("language")
    if lang and lang in ("ja", "vi"):
        return lang
    return detect_language(text, default)
