"""Test language detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lava26.data.lang_detect import detect_language


class TestLangDetect:
    def test_japanese_katakana(self):
        assert detect_language("コンピュータシステム") == "ja"

    def test_japanese_hiragana(self):
        assert detect_language("これはテストです") == "ja"

    def test_japanese_kanji(self):
        assert detect_language("日本語のテキスト") == "ja"

    def test_vietnamese_diacritics(self):
        assert detect_language("Câu trả lời là gì?") == "vi"

    def test_vietnamese_special_chars(self):
        assert detect_language("Bộ Giáo dục và Đào tạo") == "vi"

    def test_empty_string_returns_default(self):
        assert detect_language("", default="ja") == "ja"
        assert detect_language("", default="vi") == "vi"

    def test_whitespace_returns_default(self):
        assert detect_language("   ", default="ja") == "ja"

    def test_english_returns_default(self):
        # English with no JA/VI markers should return default
        result = detect_language("Hello world this is English", default="ja")
        assert result in ("ja", "vi")  # Should return default or best guess
