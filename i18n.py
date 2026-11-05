from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QLocale, QTranslator
from PySide6.QtWidgets import QApplication

from translations.ui_ja import UI_TRANSLATIONS as JA_TRANSLATIONS


SUPPORTED_LANGUAGES = {"ja", "en"}


class DictTranslator(QTranslator):
    def __init__(self, translations: dict[str, str], parent=None):
        super().__init__(parent)
        self._translations = translations

    def translate(
        self,
        context: str,
        source_text: str,
        disambiguation: Optional[str] = None,
        n: int = -1,
    ) -> str:
        return self._translations.get(source_text, source_text)


def resolve_language(language: str) -> str:
    if language != "auto":
        return language
    locale_name = QLocale.system().name().lower()
    if locale_name.startswith("ja"):
        return "ja"
    return "en"


def install_translator(app: QApplication, language: str = "auto") -> Optional[QTranslator]:
    resolved_language = resolve_language(language)
    if resolved_language == "en":
        return None
    if resolved_language not in SUPPORTED_LANGUAGES:
        return None

    translator = DictTranslator(JA_TRANSLATIONS, app)
    app.installTranslator(translator)
    return translator
