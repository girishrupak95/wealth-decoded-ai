"""Prompt template loader."""

from pathlib import Path

from loguru import logger


class PromptLoader:
    def __init__(self, prompt_root: Path) -> None:
        self._prompt_root = prompt_root.resolve()
        self._cache: dict[Path, tuple[int, str]] = {}
        self._logger = logger.bind(component=self.__class__.__name__)

    def load(self, template_name: str) -> str:
        path = (self._prompt_root / template_name).resolve()
        path.relative_to(self._prompt_root)
        modified_at = path.stat().st_mtime_ns
        cached = self._cache.get(path)
        if cached is not None and cached[0] == modified_at:
            return cached[1]
        content = path.read_text(encoding="utf-8")
        self._cache[path] = (modified_at, content)
        self._logger.info("prompt_loaded", template=template_name)
        return content
