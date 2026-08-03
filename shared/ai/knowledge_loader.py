"""Machine-readable knowledge loader."""

import json
from pathlib import Path
from typing import Any


class KnowledgeLoader:
    def __init__(self, knowledge_root: Path) -> None:
        self._knowledge_root = knowledge_root.resolve()
        self._cache: dict[str, Any] | None = None

    def load_all(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache.copy()
        content: dict[str, Any] = {}
        for path in self._knowledge_root.rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".md"}:
                continue
            value = path.read_text(encoding="utf-8")
            content[path.relative_to(self._knowledge_root).as_posix()] = (
                json.loads(value) if path.suffix == ".json" else value
            )
        self._cache = content
        return content.copy()
