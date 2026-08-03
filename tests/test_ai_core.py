import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.exceptions.ai import OutputValidationError


class Payload(BaseModel):
    value: str


def test_prompt_loader_reads_templates(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("template", encoding="utf-8")

    assert PromptLoader(tmp_path).load("prompt.md") == "template"


def test_knowledge_loader_exposes_json_and_markdown(tmp_path: Path) -> None:
    (tmp_path / "brand.md").write_text("voice", encoding="utf-8")
    (tmp_path / "brand.json").write_text(json.dumps({"name": "Wealth"}), encoding="utf-8")

    assert KnowledgeLoader(tmp_path).load_all() == {
        "brand.md": "voice",
        "brand.json": {"name": "Wealth"},
    }


def test_output_validator_validates_pydantic_schema() -> None:
    assert OutputValidator().validate('{"value": "valid"}', Payload) == {"value": "valid"}


def test_output_validator_rejects_invalid_json() -> None:
    with pytest.raises(OutputValidationError):
        OutputValidator().validate("invalid", Payload)
