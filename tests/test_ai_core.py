import json
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from pydantic import BaseModel

from shared.ai.base_agent import AgentRequest, BaseAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.exceptions.ai import OutputValidationError


class Payload(BaseModel):
    value: str


class AgentPayload(BaseModel):
    value: str


class RecordingLLMClient(LLMClient):
    def __init__(self, output: str) -> None:
        super().__init__()
        self.output = output
        self.request: LLMRequest | None = None

    async def generate(self, request: LLMRequest) -> str:
        self.request = request
        return self.output

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class SchemaAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "schema-agent"

    @property
    def output_schema(self) -> type[BaseModel]:
        return AgentPayload


def build_schema_agent(tmp_path: Path, output: str) -> tuple[SchemaAgent, RecordingLLMClient]:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "user.md").write_text("User prompt: $value", encoding="utf-8")
    (prompts / "system.md").write_text("System prompt: $value", encoding="utf-8")
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "rule.md").write_text("Knowledge remains unchanged", encoding="utf-8")
    llm_client = RecordingLLMClient(output)
    return (
        SchemaAgent(
            llm_client=llm_client,
            prompt_loader=PromptLoader(prompts),
            knowledge_loader=KnowledgeLoader(knowledge),
            output_validator=OutputValidator(),
        ),
        llm_client,
    )


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


@pytest.mark.asyncio
async def test_base_agent_appends_generated_schema_without_mutating_request_data(
    tmp_path: Path,
) -> None:
    agent, llm_client = build_schema_agent(tmp_path, '{"value": "valid"}')
    context: dict[str, Any] = {"value": "unchanged"}

    await agent.execute(
        AgentRequest(prompt_name="user.md", system_prompt_name="system.md", context=context)
    )

    assert llm_client.request is not None
    assert llm_client.request.template == "User prompt: unchanged"
    assert llm_client.request.context == context
    assert llm_client.request.knowledge == {"rule.md": "Knowledge remains unchanged"}
    system_template = llm_client.request.system_template
    assert system_template is not None
    assert system_template.startswith("System prompt: unchanged\n\n")
    assert json.dumps(AgentPayload.model_json_schema(), sort_keys=True) in system_template
    assert "Do not include fields not present in the schema." in system_template
    assert "Do not include Markdown fences." in system_template


@pytest.mark.asyncio
async def test_base_agent_creates_schema_system_instruction_without_system_prompt(
    tmp_path: Path,
) -> None:
    agent, llm_client = build_schema_agent(tmp_path, '{"value": "valid"}')

    await agent.execute(AgentRequest(prompt_name="user.md", context={"value": "unchanged"}))

    assert llm_client.request is not None
    system_template = llm_client.request.system_template
    assert system_template is not None
    assert system_template.startswith("Return exactly one valid JSON object.")
    assert llm_client.request.template == "User prompt: unchanged"


@pytest.mark.asyncio
async def test_base_agent_rejects_extra_output_without_logging_raw_response(tmp_path: Path) -> None:
    raw_output = '{"value": "valid", "invented": "private output"}'
    agent, _ = build_schema_agent(tmp_path, raw_output)
    log_messages: list[str] = []
    handler_id = logger.add(log_messages.append, format="{message}")
    try:
        with pytest.raises(OutputValidationError):
            await agent.execute(AgentRequest(prompt_name="user.md", context={"value": "unchanged"}))
    finally:
        logger.remove(handler_id)

    messages = "".join(log_messages)
    assert "agent_output_validation_failed" in messages
    assert "private output" not in messages


@pytest.mark.asyncio
async def test_debug_raw_output_is_not_written_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WEALTH_DEBUG_SAVE_RAW_LLM", raising=False)
    agent, _ = build_schema_agent(tmp_path, '{"value": "valid"}')

    await agent.execute(AgentRequest(prompt_name="user.md", context={"value": "unchanged"}))

    assert not (tmp_path / "generated" / "debug" / "schema_agent-raw.json").exists()


@pytest.mark.asyncio
async def test_debug_raw_output_pretty_prints_json_and_uses_agent_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEALTH_DEBUG_SAVE_RAW_LLM", "1")
    agent, _ = build_schema_agent(tmp_path, '{"value":"valid"}')

    await agent.execute(AgentRequest(prompt_name="user.md", context={"value": "unchanged"}))

    path = tmp_path / "generated" / "debug" / "schema_agent-raw.json"
    assert agent._debug_output_path() == path
    assert path.read_text(encoding="utf-8") == '{\n  "value": "valid"\n}'


@pytest.mark.asyncio
async def test_debug_raw_output_preserves_invalid_json_and_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEALTH_DEBUG_SAVE_RAW_LLM", "1")
    raw_output = "not valid json"
    agent, _ = build_schema_agent(tmp_path, raw_output)

    with pytest.raises(OutputValidationError):
        await agent.execute(AgentRequest(prompt_name="user.md", context={"value": "unchanged"}))

    assert (tmp_path / "generated" / "debug" / "schema_agent-raw.json").read_text(
        encoding="utf-8"
    ) == raw_output


@pytest.mark.asyncio
async def test_debug_raw_output_write_failure_does_not_interrupt_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEALTH_DEBUG_SAVE_RAW_LLM", "1")
    agent, _ = build_schema_agent(tmp_path, '{"value": "valid"}')
    log_messages: list[str] = []
    handler_id = logger.add(log_messages.append, format="{message}")
    original_write_text = Path.write_text

    def failing_write_text(self: Path, data: str, encoding: str) -> int:
        if self.name == "schema_agent-raw.json":
            raise OSError("unavailable")
        return original_write_text(self, data, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", failing_write_text)
    try:
        execution = await agent.execute(
            AgentRequest(prompt_name="user.md", context={"value": "unchanged"})
        )
    finally:
        logger.remove(handler_id)

    assert execution.output == {"value": "valid"}
    assert "debug_output_save_failed" in "".join(log_messages)


@pytest.mark.asyncio
async def test_debug_raw_output_overwrites_previous_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEALTH_DEBUG_SAVE_RAW_LLM", "1")
    agent, llm_client = build_schema_agent(tmp_path, '{"value": "first"}')
    request = AgentRequest(prompt_name="user.md", context={"value": "unchanged"})

    await agent.execute(request)
    llm_client.output = '{"value": "second"}'
    await agent.execute(request)

    content = (tmp_path / "generated" / "debug" / "schema_agent-raw.json").read_text(
        encoding="utf-8"
    )
    assert '"second"' in content and '"first"' not in content
