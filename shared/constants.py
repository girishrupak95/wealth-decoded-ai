"""Shared constants used by agent infrastructure."""

from enum import StrEnum


class ExecutionStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


MILLISECONDS_PER_SECOND = 1_000
DEFAULT_TOPIC_CATEGORY = "Personal Finance"
GENERATED_DIRECTORY_NAME = "generated"
JSON_FILE_SUFFIX = ".json"
MARKDOWN_FILE_SUFFIX = ".md"
TOPIC_AGENT_NAME = "topic-agent"
CONCEPT_AGENT_NAME = "concept-agent"
RESEARCH_AGENT_NAME = "research-agent"
TOPIC_AGENT_SYSTEM_PROMPT = "topic_agent/system.md"
TOPIC_AGENT_USER_PROMPT = "topic_agent/user.md"
CONCEPT_AGENT_SYSTEM_PROMPT = "concept_agent/system.md"
CONCEPT_AGENT_USER_PROMPT = "concept_agent/user.md"
RESEARCH_AGENT_SYSTEM_PROMPT = "research_agent/system.md"
RESEARCH_AGENT_USER_PROMPT = "research_agent/user.md"
TOPIC_CANDIDATES_KEY = "topics"
TOPICS_DIRECTORY_NAME = "topics"
CONCEPTS_DIRECTORY_NAME = "concepts"
RESEARCH_DIRECTORY_NAME = "research"
SCRIPT_AGENT_NAME = "script-agent"
SCRIPT_AGENT_SYSTEM_PROMPT = "script_agent/system.md"
SCRIPT_AGENT_USER_PROMPT = "script_agent/user.md"
SCRIPTS_DIRECTORY_NAME = "scripts"
DEFAULT_SCRIPT_WORDS_PER_MINUTE = 145
DEFAULT_SCRIPT_VISUAL_PAUSE_SECONDS = 0
DEFAULT_SCRIPT_MIN_WORDS = 600
DEFAULT_SCRIPT_MAX_WORDS = 900
DEFAULT_SCRIPT_MAX_RETRIES = 1
SECTION_DURATION_MISMATCH_SECONDS = 30
