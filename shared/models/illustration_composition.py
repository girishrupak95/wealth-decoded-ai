"""Dormant domain contracts for illustrated-scene content responsibility."""

import re
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from shared.models.base import BaseModel
from shared.models.characters import CHARACTER_ID_PATTERN

ELEMENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class IllustrationContentResponsibility(StrEnum):
    """Authority responsible for producing a visible scene element."""

    GENERATIVE = "generative"
    DETERMINISTIC = "deterministic"
    HYBRID = "hybrid"


class IllustrationElementType(StrEnum):
    """Supported semantic element categories in an illustrated frame."""

    CHARACTER = "character"
    ENVIRONMENT = "environment"
    FINANCIAL_OBJECT = "financial_object"
    VISUAL_METAPHOR = "visual_metaphor"
    DECORATIVE_OBJECT = "decorative_object"
    TITLE_TEXT = "title_text"
    LABEL_TEXT = "label_text"
    FINANCIAL_VALUE = "financial_value"
    FINANCIAL_GRAPHIC = "financial_graphic"


class IllustrationElementSpec(BaseModel):
    """One semantic frame element and its production responsibility."""

    element_id: str
    element_type: IllustrationElementType
    responsibility: IllustrationContentResponsibility
    description: str
    character_id: str | None = None
    text_content: str | None = None
    financial_value: str | None = None

    @field_validator("element_id")
    @classmethod
    def validate_element_id(cls, value: str) -> str:
        """Trim and validate a deterministic lowercase element identifier."""
        normalized = value.strip()
        if not normalized or ELEMENT_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("element_id must use a lowercase underscore identifier")
        return normalized

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        """Trim required editorial description text."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("description must not be blank")
        return normalized

    @field_validator("character_id")
    @classmethod
    def validate_character_id(cls, value: str | None) -> str | None:
        """Validate syntax without consulting the canonical catalog."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or CHARACTER_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("character_id must use an uppercase underscore-separated identifier")
        return normalized

    @field_validator("text_content", "financial_value")
    @classmethod
    def normalize_optional_content(cls, value: str | None) -> str | None:
        """Trim optional deterministic content and normalize blanks to absent."""
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def validate_element_semantics(self) -> "IllustrationElementSpec":
        """Enforce the authority boundary for each semantic element type."""
        generative_or_hybrid = {
            IllustrationContentResponsibility.GENERATIVE,
            IllustrationContentResponsibility.HYBRID,
        }
        element_type = self.element_type
        if element_type == IllustrationElementType.CHARACTER:
            self._require_responsibility(generative_or_hybrid)
            if self.character_id is None:
                raise ValueError("character elements require character_id")
            self._reject_content(text=True, financial=True)
        elif element_type in {
            IllustrationElementType.TITLE_TEXT,
            IllustrationElementType.LABEL_TEXT,
        }:
            self._require_responsibility({IllustrationContentResponsibility.DETERMINISTIC})
            if self.text_content is None:
                raise ValueError("text elements require text_content")
            self._reject_character_and_financial()
        elif element_type == IllustrationElementType.FINANCIAL_VALUE:
            self._require_responsibility({IllustrationContentResponsibility.DETERMINISTIC})
            if self.financial_value is None:
                raise ValueError("financial value elements require financial_value")
            if self.character_id is not None or self.text_content is not None:
                raise ValueError("financial value elements reject character_id and text_content")
        elif element_type == IllustrationElementType.FINANCIAL_GRAPHIC:
            self._require_responsibility({IllustrationContentResponsibility.DETERMINISTIC})
            self._reject_content(character=True, text=True, financial=True)
        elif element_type == IllustrationElementType.ENVIRONMENT:
            self._require_responsibility(generative_or_hybrid)
            self._reject_content(character=True, text=True, financial=True)
        elif element_type == IllustrationElementType.FINANCIAL_OBJECT:
            self._reject_content(character=True, text=True, financial=True)
        elif element_type == IllustrationElementType.VISUAL_METAPHOR:
            self._require_responsibility(generative_or_hybrid)
            self._reject_content(character=True, text=True, financial=True)
        elif element_type == IllustrationElementType.DECORATIVE_OBJECT:
            self._require_responsibility({IllustrationContentResponsibility.GENERATIVE})
            self._reject_content(character=True, text=True, financial=True)
        return self

    def _require_responsibility(self, allowed: set[IllustrationContentResponsibility]) -> None:
        if self.responsibility not in allowed:
            raise ValueError(
                f"{self.element_type.value} does not support "
                f"{self.responsibility.value} responsibility"
            )

    def _reject_character_and_financial(self) -> None:
        if self.character_id is not None or self.financial_value is not None:
            raise ValueError("text elements reject character_id and financial_value")

    def _reject_content(
        self,
        *,
        character: bool = False,
        text: bool = False,
        financial: bool = False,
    ) -> None:
        if character and self.character_id is not None:
            raise ValueError(f"{self.element_type.value} rejects character_id")
        if text and self.text_content is not None:
            raise ValueError(f"{self.element_type.value} rejects text_content")
        if financial and self.financial_value is not None:
            raise ValueError(f"{self.element_type.value} rejects financial_value")


class IllustrationCompositionContract(BaseModel):
    """Ordered responsibility contract for one future hybrid scene."""

    contract_version: str = "1.0"
    scene_id: str
    elements: list[IllustrationElementSpec] = Field(min_length=1)

    @field_validator("contract_version")
    @classmethod
    def validate_contract_version(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError("contract_version must equal 1.0")
        return value

    @field_validator("scene_id")
    @classmethod
    def validate_scene_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("scene_id must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_unique_element_ids(self) -> "IllustrationCompositionContract":
        element_ids = [element.element_id for element in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("element_id values must be unique")
        return self

    @property
    def generative_elements(self) -> list[IllustrationElementSpec]:
        return self._with_responsibility(IllustrationContentResponsibility.GENERATIVE)

    @property
    def deterministic_elements(self) -> list[IllustrationElementSpec]:
        return self._with_responsibility(IllustrationContentResponsibility.DETERMINISTIC)

    @property
    def hybrid_elements(self) -> list[IllustrationElementSpec]:
        return self._with_responsibility(IllustrationContentResponsibility.HYBRID)

    def _with_responsibility(
        self, responsibility: IllustrationContentResponsibility
    ) -> list[IllustrationElementSpec]:
        return [element for element in self.elements if element.responsibility == responsibility]
