"""Deterministic resolution of authoritative illustrated characters."""

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.characters import CharacterCatalog, CharacterDefinition

CHARACTER_CATALOG_KEY = "style/characters.json"


class CharacterResolutionError(ValueError):
    """Raised when an exact canonical character cannot be resolved."""


class CharacterResolver:
    """Load and resolve canonical characters without provider knowledge."""

    def __init__(self, knowledge_loader: KnowledgeLoader) -> None:
        payload = knowledge_loader.load_all().get(CHARACTER_CATALOG_KEY)
        if payload is None:
            raise CharacterResolutionError("Character resolution failed: catalog unavailable.")
        try:
            self._catalog = CharacterCatalog.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise CharacterResolutionError(
                "Character resolution failed: catalog is invalid."
            ) from error
        self._characters = {
            character.character_id: character for character in self._catalog.characters
        }

    @property
    def catalog(self) -> CharacterCatalog:
        """Return the validated authoritative character catalog."""
        return self._catalog

    def resolve(self, character_id: str) -> CharacterDefinition:
        """Resolve one exact canonical ID without normalization or guessing."""
        try:
            return self._characters[character_id]
        except KeyError as error:
            raise CharacterResolutionError(
                f"Character resolution failed: unknown character ID '{character_id}'."
            ) from error

    def resolve_many(self, character_ids: list[str]) -> list[CharacterDefinition]:
        """Resolve unique requested IDs in first-seen order or fail completely."""
        unique_ids = list(dict.fromkeys(character_ids))
        return [self.resolve(character_id) for character_id in unique_ids]
