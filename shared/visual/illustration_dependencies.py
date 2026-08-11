"""Construct the production illustration collaborators from authoritative knowledge."""

from dataclasses import dataclass
from pathlib import Path

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolver,
)
from shared.visual.character_reference_selector import CharacterReferenceSelector
from shared.visual.character_resolver import CharacterResolver
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_prompt import IllustrationPromptBuilder


@dataclass(frozen=True)
class ProductionIllustrationDependencies:
    prompt_builder: IllustrationPromptBuilder
    composition_planner: CompositionPlanner
    reference_selector: CharacterReferenceSelector


def build_production_illustration_dependencies(
    knowledge_loader: KnowledgeLoader,
    repository_root: Path,
) -> ProductionIllustrationDependencies:
    """Build injected illustrated-generation dependencies without provider activity."""
    character_resolver = CharacterResolver(knowledge_loader)
    reference_resolver = CanonicalCharacterReferenceResolver(knowledge_loader, repository_root)
    return ProductionIllustrationDependencies(
        IllustrationPromptBuilder(knowledge_loader, character_resolver),
        CompositionPlanner(),
        CharacterReferenceSelector(reference_resolver),
    )
