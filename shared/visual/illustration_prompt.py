"""Deterministic provider-neutral prompt construction for illustrations."""

from dataclasses import dataclass
from typing import Any

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.characters import CharacterDefinition
from shared.models.composition_plan import IllustrationCompositionPlan
from shared.models.illustration import IllustrationPaletteEmphasis, IllustrationSpec
from shared.visual.character_resolver import CharacterResolver

ILLUSTRATION_STYLE_PROFILE_KEY = "style/illustration.json"


@dataclass(frozen=True)
class IllustrationPromptContext:
    """Limited editorial context that may inform a still illustration."""

    narration_excerpt: str | None = None
    on_screen_text: tuple[str, ...] = ()


@dataclass(frozen=True)
class IllustrationPromptResult:
    """Stable provider-neutral output from illustration prompt construction."""

    prompt: str
    negative_prompt: str | None
    style_profile_version: str
    spec_version: str


class IllustrationPromptBuilder:
    """Combine illustration intent with the authoritative Wealth Decoded style."""

    def __init__(
        self,
        knowledge_loader: KnowledgeLoader,
        character_resolver: CharacterResolver | None = None,
    ) -> None:
        profile = knowledge_loader.load_all().get(ILLUSTRATION_STYLE_PROFILE_KEY)
        if not isinstance(profile, dict):
            raise ValueError("The authoritative illustration style profile is unavailable.")
        self._profile = profile
        self._knowledge_loader = knowledge_loader
        self._character_resolver = character_resolver
        self._validate_profile()

    def build(
        self,
        spec: IllustrationSpec,
        *,
        scene_context: IllustrationPromptContext | None = None,
        composition_plan: IllustrationCompositionPlan | None = None,
    ) -> IllustrationPromptResult:
        """Build the same clean prompt for every equivalent set of editorial inputs."""
        context = scene_context or IllustrationPromptContext()
        lines = [
            self._visual_identity(),
            f"Scene purpose: {spec.purpose}",
            f"Scene description: {spec.description}",
            f"Scene type: {spec.scene_type.value}",
            self._composition(spec),
            f"Framing: {spec.composition.framing.value}",
            self._guidance("Character style", "character_language"),
        ]
        if spec.character_ids:
            resolver = self._character_resolver or CharacterResolver(self._knowledge_loader)
            characters = resolver.resolve_many(spec.character_ids)
            lines.extend(
                self._character_identity(index, character)
                for index, character in enumerate(characters, start=1)
            )
        if spec.environment:
            lines.append(f"Environment: {spec.environment}")
        if spec.key_objects:
            lines.append(f"Key financial or visual objects: {self._join(spec.key_objects)}")
        if spec.visual_metaphor:
            lines.append(f"Visual metaphor: {spec.visual_metaphor}")
        lines.append(self._guidance("Visual-metaphor guidance", "visual_metaphor_language"))
        if spec.mood:
            lines.append(f"Mood: {spec.mood}")
        if spec.palette_emphasis:
            lines.append(self._palette_emphasis(spec.palette_emphasis))

        narration = self._optional_text(context.narration_excerpt)
        if narration:
            lines.append(f"Relevant narration context: {narration}")
        on_screen_text = self._normalized_items(context.on_screen_text)
        if on_screen_text:
            lines.append(
                "Allowed embedded text only: "
                f"{self._join(on_screen_text)}. Do not add labels, paragraphs, or numbers."
            )
        else:
            lines.append(
                "Embedded text: Prefer no embedded text unless intrinsically required by the "
                "scene description; do not invent labels, paragraphs, or numbers."
            )
        lines.append(self._guidance("Text policy", "text_policy"))

        lines.extend(
            [
                self._drawing_language(),
                self._guidance("Financial-object guidance", "financial_object_language"),
                self._guidance("Composition guidance", "composition"),
                self._financial_accuracy(spec),
            ]
        )
        if composition_plan is not None:
            lines.append(self._composition_plan(composition_plan))
        prompt = "\n".join(line for line in lines if line.strip())
        negative_prompt = "; ".join(
            self._deduplicate(
                [*self._string_list("negative_constraints"), *spec.prohibited_elements]
            )
        )
        return IllustrationPromptResult(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            style_profile_version=str(self._profile["profile_version"]),
            spec_version=spec.spec_version,
        )

    @staticmethod
    def _composition_plan(plan: IllustrationCompositionPlan) -> str:
        elements = " | ".join(
            (
                f"element {element.element_id}; type {element.element_type}; "
                f"importance {element.importance}; placement {element.placement.value}; "
                f"depth {element.depth.value}; description {element.description}"
            )
            for element in plan.elements
        )
        return (
            f"Editorial composition plan: Template {plan.template_name}; "
            f"Primary focus {plan.primary_focus}; Camera {plan.camera.value}; "
            f"Lighting {plan.lighting.value}; Negative space {plan.negative_space}; "
            f"Text-safe region {plan.text_safe_region}; "
            f"Gold accent strategy {plan.gold_accent_strategy}; Elements {elements}"
        )

    def _character_identity(self, index: int, character: CharacterDefinition) -> str:
        """Render canonical identity without inferring scene-specific behavior."""
        parts = [
            f"Canonical recurring character {index}",
            f"character ID {character.character_id}",
            f"display name {character.display_name}",
            f"role {character.role.value}",
            f"canonical visual identity {character.visual_identity}",
        ]
        if character.wardrobe:
            parts.append(f"wardrobe {self._join(character.wardrobe)}")
        if character.signature_features:
            parts.append(f"signature features {self._join(character.signature_features)}")
        if character.default_expression:
            parts.append(
                "default expression baseline "
                f"{character.default_expression} The scene description remains authoritative "
                "for the current scene's action and emotion"
            )
        if character.palette_emphasis:
            parts.append(f"character identity palette {self._join(character.palette_emphasis)}")
        if character.prohibited_changes:
            parts.append(
                "character continuity constraints; do not casually alter "
                f"{self._join(character.prohibited_changes)}"
            )
        return ": ".join((parts[0], ". ".join(parts[1:])))

    def _visual_identity(self) -> str:
        style = self._mapping("style_character")
        palette = self._mapping("palette")
        attributes = self._strings(style.get("attributes"), "style_character.attributes")
        colors = [
            f"paper {self._color(palette, 'paper_background')}",
            f"primary ink {self._color(palette, 'primary_ink')}",
            f"selective gold accent {self._color(palette, 'accent')}",
            f"restrained secondary {self._color(palette, 'secondary')}",
        ]
        return (
            f"Wealth Decoded visual identity: {self._profile['direction']} "
            f"Style: {self._join(attributes)}. Palette: {self._join(colors)}; navy ink and "
            "paper dominate, with gold used selectively."
        )

    @staticmethod
    def _composition(spec: IllustrationSpec) -> str:
        composition = spec.composition
        details = [
            f"focal position {composition.focal_position}",
            f"background complexity {composition.background_complexity}",
        ]
        if composition.focal_subject:
            details.insert(0, f"focal subject {composition.focal_subject}")
        return f"Composition: {IllustrationPromptBuilder._join(details)}"

    def _palette_emphasis(self, emphasis: list[IllustrationPaletteEmphasis]) -> str:
        palette = self._mapping("palette")
        profile_keys = {
            IllustrationPaletteEmphasis.GOLD: "accent",
            IllustrationPaletteEmphasis.MUTED: "secondary",
        }
        resolved = [
            f"{item.value} {self._color(palette, profile_keys.get(item, item.value))}"
            for item in emphasis
        ]
        return (
            f"Palette emphasis: {self._join(resolved)}. Keep the global paper-and-ink identity "
            "dominant and gold selective."
        )

    def _drawing_language(self) -> str:
        drawing = self._mapping("drawing_language")
        linework = self._nested_strings(drawing, "linework", "preferred")
        shading = self._nested_strings(drawing, "shading", "preferred")
        detail = self._strings(drawing.get("detail_level"), "drawing_language.detail_level")
        return (
            f"Drawing language: linework—{self._join(linework)}; "
            f"shading—{self._join(shading)}; detail—{self._join(detail)}"
        )

    def _guidance(self, label: str, key: str) -> str:
        values = self._strings(self._mapping(key).get("rules"), f"{key}.rules")
        return f"{label}: {self._join(values)}"

    def _financial_accuracy(self, spec: IllustrationSpec) -> str:
        rules = self._strings(
            self._mapping("financial_accuracy").get("rules"), "financial_accuracy.rules"
        )
        if spec.scene_type.value == "data":
            rules = [
                *rules,
                "Treat this as conceptual editorial data visualization only; do not create a "
                "precise numerical chart or invent chart values.",
            ]
        return f"Financial accuracy: {self._join(rules)}"

    def _validate_profile(self) -> None:
        required = {
            "profile_version",
            "direction",
            "style_character",
            "palette",
            "drawing_language",
            "character_language",
            "financial_object_language",
            "visual_metaphor_language",
            "composition",
            "text_policy",
            "financial_accuracy",
            "negative_constraints",
        }
        if not required <= self._profile.keys():
            raise ValueError("The authoritative illustration style profile is incomplete.")

    def _mapping(self, key: str) -> dict[str, Any]:
        value = self._profile.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Illustration style section '{key}' is invalid.")
        return value

    def _string_list(self, key: str) -> list[str]:
        return self._strings(self._profile.get(key), key)

    @staticmethod
    def _strings(value: object, key: str) -> list[str]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(f"Illustration style value '{key}' is invalid.")
        return [item.strip() for item in value]

    @staticmethod
    def _nested_strings(mapping: dict[str, Any], section: str, key: str) -> list[str]:
        nested = mapping.get(section)
        if not isinstance(nested, dict):
            raise ValueError(f"Illustration style section '{section}' is invalid.")
        return IllustrationPromptBuilder._strings(nested.get(key), f"{section}.{key}")

    @staticmethod
    def _color(palette: dict[str, Any], key: str) -> str:
        entry = palette.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("hex"), str):
            raise ValueError(f"Illustration palette color '{key}' is invalid.")
        return str(entry["hex"])

    @staticmethod
    def _optional_text(value: str | None) -> str | None:
        normalized = value.strip() if value else ""
        return normalized or None

    @staticmethod
    def _normalized_items(values: tuple[str, ...]) -> list[str]:
        return IllustrationPromptBuilder._deduplicate(
            [item.strip() for item in values if item.strip()]
        )

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(value.split())
            identity = normalized.casefold()
            if normalized and identity not in seen:
                seen.add(identity)
                result.append(normalized)
        return result

    @staticmethod
    def _join(values: list[str]) -> str:
        return "; ".join(values)
