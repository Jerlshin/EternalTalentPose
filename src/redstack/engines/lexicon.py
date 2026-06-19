

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, Field

from redstack.config.schema import CompiledLexicon
from redstack.domain.enums import EvidenceKind
from redstack.domain.ids import SkillName, UnitScore
from redstack.domain.provenance import EvidenceRef
from redstack.features.view import bounded_log_scale, clamp_unit, make_evidence


@final
class LexiconMatch(BaseModel):
    """Result of symbolic matching against the compiled lexicon.

    ``concept_hits`` maps concept id → number of corroborating surface hits;
    ``mention_count`` is the total term/phrase hit count; ``corroboration`` is the
    bounded ``[0, 1]`` aggregate consumed by competency/credibility fusion.
    ``evidence`` is non-empty whenever any hit fired, so a cited corroboration is
    always backed by a real surface mention.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True, validate_default=True
    )

    concept_hits: tuple[tuple[str, int], ...]
    mention_count: int = Field(ge=0)
    corroboration: UnitScore = Field(ge=0.0, le=1.0)
    evidence: tuple[EvidenceRef, ...]


@final
class LexiconEngine(BaseModel):
    """Stateless, pure symbolic matcher over normalized free-text."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    lexicon: CompiledLexicon
    # A mention equal to this count saturates corroboration to ~1.0.
    mention_saturation: float = Field(default=4.0, gt=0.0)

    # ------------------------------------------------------------------ public
    def match_text(self, concept: str, normalized_text: str, *, path: str) -> LexiconMatch:
        """Count term/phrase hits for one concept inside one normalized text span."""
        return self._match_tokens(concept, (normalized_text,), path=path)

    def corroborate_skill(
        self, skill: SkillName, descriptions: tuple[str, ...], *, path: str
    ) -> UnitScore:
        """Corroboration score for a single skill across all career descriptions.

        Returns ``[0, 1]``: zero when the skill name never appears in any
        description (the stuffer signature), saturating as mentions accumulate.
        """
        concept = str(skill)
        match = self._match_tokens(concept, descriptions, path=path)
        return match.corroboration

    # --------------------------------------------------------------- internals
    def _surface_forms(self, concept: str) -> tuple[frozenset[str], tuple[str, ...]]:
        entry = self.lexicon.concepts.get(concept)
        if entry is None:
            # Unknown concept maps to itself (Normalization never drops tokens).
            return frozenset((concept.lower(),)), ()
        return entry.terms, entry.phrases

    def _match_tokens(
        self, concept: str, texts: tuple[str, ...], *, path: str
    ) -> LexiconMatch:
        terms, phrases = self._surface_forms(concept)
        hits = 0
        evidence: list[EvidenceRef] = []
        for offset, raw_text in enumerate(texts):
            text = raw_text.lower()
            if not text:
                continue
            tokens = frozenset(text.split())
            term_hits = len(terms & tokens)
            phrase_hits = sum(1 for phrase in phrases if phrase and phrase in text)
            local = term_hits + phrase_hits
            if local <= 0:
                continue
            hits += local
            evidence.append(
                make_evidence(
                    EvidenceKind.CAREER_FIELD,
                    f"{path}[{offset}]",
                    f"{concept}:{local}",
                )
            )
        if hits <= 0:
            evidence.append(
                make_evidence(EvidenceKind.DERIVED, f"{path}.lexicon_miss", concept)
            )
            corroboration = UnitScore(0.0)
        else:
            corroboration = UnitScore(
                clamp_unit(bounded_log_scale(float(hits), saturation=self.mention_saturation))
            )
        concept_hits = ((concept, hits),) if hits > 0 else ()
        return LexiconMatch(
            concept_hits=concept_hits,
            mention_count=hits,
            corroboration=corroboration,
            evidence=tuple(evidence),
        )


__all__: tuple[str, ...] = ("LexiconEngine", "LexiconMatch")