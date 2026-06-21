from __future__ import annotations

from redstack.config.schema import CompiledLexicon, CompiledLexiconConcept
from redstack.domain.ids import SkillName
from redstack.engines.lexicon import LexiconEngine

_LEXICON = CompiledLexicon(
    concepts={
        "python": CompiledLexiconConcept(
            terms=frozenset({"python", "django"}),
            phrases=("machine learning",),
        )
    }
)
_ENGINE = LexiconEngine(lexicon=_LEXICON)


def test_lexicon_engine_constructs_from_config_schema() -> None:
    """Regression: redstack.config.schema.CompiledLexicon must exist and be the
    type LexiconEngine.lexicon is annotated with, or the module fails to import."""
    assert _ENGINE.lexicon is _LEXICON


def test_term_hit_produces_positive_corroboration_and_evidence() -> None:
    match = _ENGINE.match_text("python", "I write python and django code", path="p")
    assert match.mention_count == 2
    assert match.corroboration > 0.0
    assert match.evidence


def test_phrase_hit_is_detected() -> None:
    match = _ENGINE.match_text(
        "python", "background in machine learning research", path="p"
    )
    assert match.mention_count == 1
    assert match.concept_hits == (("python", 1),)


def test_no_hit_yields_zero_corroboration_with_miss_evidence() -> None:
    match = _ENGINE.match_text("python", "I cook and garden on weekends", path="p")
    assert match.mention_count == 0
    assert match.corroboration == 0.0
    assert match.concept_hits == ()
    assert match.evidence[0].path.endswith(".lexicon_miss")


def test_corroborate_skill_aggregates_across_descriptions() -> None:
    score = _ENGINE.corroborate_skill(
        SkillName("python"),
        ("worked with python", "more python and django"),
        path="career_history",
    )
    assert score > 0.0
