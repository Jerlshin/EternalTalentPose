from __future__ import annotations

import random
from datetime import date

import pytest

from redstack.domain.ids import Similarity
from redstack.domain.source import RawCandidate
from redstack.features.extraction import (
    _assemble,
    build_cells,
    extract_row_with_base_cells,
    fold_semantic,
)
from redstack.features.parsing import validate as parse_raw
from redstack.features.registry import FEATURE_REGISTRY
from tests.integration.test_online_pipeline import (
    _make_candidate,
    _make_consulting_only,
    _make_honeypot,
)

pytestmark = pytest.mark.unit

_AS_OF = date(2024, 6, 18)
#: Deliberately covers every competency group with a distinct sign/magnitude
#: (positive, negative, zero, near +-1) so a stale semantic-dependent cell
#: would show up as a real value mismatch, not a coincidental zero == zero.
_SEMANTIC_MAP: dict[str, Similarity] = {
    "jd.retr": Similarity(0.6),
    "jd.rank": Similarity(-0.2),
    "jd.recsys": Similarity(0.1),
    "jd.ir": Similarity(0.9),
    "jd.nlp": Similarity(-0.7),
    "jd.llm": Similarity(0.0),
    "jd.mle": Similarity(0.33),
    "jd.mlops": Similarity(-0.95),
    "jd.eval": Similarity(0.5),
}


def _raw_candidates() -> list[RawCandidate]:
    rng = random.Random(4242)
    return [
        parse_raw(_make_candidate(1, rng)),
        parse_raw(_make_candidate(2, rng)),
        parse_raw(_make_honeypot("CAND_9000001")),
        parse_raw(_make_consulting_only("CAND_9000004")),
    ]


@pytest.mark.parametrize("raw", _raw_candidates(), ids=lambda r: str(r.candidate_id))
def test_cached_base_cells_path_matches_from_scratch_build_cells(
    raw: RawCandidate,
) -> None:
    """R2's extract_row_with_base_cells + R3's fold_semantic (the cached path)
    must produce the exact same 145 cell values as calling build_cells directly
    with the resolved semantic map (the from-scratch path).

    fold_semantic skips re-deriving the semantic-independent extractors
    (career/pvs/geography/education/_simple_groups/signals/honeypot) by reusing
    R2's cached base_cells -- a wrong dependency boundary here would silently
    feed a stale (pre-semantic) value into scoring without raising anything.
    """
    expected = build_cells(raw, as_of=_AS_OF, semantic=_SEMANTIC_MAP)

    row, conf, base_cells = extract_row_with_base_cells(
        raw, FEATURE_REGISTRY, as_of=_AS_OF
    )
    # extract_row_with_base_cells (like extract_row) returns read-only arrays
    # (_assemble sets write=False); the real R2->R3 flow copies them into the
    # bulk writable (N,D) CQV before R3 ever mutates a row in place.
    row = row.copy()
    conf = conf.copy()
    actual = fold_semantic(
        row, conf, raw, FEATURE_REGISTRY, base_cells, semantic=_SEMANTIC_MAP
    )

    assert set(actual) == set(expected)
    for feature_id, expected_cell in expected.items():
        actual_cell = actual[feature_id]
        assert actual_cell.value == pytest.approx(expected_cell.value, abs=1e-9), (
            f"{feature_id}: value diverged between cached and from-scratch paths"
        )
        assert actual_cell.confidence == pytest.approx(
            expected_cell.confidence, abs=1e-9
        ), f"{feature_id}: confidence diverged between cached and from-scratch paths"


@pytest.mark.parametrize("raw", _raw_candidates(), ids=lambda r: str(r.candidate_id))
def test_folded_row_matches_assembling_the_from_scratch_cells(
    raw: RawCandidate,
) -> None:
    """The (row, confidence) arrays fold_semantic mutates in place must match
    assembling build_cells' from-scratch output through the same registry
    layout -- i.e. the cached path and the from-scratch path must agree not
    just on the cell dict but on the final numeric CQV row R5 scores from.
    """
    expected_cells = build_cells(raw, as_of=_AS_OF, semantic=_SEMANTIC_MAP)
    expected_row, expected_conf = _assemble(expected_cells, FEATURE_REGISTRY)

    row, conf, base_cells = extract_row_with_base_cells(
        raw, FEATURE_REGISTRY, as_of=_AS_OF
    )
    row = row.copy()
    conf = conf.copy()
    fold_semantic(row, conf, raw, FEATURE_REGISTRY, base_cells, semantic=_SEMANTIC_MAP)

    assert row.tolist() == pytest.approx(expected_row.tolist(), abs=1e-6)
    assert conf.tolist() == pytest.approx(expected_conf.tolist(), abs=1e-6)
