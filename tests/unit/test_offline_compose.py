

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from redstack.domain.enums import ScoreComponent
from redstack.domain.scoring import ScoringWeights
from redstack.pipelines.offline.compose import (
    GoldLabelSeedMissingError,
    _load_gold_label_seed,
)

_HEADER = "candidate_id,tier,reasoning,reviewer,archetype_id,is_honeypot_suspect,is_borderline,cited_features\n"


def test_load_gold_label_seed_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldLabelSeedMissingError):
        _load_gold_label_seed(tmp_path / "golden_labels.csv")


def test_load_gold_label_seed_parses_csv_rows(tmp_path: Path) -> None:
    path = tmp_path / "golden_labels.csv"
    path.write_text(
        _HEADER
        + 'CAND_0000001,3,"strong production ML fit",alice,2,false,false,career.experience_authenticity;skill.python\n'
        + "CAND_0000002,0,honeypot,bob,,true,false,\n",
        encoding="utf-8",
    )
    seed = _load_gold_label_seed(path)
    assert len(seed.tags) == 2
    first, second = seed.tags
    assert first.candidate_id == "CAND_0000001"
    assert first.tier == 3
    assert first.reviewer == "alice"
    assert first.archetype_id == 2
    assert first.is_honeypot_suspect is False
    assert first.cited_features == ("career.experience_authenticity", "skill.python")
    assert second.archetype_id is None
    assert second.is_honeypot_suspect is True
    assert second.cited_features == ()


def test_load_gold_label_seed_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden_labels.csv"
    path.write_text(_HEADER, encoding="utf-8")
    with pytest.raises(GoldLabelSeedMissingError):
        _load_gold_label_seed(path)


def test_load_gold_label_seed_malformed_row_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden_labels.csv"
    path.write_text(
        _HEADER + "CAND_0000001,not-an-int,x,alice,,false,false,\n", encoding="utf-8"
    )
    with pytest.raises(GoldLabelSeedMissingError):
        _load_gold_label_seed(path)


def test_seed_weights_file_matches_score_component_exactly() -> None:
    """Regression: configs/weights/scoring_weights.yaml's keys must equal
    ScoreComponent exactly -- this is what run_offline_build_with_locked_heuristics
    packages verbatim as scoring_weights.locked.yaml when --golden-labels is not
    given, and ScoringWeights enforces the exact-match contract at load time."""
    path = (
        Path(__file__).resolve().parents[2] / "configs" / "weights" / "scoring_weights.yaml"
    )
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    weights = {ScoreComponent(k): float(v) for k, v in doc["weights"].items()}
    scoring_weights = ScoringWeights(weights=weights, schema_version=doc["layout_version"])
    assert set(scoring_weights.weights) == set(ScoreComponent)
    assert abs(sum(scoring_weights.weights.values()) - 1.0) < 1e-9
