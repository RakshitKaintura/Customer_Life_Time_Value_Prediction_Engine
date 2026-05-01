"""Unit tests for explainability module."""

from __future__ import annotations

import pytest

from backend.ml.explainability import (
    generate_driver_narratives,
    FEATURE_DESCRIPTIONS,
)


def test_generate_driver_narratives_returns_list() -> None:
    shap_contributions = [
        {"feature": "frequency", "value": 8.0, "shap_contribution": 340.0, "direction": "increases"},
        {"feature": "monetary_avg", "value": 250.0, "shap_contribution": 180.0, "direction": "increases"},
        {"feature": "probability_alive", "value": 0.92, "shap_contribution": 120.0, "direction": "increases"},
    ]
    meta = {"frequency": 8.0, "monetary_avg": 250.0, "probability_alive": 0.92}
    narratives = generate_driver_narratives(shap_contributions, meta, top_n=3)
    assert isinstance(narratives, list)
    assert len(narratives) == 3
    for n in narratives:
        assert isinstance(n, str)
        assert len(n) > 10


def test_generate_driver_narratives_top_n() -> None:
    shap_contributions = [
        {"feature": "frequency", "value": 5.0, "shap_contribution": 100.0, "direction": "increases"},
        {"feature": "monetary_avg", "value": 50.0, "shap_contribution": 80.0, "direction": "increases"},
        {"feature": "recency_days", "value": 10.0, "shap_contribution": -50.0, "direction": "decreases"},
    ]
    narratives = generate_driver_narratives(shap_contributions, {}, top_n=2)
    assert len(narratives) == 2


def test_feature_descriptions_coverage() -> None:
    from backend.ml.fusion import META_FEATURES
    for f in META_FEATURES:
        assert f in FEATURE_DESCRIPTIONS, f"No description for feature: {f}"


def test_driver_narrative_contains_direction() -> None:
    shap_contributions = [
        {"feature": "frequency", "value": 2.0, "shap_contribution": -100.0, "direction": "decreases"},
    ]
    narratives = generate_driver_narratives(shap_contributions, {}, top_n=1)
    assert "decreases" in narratives[0].lower()