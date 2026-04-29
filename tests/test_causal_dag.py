"""Unit tests for causal DAG helpers."""

from __future__ import annotations

from backend.ml.causal_dag import (
    get_dag_records,
    DAG_NODE_METADATA,
    CAUSAL_DAG_GML,
)


def test_dag_node_metadata_has_required_keys() -> None:
    for name, meta in DAG_NODE_METADATA.items():
        assert "type" in meta, f"{name} missing 'type'"
        assert "description" in meta, f"{name} missing 'description'"
        assert meta["type"] in {"treatment", "outcome", "confounder", "instrument"}


def test_dag_has_outcome_node() -> None:
    outcome_nodes = [k for k, v in DAG_NODE_METADATA.items() if v["type"] == "outcome"]
    assert len(outcome_nodes) >= 1


def test_dag_has_treatment_nodes() -> None:
    treatments = [k for k, v in DAG_NODE_METADATA.items() if v["type"] == "treatment"]
    assert len(treatments) >= 3


def test_get_dag_records_returns_nodes_and_edges() -> None:
    nodes, edges = get_dag_records("test_v1")
    assert len(nodes) > 0
    assert len(edges) > 0


def test_get_dag_records_model_version() -> None:
    nodes, edges = get_dag_records("my_version")
    for n in nodes:
        assert n["model_version"] == "my_version"
    for e in edges:
        assert e["model_version"] == "my_version"


def test_get_dag_records_node_types() -> None:
    nodes, _ = get_dag_records("v1")
    valid_types = {"treatment", "outcome", "confounder", "instrument"}
    for n in nodes:
        assert n["node_type"] in valid_types


def test_gml_string_not_empty() -> None:
    assert len(CAUSAL_DAG_GML.strip()) > 100
    assert "node" in CAUSAL_DAG_GML
    assert "edge" in CAUSAL_DAG_GML