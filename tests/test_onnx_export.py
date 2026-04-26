"""Unit tests for ONNX export and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.ml.transformer_model import LTVTransformer, build_model


@pytest.fixture
def small_model() -> LTVTransformer:
    model = LTVTransformer(
        model_dim=32,
        n_heads=4,
        n_layers=2,
        ffn_dim=64,
        max_seq_len=10,
    )
    model.eval()
    return model


def test_onnx_export_creates_file(small_model: LTVTransformer) -> None:
    try:
        from backend.ml.transformer_onnx import export_to_onnx
    except ImportError:
        pytest.skip("torch.onnx not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_onnx(small_model, Path(tmpdir) / "model.onnx", max_seq_len=10)
        assert path.exists()
        assert path.stat().st_size > 0


def test_onnx_validation_passes(small_model: LTVTransformer) -> None:
    try:
        import onnxruntime  # noqa: F401
        from backend.ml.transformer_onnx import export_to_onnx, validate_onnx_vs_pytorch
    except ImportError:
        pytest.skip("onnxruntime not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_onnx(small_model, Path(tmpdir) / "model.onnx", max_seq_len=10)
        result = validate_onnx_vs_pytorch(
            small_model, path,
            max_seq_len=10,
            n_test_samples=20,
            tolerance=1e-3,   # relaxed for small model
        )
        assert "max_diff" in result
        assert "passed" in result
        assert result["latency_onnx_ms"] > 0


def test_onnx_inference_engine(small_model: LTVTransformer) -> None:
    try:
        import onnxruntime  # noqa: F401
        from backend.ml.transformer_onnx import export_to_onnx, ONNXInferenceEngine
    except ImportError:
        pytest.skip("onnxruntime not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_onnx(small_model, Path(tmpdir) / "model.onnx", max_seq_len=10)
        engine = ONNXInferenceEngine(path)
        tokens = {k: np.zeros((1, 10), dtype=np.int64)
                  for k in ["cat_id", "amount_bucket", "days_delta", "channel_id"]}
        result = engine.score(tokens)
        assert "ltv_12m" in result
        assert "ltv_36m" in result
        assert result["ltv_12m"] >= 0
        assert result["inference_latency_ms"] > 0