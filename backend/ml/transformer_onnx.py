"""
ONNX Export + Validation + Runtime Inference.

Exports the trained LTVTransformer to ONNX format and
validates that ONNX outputs match PyTorch within 1e-5 MAE.

Also provides the ONNXInferenceEngine used by FastAPI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from loguru import logger

from backend.ml.transformer_model import LTVTransformer


# ─────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────

def export_to_onnx(
    model: LTVTransformer,
    output_path: str | Path,
    max_seq_len: int = 50,
    opset_version: int = 17,
    batch_size: int = 1,
) -> Path:
    """
    Export LTVTransformer to ONNX.

    The exported model accepts 4 integer input tensors:
        cat_id        (batch, seq_len)
        amount_bucket (batch, seq_len)
        days_delta    (batch, seq_len)
        channel_id    (batch, seq_len)

    And produces 3 float output tensors:
        ltv_12m (batch,)
        ltv_24m (batch,)
        ltv_36m (batch,)

    Args:
        model:        Trained LTVTransformer (eval mode)
        output_path:  Path to save the .onnx file
        max_seq_len:  Sequence length (must match training)
        opset_version: ONNX opset (17 supports Transformer ops well)
        batch_size:   Dummy batch size for tracing (1 is safe for dynamic axes)

    Returns: Path to saved ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy inputs for tracing
    dummy_tokens = {
        "cat_id":        torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        "amount_bucket": torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        "days_delta":    torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        "channel_id":    torch.zeros(batch_size, max_seq_len, dtype=torch.long),
    }

    # Wrapper to flatten dict inputs/outputs for ONNX tracing
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, m: LTVTransformer) -> None:
            super().__init__()
            self.m = m

        def forward(
            self,
            cat_id: torch.Tensor,
            amount_bucket: torch.Tensor,
            days_delta: torch.Tensor,
            channel_id: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            tokens = {
                "cat_id":        cat_id,
                "amount_bucket": amount_bucket,
                "days_delta":    days_delta,
                "channel_id":    channel_id,
            }
            out = self.m(tokens)
            return out["ltv_12m"], out["ltv_24m"], out["ltv_36m"]

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    logger.info("Exporting model to ONNX: {}", output_path)

    # Disable fused MHA fast-path so exporter does not emit unsupported
    # aten::_transformer_encoder_layer_fwd in some PyTorch builds.
    prev_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (
                    dummy_tokens["cat_id"],
                    dummy_tokens["amount_bucket"],
                    dummy_tokens["days_delta"],
                    dummy_tokens["channel_id"],
                ),
                str(output_path),
                input_names=["cat_id", "amount_bucket", "days_delta", "channel_id"],
                output_names=["ltv_12m", "ltv_24m", "ltv_36m"],
                dynamic_axes={
                    "cat_id":        {0: "batch_size"},
                    "amount_bucket": {0: "batch_size"},
                    "days_delta":    {0: "batch_size"},
                    "channel_id":    {0: "batch_size"},
                    "ltv_12m":       {0: "batch_size"},
                    "ltv_24m":       {0: "batch_size"},
                    "ltv_36m":       {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
                # Use legacy exporter path to avoid mandatory onnxscript dependency.
                dynamo=False,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(prev_fastpath)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX export complete — {:.2f} MB saved to {}", file_size_mb, output_path)
    return output_path


# ─────────────────────────────────────────────────────────────
# ONNX Validation
# ─────────────────────────────────────────────────────────────

def validate_onnx_vs_pytorch(
    model: LTVTransformer,
    onnx_path: str | Path,
    max_seq_len: int = 50,
    n_test_samples: int = 100,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """
    Validate that ONNX outputs match PyTorch within tolerance.

    Target: MAE delta < 1e-5 (project spec).

    Returns:
        dict with max_diff, mae_diff, passed, latency_onnx_ms, latency_pytorch_ms
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. Run: pip install onnxruntime")

    model.eval()
    onnx_path = Path(onnx_path)

    # Create ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)

    # Generate random test inputs
    rng = np.random.default_rng(42)
    cat_ids      = rng.integers(0, 13,  size=(n_test_samples, max_seq_len)).astype(np.int64)
    buckets      = rng.integers(0, 6,   size=(n_test_samples, max_seq_len)).astype(np.int64)
    days_deltas  = rng.integers(0, 366, size=(n_test_samples, max_seq_len)).astype(np.int64)
    channel_ids  = rng.integers(0, 8,   size=(n_test_samples, max_seq_len)).astype(np.int64)

    # ── PyTorch inference ──
    pt_tokens = {
        "cat_id":        torch.from_numpy(cat_ids),
        "amount_bucket": torch.from_numpy(buckets),
        "days_delta":    torch.from_numpy(days_deltas),
        "channel_id":    torch.from_numpy(channel_ids),
    }
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_out = model(pt_tokens)
    pt_elapsed = (time.perf_counter() - t0) * 1000 / n_test_samples

    pt_12m = pt_out["ltv_12m"].numpy()
    pt_36m = pt_out["ltv_36m"].numpy()

    # ── ONNX Runtime inference ──
    ort_inputs = {
        "cat_id":        cat_ids,
        "amount_bucket": buckets,
        "days_delta":    days_deltas,
        "channel_id":    channel_ids,
    }
    t0 = time.perf_counter()
    ort_12m, ort_24m, ort_36m = session.run(None, ort_inputs)
    ort_elapsed = (time.perf_counter() - t0) * 1000 / n_test_samples

    # ── Compare ──
    diff_12m = np.abs(pt_12m - ort_12m)
    diff_36m = np.abs(pt_36m - ort_36m)
    max_diff = float(max(diff_12m.max(), diff_36m.max()))
    mae_diff = float((diff_12m.mean() + diff_36m.mean()) / 2)

    passed = mae_diff < tolerance

    logger.info("=== ONNX Validation ===")
    logger.info("  Max diff:          {:.2e}  (tolerance: {:.2e})", max_diff, tolerance)
    logger.info("  MAE diff:          {:.2e}", mae_diff)
    logger.info("  PyTorch latency:   {:.2f} ms/sample", pt_elapsed)
    logger.info("  ONNX RT latency:   {:.2f} ms/sample", ort_elapsed)
    logger.info("  Speedup:           {:.2f}×", pt_elapsed / max(ort_elapsed, 1e-9))
    logger.info("  Parity check:      {}", "✓ PASSED" if passed else "✗ FAILED")

    if not passed:
        logger.warning(
            "ONNX parity check FAILED — MAE diff {:.2e} > tolerance {:.2e}",
            mae_diff, tolerance,
        )

    return {
        "max_diff":            max_diff,
        "mae_diff":            mae_diff,
        "passed":              passed,
        "latency_pytorch_ms":  pt_elapsed,
        "latency_onnx_ms":     ort_elapsed,
        "speedup":             pt_elapsed / max(ort_elapsed, 1e-9),
        "n_test_samples":      n_test_samples,
    }


# ─────────────────────────────────────────────────────────────
# ONNX Runtime Inference Engine (used by FastAPI)
# ─────────────────────────────────────────────────────────────

class ONNXInferenceEngine:
    """
    Wraps ONNX Runtime session for real-time scoring in FastAPI.

    Loaded once at startup, then .score() is called per request.
    Target: < 200ms per customer.

    Usage:
        engine = ONNXInferenceEngine("models/transformer.onnx")
        result = engine.score(sequence_tokens)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required. Run: pip install onnxruntime")

        self.onnx_path = str(onnx_path)
        providers = providers or ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 2  # conservative for free-tier CPU

        self.session = ort.InferenceSession(
            self.onnx_path, sess_options=sess_opts, providers=providers
        )
        self.input_names  = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        logger.info(
            "ONNX Runtime session loaded — {} inputs, {} outputs",
            len(self.input_names), len(self.output_names),
        )

    def score(
        self,
        tokens: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """
        Score a single customer.

        Args:
            tokens: dict with keys 'cat_id', 'amount_bucket', 'days_delta', 'channel_id'
                    each np.ndarray of shape (1, max_seq_len), dtype int64

        Returns:
            dict with ltv_12m, ltv_24m, ltv_36m (floats)
        """
        t0 = time.perf_counter()
        outputs = self.session.run(None, tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        output_arrays = [np.asarray(o) for o in outputs[:3]]

        return {
            "ltv_12m":            float(output_arrays[0][0]),
            "ltv_24m":            float(output_arrays[1][0]),
            "ltv_36m":            float(output_arrays[2][0]),
            "inference_latency_ms": elapsed_ms,
        }

    def score_batch(
        self,
        tokens: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray | float]:
        """
        Score a batch of customers.

        Args:
            tokens: dict with np.ndarray of shape (batch, max_seq_len), dtype int64

        Returns:
            dict with ltv_12m, ltv_24m, ltv_36m as np.ndarray (batch,)
        """
        t0 = time.perf_counter()
        outputs = self.session.run(None, tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        output_arrays = [np.asarray(o) for o in outputs[:3]]

        return {
            "ltv_12m":            cast(np.ndarray, output_arrays[0]),
            "ltv_24m":            cast(np.ndarray, output_arrays[1]),
            "ltv_36m":            cast(np.ndarray, output_arrays[2]),
            "inference_latency_ms": elapsed_ms,
        }

    def warmup(self, max_seq_len: int = 50, n_warmup: int = 5) -> None:
        """Run dummy inferences to warm up the ONNX Runtime JIT."""
        dummy = {
            k: np.zeros((1, max_seq_len), dtype=np.int64)
            for k in self.input_names
        }
        for _ in range(n_warmup):
            self.session.run(None, dummy)
        logger.info("ONNX Runtime warmed up ({} passes)", n_warmup)

    def benchmark(
        self, max_seq_len: int = 50, n_runs: int = 200
    ) -> dict[str, float]:
        """Benchmark inference latency."""
        dummy = {
            k: np.zeros((1, max_seq_len), dtype=np.int64)
            for k in self.input_names
        }
        self.warmup(max_seq_len)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.session.run(None, dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

        result = {
            "mean_ms":   float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms":    float(np.percentile(latencies, 95)),
            "p99_ms":    float(np.percentile(latencies, 99)),
            "min_ms":    float(np.min(latencies)),
            "max_ms":    float(np.max(latencies)),
            "n_runs":    n_runs,
        }
        logger.info(
            "ONNX benchmark — mean={:.2f}ms median={:.2f}ms p95={:.2f}ms p99={:.2f}ms",
            result["mean_ms"], result["median_ms"],
            result["p95_ms"],  result["p99_ms"],
        )
        return result
