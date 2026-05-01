"""
Real-Time LTV Scoring Engine.

Orchestrates all model components for a single customer score request:
  1. Load BG/NBD model → predict purchases + probability alive
  2. Load ONNX Runtime session → predict Transformer LTV
  3. Build meta-features → run XGBoost meta-learner
  4. Apply segmentation + CAC recommendation
  5. Attach causal levers from DB
  6. Return full score response

Target latency: < 200ms (full model), < 20ms (cold-start)

Used by FastAPI /score endpoint in Phase 6.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from backend.ml.segmentation import assign_segment, compute_max_cac
from backend.ml.cold_start import ColdStartScorer


class LTVScoringEngine:
    """
    Unified real-time scoring engine.

    Loaded once at FastAPI startup; .score() called per request.

    Usage:
        engine = LTVScoringEngine.from_config(config)
        result = engine.score(customer_id="cust_9821")
        result = engine.score_cold_start(vertical="healthcare", ...)
    """

    def __init__(
        self,
        bgnbd_model: Any,           # BGNBDModel
        onnx_engine: Any,           # ONNXInferenceEngine
        fusion_model: Any,          # XGBoostMetaLearner
        cold_start_scorer: Any,     # ColdStartScorer
        db_client: Any,             # SupabaseClient
        max_seq_len: int = 50,
        model_version: str = "fusion_v1",
    ) -> None:
        self.bgnbd    = bgnbd_model
        self.onnx     = onnx_engine
        self.fusion   = fusion_model
        self.cold_start = cold_start_scorer
        self.db       = db_client
        self.max_seq_len = max_seq_len
        self.model_version = model_version

        # Cache for fast lookup
        self._rfm_cache: dict[str, dict] = {}
        self._seq_cache: dict[str, list] = {}
        self._causal_cache: dict[str, list] = {}

    # ── Full model scoring ────────────────────────────────────

    def score(
        self,
        customer_id: str,
        return_components: bool = False,
    ) -> dict:
        """
        Score an existing customer with full ensemble.

        Returns the complete API response dict.
        """
        t0 = time.perf_counter()

        # 1. Fetch RFM features from DB
        rfm = self._get_rfm(customer_id)
        if rfm is None:
            return self.score_cold_start_from_id(customer_id)

        # 2. BG/NBD scoring
        bgnbd_result = self.bgnbd.predict_single(
            frequency   = rfm.get("frequency", 0),
            recency_days= rfm.get("recency_days", 0),
            t_days      = rfm.get("t_days", 365),
            monetary_avg= rfm.get("monetary_avg", 50.0),
        )

        # 3. Transformer ONNX scoring
        seq_tokens = self._get_sequence_tokens(customer_id)
        onnx_result = self.onnx.score(seq_tokens)

        # 4. Build meta-features and run fusion
        meta = {
            "bgnbd_ltv_12m":          bgnbd_result["ltv_12m"],
            "bgnbd_ltv_36m":          bgnbd_result["ltv_36m"],
            "transformer_ltv_12m":    onnx_result["ltv_12m"],
            "transformer_ltv_36m":    onnx_result["ltv_36m"],
            "probability_alive":      bgnbd_result["probability_alive"],
            "frequency":              rfm.get("frequency", 0),
            "monetary_avg":           rfm.get("monetary_avg", 50.0),
            "recency_days":           rfm.get("recency_days", 0),
            "t_days":                 rfm.get("t_days", 365),
            "purchase_variance":      rfm.get("purchase_variance", 0),
            "orders_count":           rfm.get("orders_count", 1),
            "avg_days_between_orders":rfm.get("avg_days_between_orders", 30.0),
            "unique_categories":      rfm.get("unique_categories", 1),
        }

        fusion_result = self.fusion.predict_single(meta)

        # 5. Segmentation
        ltv_36m = fusion_result["ltv_36m"]
        segment = assign_segment(ltv_36m)
        max_cac = compute_max_cac(ltv_36m)

        # 6. SHAP explanation
        top_drivers = []
        try:
            shap_contributions = self.fusion.get_customer_shap_explanation(meta, top_n=3)
            top_drivers = [
                f"{c['feature'].replace('_', ' ').title()} "
                f"({c['direction']} LTV by £{abs(c['shap_contribution']):.0f})"
                for c in shap_contributions
            ]
        except Exception:
            pass

        # 7. Causal levers
        causal_levers = self._get_causal_levers(customer_id)

        # 8. Lookalike IDs
        lookalike_ids = self._get_lookalikes(customer_id)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        response = {
            "customer_id":          customer_id,
            "ltv_source":           "full_model",
            "ltv_12m":              round(fusion_result["ltv_12m"], 2),
            "ltv_24m":              round(fusion_result["ltv_24m"], 2),
            "ltv_36m":              round(ltv_36m, 2),
            "ltv_percentile":       self._get_percentile(ltv_36m),
            "segment":              segment,
            "probability_alive_12m": round(bgnbd_result["probability_alive"], 4),
            "recommended_max_cac":  round(max_cac, 2),
            "confidence_interval_36m": self._get_ci(customer_id, ltv_36m),
            "top_ltv_drivers":      top_drivers,
            "causal_levers":        causal_levers,
            "lookalike_customer_ids": lookalike_ids,
            "scoring_latency_ms":   elapsed_ms,
        }

        if return_components:
            response["_components"] = {
                "bgnbd":       bgnbd_result,
                "transformer": onnx_result,
                "fusion":      fusion_result,
                "meta_features": meta,
            }

        logger.debug(
            "Scored {} → £{:.0f} LTV_36m ({}) in {}ms",
            customer_id, ltv_36m, segment, elapsed_ms,
        )
        return response

    # ── Cold-start scoring ────────────────────────────────────

    def score_cold_start(
        self,
        vertical: str,
        company_size: str,
        channel: str,
        plan_tier: str,
        customer_id: str | None = None,
    ) -> dict:
        """Score a zero-transaction customer using firmographic prior."""
        t0 = time.perf_counter()
        result = self.cold_start.score(vertical, company_size, channel, plan_tier)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if customer_id:
            result["customer_id"] = customer_id
        result["scoring_latency_ms"] = elapsed_ms

        return result

    def score_cold_start_from_id(self, customer_id: str) -> dict:
        """Fallback cold-start for existing customer with no RFM data."""
        cust_data = self._get_customer_firmographic(customer_id)
        return self.score_cold_start(
            vertical     = cust_data.get("vertical", "other"),
            company_size = cust_data.get("company_size", "smb"),
            channel      = cust_data.get("acquisition_channel", "organic"),
            plan_tier    = cust_data.get("plan_tier", "free"),
            customer_id  = customer_id,
        )

    # ── Batch scoring ─────────────────────────────────────────

    def score_batch(
        self,
        customer_ids: list[str],
    ) -> list[dict]:
        """Score multiple customers. Used by /batch-score endpoint."""
        results = []
        for cid in customer_ids:
            try:
                results.append(self.score(cid))
            except Exception as exc:
                logger.warning("Scoring failed for {}: {}", cid, exc)
                results.append({
                    "customer_id": cid,
                    "error":       str(exc),
                    "ltv_source":  "error",
                })
        return results

    # ── Cache + DB helpers ─────────────────────────────────────

    def _get_rfm(self, customer_id: str) -> dict | None:
        if customer_id in self._rfm_cache:
            return self._rfm_cache[customer_id]

        rows = self.db.execute_sql(
            """
            SELECT frequency, recency_days, t_days, monetary_avg,
                   monetary_std, purchase_variance, orders_count,
                   avg_days_between_orders, unique_categories
            FROM rfm_features
            WHERE customer_id = :cid
            ORDER BY observation_end_date DESC
            LIMIT 1
            """,
            {"cid": customer_id},
        )
        if not rows:
            return None
        rfm = rows[0]
        self._rfm_cache[customer_id] = rfm
        return rfm

    def _get_sequence_tokens(self, customer_id: str) -> dict:
        """Load purchase sequence and convert to ONNX input tensors."""
        import json

        if customer_id in self._seq_cache:
            tokens_list = self._seq_cache[customer_id]
        else:
            rows = self.db.execute_sql(
                """
                SELECT sequence_json FROM purchase_sequences
                WHERE customer_id = :cid LIMIT 1
                """,
                {"cid": customer_id},
            )
            if rows and rows[0].get("sequence_json"):
                raw = rows[0]["sequence_json"]
                tokens_list = json.loads(raw) if isinstance(raw, str) else raw
            else:
                tokens_list = []
            self._seq_cache[customer_id] = tokens_list

        # Build padded arrays
        seq_len = self.max_seq_len
        cat_ids     = [t.get("cat_id",       0) for t in tokens_list][-seq_len:]
        buckets     = [t.get("amount_bucket", 0) for t in tokens_list][-seq_len:]
        days_deltas = [t.get("days_delta",    0) for t in tokens_list][-seq_len:]
        channels    = [t.get("channel_id",    0) for t in tokens_list][-seq_len:]

        pad = seq_len - len(cat_ids)
        cat_ids     = [0] * pad + cat_ids
        buckets     = [0] * pad + buckets
        days_deltas = [0] * pad + days_deltas
        channels    = [0] * pad + channels

        return {
            "cat_id":        np.array([cat_ids],     dtype=np.int64),
            "amount_bucket": np.array([buckets],     dtype=np.int64),
            "days_delta":    np.array([days_deltas], dtype=np.int64),
            "channel_id":    np.array([channels],    dtype=np.int64),
        }

    def _get_causal_levers(self, customer_id: str) -> list[str]:
        if customer_id in self._causal_cache:
            return self._causal_cache[customer_id]
        try:
            rows = self.db.execute_sql(
                """
                SELECT lever_json FROM causal_lever_recommendations
                WHERE customer_id = :cid
                ORDER BY computed_at DESC LIMIT 1
                """,
                {"cid": customer_id},
            )
            if rows and rows[0].get("lever_json"):
                import json
                levers_raw = rows[0]["lever_json"]
                levers = json.loads(levers_raw) if isinstance(levers_raw, str) else levers_raw
                result = [
                    f"{l['description']} (effect: £{l['effect']:.0f})"
                    for l in levers[:3]
                ]
                self._causal_cache[customer_id] = result
                return result
        except Exception:
            pass
        return []

    def _get_lookalikes(self, customer_id: str, top_n: int = 3) -> list[str]:
        try:
            rows = self.db.execute_sql(
                """
                SELECT candidate_customer_id
                FROM find_lookalikes(:cid, 'transformer_v1', :n)
                ORDER BY similarity DESC
                """,
                {"cid": customer_id, "n": top_n},
            )
            return [r["candidate_customer_id"] for r in rows]
        except Exception:
            return []

    def _get_percentile(self, ltv_36m: float) -> int:
        """Quick percentile estimate from distribution quantiles."""
        # Rough mapping based on typical LTV distribution
        if ltv_36m > 10_000: return 95
        if ltv_36m > 5_000:  return 85
        if ltv_36m > 2_000:  return 70
        if ltv_36m > 1_000:  return 55
        if ltv_36m > 500:    return 40
        return 20

    def _get_ci(
        self, customer_id: str, ltv_36m: float
    ) -> tuple[float, float]:
        """Return confidence interval for 36m LTV."""
        try:
            rows = self.db.execute_sql(
                """
                SELECT ltv_36m_lower, ltv_36m_upper
                FROM transformer_predictions
                WHERE customer_id = :cid
                ORDER BY predicted_at DESC LIMIT 1
                """,
                {"cid": customer_id},
            )
            if rows and rows[0].get("ltv_36m_lower") is not None:
                return (float(rows[0]["ltv_36m_lower"]),
                        float(rows[0]["ltv_36m_upper"]))
        except Exception:
            pass
        # Fallback: ±40% of point estimate
        return (round(ltv_36m * 0.60, 2), round(ltv_36m * 1.40, 2))

    def _get_customer_firmographic(self, customer_id: str) -> dict:
        try:
            rows = self.db.execute_sql(
                """
                SELECT vertical, company_size, acquisition_channel, plan_tier
                FROM customers WHERE customer_id = :cid LIMIT 1
                """,
                {"cid": customer_id},
            )
            return rows[0] if rows else {}
        except Exception:
            return {}

    def warm_cache(self, customer_ids: list[str]) -> None:
        """Pre-warm RFM and sequence caches for a list of customers."""
        logger.info("Warming cache for {} customers…", len(customer_ids))
        for cid in customer_ids:
            self._get_rfm(cid)
            self._get_sequence_tokens(cid)
        logger.info("Cache warmed")