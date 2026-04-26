"""
CLS token embedding storage for pgvector integration.

After Transformer training, CLS token embeddings are extracted and stored in
Supabase `customer_embeddings`. These power the lookalike similarity endpoint.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from backend.ml.sequence_dataset import PurchaseSequenceDataset, collate_fn
from backend.ml.transformer_model import LTVTransformer


@torch.no_grad()
def extract_embeddings(
    model: LTVTransformer,
    dataset: PurchaseSequenceDataset,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Extract CLS token embeddings for all customers.

    Args:
        model:      Trained LTVTransformer (eval mode)
        dataset:    PurchaseSequenceDataset
        batch_size: Inference batch size
        device:     torch.device

    Returns:
        (customer_ids list, embeddings np.ndarray of shape (N, model_dim))
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_ids: list[str] = []
    all_embs: list[np.ndarray] = []

    for batch in loader:
        tokens = {k: v.to(device) for k, v in batch["tokens"].items()}
        output = model(tokens, return_embedding=True)
        embs = output["embedding"].cpu().numpy()

        all_ids.extend(batch["customer_ids"])
        all_embs.append(embs)

    embeddings = np.vstack(all_embs)
    logger.info(
        "Extracted {} embeddings - shape {} - dtype {}",
        len(all_ids), embeddings.shape, embeddings.dtype,
    )
    return all_ids, embeddings


def _get_pgvector_dim(
    db_client: Any,
    table_name: str,
    column_name: str,
) -> int | None:
    """
    Resolve pgvector dimension from Postgres metadata.

    Returns None when the column is missing or not typed as vector(n).
    """
    rows = db_client.execute_sql(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS col_type
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = :table_name
          AND a.attname = :column_name
          AND a.attnum > 0
          AND NOT a.attisdropped
        """,
        params={"table_name": table_name, "column_name": column_name},
    )
    if not rows:
        return None

    col_type = str(rows[0].get("col_type") or "")
    match = re.search(r"vector\((\d+)\)", col_type)
    return int(match.group(1)) if match else None


def store_embeddings(
    customer_ids: list[str],
    embeddings: np.ndarray,
    model_version: str,
    db_client: Any,
    batch_size: int = 200,
) -> int:
    """
    Upsert CLS embeddings into Supabase customer_embeddings table.

    The pgvector HNSW index handles ANN queries automatically.

    If incoming embedding dimensionality does not match the pgvector column
    dimensionality, vectors are aligned to the DB shape:
    - larger -> truncated
    - smaller -> zero-padded

    Args:
        customer_ids:  list of customer_id strings
        embeddings:    np.ndarray (N, D)
        model_version: version tag for the model
        db_client:     SupabaseClient instance
        batch_size:    upsert batch size

    Returns: number of rows inserted
    """
    from datetime import datetime, timezone

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape (N, D), got {embeddings.shape}")
    assert len(customer_ids) == len(embeddings), "ID / embedding count mismatch"

    source_dim = int(embeddings.shape[1])
    target_dim = _get_pgvector_dim(
        db_client=db_client,
        table_name="customer_embeddings",
        column_name="embedding",
    )
    if target_dim is not None and source_dim != target_dim:
        logger.warning(
            "Embedding dim mismatch for customer_embeddings.embedding: model={} db={}. "
            "Applying automatic alignment (truncate/pad).",
            source_dim,
            target_dim,
        )
        if source_dim > target_dim:
            embeddings = embeddings[:, :target_dim]
        else:
            pad_width = target_dim - source_dim
            embeddings = np.pad(embeddings, ((0, 0), (0, pad_width)), mode="constant")

    records = []
    created_at = datetime.now(timezone.utc).isoformat()
    for cid, emb in zip(customer_ids, embeddings):
        records.append(
            {
                "customer_id": cid,
                "embedding": emb.astype(np.float32).tolist(),
                "model_version": model_version,
                "created_at": created_at,
            }
        )

    n = db_client.bulk_upsert(
        "customer_embeddings",
        records,
        conflict_columns=["customer_id", "model_version"],
        batch_size=batch_size,
    )
    logger.info("Stored {} embeddings (model_version={})", n, model_version)
    return n


def find_lookalikes(
    query_customer_id: str,
    model_version: str,
    db_client: Any,
    top_n: int = 10,
) -> list[dict]:
    """
    Find top-N lookalike customers via pgvector ANN search.
    Calls the find_lookalikes SQL function defined in schema.sql.

    Returns list of dicts: {candidate_customer_id, similarity, ltv_36m, segment}
    """
    rows = db_client.execute_sql(
        """
        SELECT candidate_customer_id, similarity, ltv_36m, segment
        FROM find_lookalikes(:query_id, :model_ver, :top_n)
        ORDER BY similarity DESC
        """,
        params={
            "query_id": query_customer_id,
            "model_ver": model_version,
            "top_n": top_n,
        },
    )
    return rows
