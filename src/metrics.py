"""
src/metrics.py
==============
Retrieval evaluation metrics:
  - NDCG@k    (Normalized Discounted Cumulative Gain)
  - Recall@k  (binary, single relevant document)
  - MRR       (Mean Reciprocal Rank)

All metrics handle single-relevant-document setting (one true product per review).
"""

import math
from typing import Dict, List, Optional, Tuple


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_id: str,
    k: int = 10,
) -> float:
    """
    NDCG@k for single relevant document.

    With binary relevance (one correct answer):
        If relevant_id is in retrieved_ids[:k] at rank r (1-indexed):
            DCG  = 1 / log2(r + 1)
            IDCG = 1 / log2(1 + 1) = 1.0   (ideal: rank 1)
            NDCG = DCG / IDCG = 1 / log2(r + 1)
        If not found: NDCG = 0.0

    Args:
        retrieved_ids : list of retrieved product IDs, ordered by relevance
        relevant_id   : the one true positive product ID
        k             : cutoff rank

    Returns:
        float in [0, 1]
    """
    top_k = retrieved_ids[:k]
    if relevant_id in top_k:
        rank = top_k.index(relevant_id) + 1   # 1-indexed
        dcg  = 1.0 / math.log2(rank + 1)
        idcg = 1.0 / math.log2(2)             # perfect: rank 1, log2(2)=1
        return dcg / idcg
    return 0.0


def recall_at_k(
    retrieved_ids: List[str],
    relevant_id: str,
    k: int = 10,
) -> float:
    """
    Recall@k for single relevant document.

    Binary: 1.0 if relevant_id in top-k, else 0.0

    Args:
        retrieved_ids : list of retrieved product IDs
        relevant_id   : the true positive product ID
        k             : cutoff rank

    Returns:
        float: 1.0 or 0.0
    """
    return float(relevant_id in retrieved_ids[:k])


def mrr(
    retrieved_ids: List[str],
    relevant_id: str,
) -> float:
    """
    Mean Reciprocal Rank (MRR) for single relevant document.

    MRR = 1/rank if found in retrieved list, else 0.

    Note: called 'mrr' but computes per-query reciprocal rank.
    The mean is computed in aggregate().

    Args:
        retrieved_ids : ranked list of retrieved product IDs
        relevant_id   : the true positive product ID

    Returns:
        float: reciprocal rank, or 0.0 if not found
    """
    if relevant_id in retrieved_ids:
        rank = retrieved_ids.index(relevant_id) + 1   # 1-indexed
        return 1.0 / rank
    return 0.0


def compute_metrics(
    retrieved_ids: List[str],
    relevant_id: str,
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved_ids : ranked list of retrieved product IDs
        relevant_id   : the true positive product ID
        k_values      : cutoff values for Recall@k

    Returns:
        dict with keys: ndcg@10, recall@1, recall@5, recall@10, mrr
        (and any other recall@k values in k_values)
    """
    metrics: Dict[str, float] = {}

    # NDCG@10 is the primary metric
    metrics["ndcg@10"] = ndcg_at_k(retrieved_ids, relevant_id, k=10)

    # Recall at multiple cutoffs
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_id, k=k)

    # MRR over the full retrieved list
    metrics["mrr"] = mrr(retrieved_ids, relevant_id)

    return metrics


def aggregate(per_query_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Macro-average all metrics over a list of per-query metric dicts.

    Args:
        per_query_list : list of dicts from compute_metrics(), one per query

    Returns:
        dict with same keys, values are macro averages
    """
    if not per_query_list:
        return {}

    keys = per_query_list[0].keys()
    result = {}
    for key in keys:
        values = [d[key] for d in per_query_list if key in d]
        result[key] = sum(values) / len(values) if values else 0.0

    return result


def print_metrics_table(metrics: Dict[str, float], title: str = "Results"):
    """Pretty-print a metrics dict as a formatted table."""
    line = "─" * 50
    print(f"\n{line}")
    print(f"  {title}")
    print(line)

    ordered_keys = ["ndcg@10", "recall@1", "recall@5", "recall@10", "mrr"]
    for k in ordered_keys:
        if k in metrics:
            print(f"  {k:<20} {metrics[k]:.4f}")

    # Print any extra keys not in ordered_keys
    for k, v in metrics.items():
        if k not in ordered_keys:
            print(f"  {k:<20} {v:.4f}")

    print(line)


def hits_at_k(
    retrieved_ids: List[str],
    relevant_id: str,
    k: int = 10,
) -> int:
    """Returns 1 if relevant_id in top-k, else 0. Used for McNemar's test."""
    return int(relevant_id in retrieved_ids[:k])
