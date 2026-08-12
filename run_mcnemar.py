"""
run_mcnemar.py
==============
McNemar's test for statistical significance between two retrieval systems.

Uses MANUAL chi2 computation — scipy.stats.mcnemar was removed in v1.14.

McNemar's test (continuity-corrected):
    n10 = queries where A correct (hits@10=1), B wrong (hits@10=0)
    n01 = queries where A wrong  (hits@10=0), B correct (hits@10=1)
    stat = (|n10 - n01| - 1)^2 / (n10 + n01)
    p    = 1 - chi2.cdf(stat, df=1)

Interpretation:
    p < 0.05 → statistically significant difference (95% confidence)
    p < 0.01 → highly significant (99% confidence)
Bonferroni correction (4 tests, α=0.05/4=0.0125):
    Adjusted significance threshold: p < 0.0125

Usage:
    python run_mcnemar.py \
        --a results/dual_hardneg \
        --b results/bm25 \
        --label-a "Dual+HardNeg" \
        --label-b "BM25" \
        --metric recall@10 \
        --out results/significance/dual_vs_bm25.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, ".")


def parse_args():
    parser = argparse.ArgumentParser(description="McNemar's test for retrieval significance")
    parser.add_argument("--a",        type=str, required=True,
                        help="Directory with per_query_metrics.parquet for system A")
    parser.add_argument("--b",        type=str, required=True,
                        help="Directory with per_query_metrics.parquet for system B")
    parser.add_argument("--label-a",  type=str, default="System_A")
    parser.add_argument("--label-b",  type=str, default="System_B")
    parser.add_argument("--metric",   type=str, default="recall@10",
                        choices=["recall@1", "recall@5", "recall@10", "hits@10"],
                        help="Binary metric to use (default: recall@10)")
    parser.add_argument("--out",      type=str, default=None,
                        help="Output JSON path (default: results/significance/{label_a}_vs_{label_b}.json)")
    return parser.parse_args()


def load_per_query(results_dir: str) -> pd.DataFrame:
    """Load per-query results parquet."""
    path = os.path.join(results_dir, "per_query_metrics.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Per-query metrics not found: {path}\n"
            f"Run the evaluation script for this system first."
        )
    df = pd.read_parquet(path)
    return df


def mcnemar_test(n10: int, n01: int) -> dict:
    """
    McNemar's test with continuity correction.

    n10 = A correct, B wrong
    n01 = A wrong,   B correct

    χ² = (|n10 - n01| - 1)² / (n10 + n01)
    p   = 1 - chi2.cdf(χ², df=1)

    Returns dict with all test statistics.
    """
    n_discordant = n10 + n01

    if n_discordant == 0:
        result = {
            "n10": n10,
            "n01": n01,
            "n_discordant": 0,
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "significant_95": False,
            "significant_99": False,
            "direction": "tie",
            "note": "No discordant pairs — systems are identical on this metric",
        }
        result["significant"] = result["significant_95"]
        result["chi2"]        = result["chi2_stat"]
        return result

    # Continuity-corrected McNemar statistic
    numerator = (abs(n10 - n01) - 1) ** 2
    chi2_stat = numerator / n_discordant
    p_value   = 1.0 - chi2.cdf(chi2_stat, df=1)

    direction = "A_better" if n10 > n01 else "B_better" if n01 > n10 else "tie"

    result = {
        "n10": n10,
        "n01": n01,
        "n_discordant": n_discordant,
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "significant_95": bool(p_value < 0.05),
        "significant_99": bool(p_value < 0.01),
        "direction": direction,
    }
    # Aliases expected by callers and audit specification
    result["significant"] = result["significant_95"]
    result["chi2"]        = result["chi2_stat"]
    return result


def align_queries(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple:
    """
    Align two per-query DataFrames on (query_text, product_id).
    Returns (aligned_a, aligned_b) DataFrames with same rows.
    """
    # Merge on query_text + product_id to ensure same ordering
    merged = df_a[["query_text", "product_id"]].copy()
    merged = merged.merge(
        df_b[["query_text", "product_id"]].assign(in_b=True),
        on=["query_text", "product_id"],
        how="inner",
    )
    n_a   = len(df_a)
    n_b   = len(df_b)
    n_int = len(merged)
    if n_int < min(n_a, n_b):
        print(f"[WARNING] Only {n_int} queries in common "
              f"(A has {n_a}, B has {n_b}). Using intersection.")

    # Filter both to common queries
    key_set = set(zip(merged["query_text"], merged["product_id"]))
    mask_a  = df_a.apply(lambda r: (r["query_text"], r["product_id"]) in key_set, axis=1)
    mask_b  = df_b.apply(lambda r: (r["query_text"], r["product_id"]) in key_set, axis=1)

    aligned_a = df_a[mask_a].sort_values(["query_text", "product_id"]).reset_index(drop=True)
    aligned_b = df_b[mask_b].sort_values(["query_text", "product_id"]).reset_index(drop=True)

    return aligned_a, aligned_b


def run_mcnemar(
    dir_a: str,
    dir_b: str,
    label_a: str,
    label_b: str,
    metric: str = "recall@10",
) -> dict:
    """
    Programmatic entry point for McNemar's test.

    Loads per_query_metrics.parquet from dir_a and dir_b,
    aligns on query_id (positional fallback when absent),
    runs mcnemar_test(), returns result dict with all keys
    including aliases: significant, chi2.

    Args:
        dir_a   : directory containing per_query_metrics.parquet for system A
        dir_b   : directory containing per_query_metrics.parquet for system B
        label_a : human-readable label for system A
        label_b : human-readable label for system B
        metric  : binary column to compare (default "recall@10")

    Returns:
        dict with keys: n10, n01, chi2, chi2_stat, p_value,
                        significant, significant_95, significant_99,
                        direction, n_discordant
    """
    import numpy as np

    df_a = load_per_query(dir_a)
    df_b = load_per_query(dir_b)

    # Alignment strategy 1: merge on query_id
    if "query_id" in df_a.columns and "query_id" in df_b.columns:
        merged = df_a.merge(df_b, on="query_id", suffixes=("_a", "_b"))
        col_a = f"{metric}_a" if f"{metric}_a" in merged.columns else metric
        col_b = f"{metric}_b" if f"{metric}_b" in merged.columns else metric
        hits_a = merged[col_a].astype(int).values
        hits_b = merged[col_b].astype(int).values
    # Alignment strategy 2: merge on query_text + product_id
    elif "query_text" in df_a.columns and "query_text" in df_b.columns:
        aligned_a, aligned_b = align_queries(df_a, df_b)
        hits_a = aligned_a[metric].astype(int).values
        hits_b = aligned_b[metric].astype(int).values
    # Alignment strategy 3: positional (same order assumed)
    else:
        n = min(len(df_a), len(df_b))
        hits_a = df_a[metric].astype(int).values[:n]
        hits_b = df_b[metric].astype(int).values[:n]

    hits_a = np.asarray(hits_a)
    hits_b = np.asarray(hits_b)

    n10 = int(((hits_a == 1) & (hits_b == 0)).sum())
    n01 = int(((hits_a == 0) & (hits_b == 1)).sum())

    result = mcnemar_test(n10, n01)
    result.update({"label_a": label_a, "label_b": label_b, "metric": metric})
    return result


def main():
    args = parse_args()

    print("=" * 60)
    print("McNemar's Statistical Significance Test")
    print("=" * 60)
    print(f"  System A : {args.label_a} ({args.a})")
    print(f"  System B : {args.label_b} ({args.b})")
    print(f"  Metric   : {args.metric}")

    # Load per-query results
    print("\nLoading per-query results...")
    df_a = load_per_query(args.a)
    df_b = load_per_query(args.b)
    print(f"  A queries: {len(df_a):,}")
    print(f"  B queries: {len(df_b):,}")

    # Align queries
    aligned_a, aligned_b = align_queries(df_a, df_b)
    n_queries = len(aligned_a)
    print(f"  Aligned:   {n_queries:,} shared queries")

    if n_queries == 0:
        print("[ERROR] No queries in common between A and B!")
        sys.exit(1)

    # Extract binary hit column
    metric_col = args.metric
    # hits@10 and recall@10 are interchangeable (both binary)
    if metric_col not in aligned_a.columns:
        if "hits@10" in aligned_a.columns:
            metric_col = "hits@10"
        elif "recall@10" in aligned_a.columns:
            metric_col = "recall@10"
        else:
            raise ValueError(
                f"Column '{args.metric}' not in A results. "
                f"Available: {list(aligned_a.columns)}"
            )

    hits_a = aligned_a[metric_col].astype(int).values
    hits_b = aligned_b[metric_col].astype(int).values

    # Compute contingency table
    n10 = int(((hits_a == 1) & (hits_b == 0)).sum())  # A correct, B wrong
    n01 = int(((hits_a == 0) & (hits_b == 1)).sum())  # A wrong,   B correct
    n11 = int(((hits_a == 1) & (hits_b == 1)).sum())  # both correct
    n00 = int(((hits_a == 0) & (hits_b == 0)).sum())  # both wrong

    print(f"\nContingency table ({n_queries} queries):")
    print(f"              B=correct   B=wrong")
    print(f"  A=correct     {n11:5d}      {n10:5d}   | A_total_correct = {n11+n10}")
    print(f"  A=wrong       {n01:5d}      {n00:5d}   | A_total_wrong   = {n01+n00}")
    print(f"                {n11+n01:5d}      {n10+n00:5d}   | N = {n_queries}")

    # McNemar's test
    result = mcnemar_test(n10, n01)
    result.update({
        "n11": n11,
        "n00": n00,
        "n_queries": n_queries,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "metric": metric_col,
    })

    # Print results
    print(f"\nMcNemar's Test (continuity-corrected):")
    print(f"  n10 (A correct, B wrong) : {result['n10']}")
    print(f"  n01 (A wrong, B correct) : {result['n01']}")
    print(f"  χ² statistic             : {result['chi2_stat']:.4f}")
    print(f"  p-value                  : {result['p_value']:.6f}")
    print(f"  Significant at α=0.05    : {result['significant_95']}")
    print(f"  Significant at α=0.01    : {result['significant_99']}")
    print(f"  Direction                : {result['direction']}")

    if result["significant_99"]:
        if result["direction"] == "A_better":
            print(f"\n  → {args.label_a} is SIGNIFICANTLY BETTER than {args.label_b} (p<0.01)")
        else:
            print(f"\n  → {args.label_b} is SIGNIFICANTLY BETTER than {args.label_a} (p<0.01)")
    elif result["significant_95"]:
        if result["direction"] == "A_better":
            print(f"\n  → {args.label_a} is significantly better than {args.label_b} (p<0.05)")
        else:
            print(f"\n  → {args.label_b} is significantly better than {args.label_a} (p<0.05)")
    else:
        print(f"\n  → No significant difference (p={result['p_value']:.4f} > 0.05)")

    # Save result
    if args.out is None:
        Path("results/significance").mkdir(parents=True, exist_ok=True)
        label_a_safe = args.label_a.replace(" ", "_").replace("/", "_")
        label_b_safe = args.label_b.replace(" ", "_").replace("/", "_")
        args.out = f"results/significance/{label_a_safe}_vs_{label_b_safe}.json"

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {args.out}")

    print("=" * 60)
    print("[DONE] McNemar's test complete.")

    return result


if __name__ == "__main__":
    main()
