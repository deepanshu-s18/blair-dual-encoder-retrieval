"""
run_error_analysis.py
=====================
Analyze failure cases from the dense retrieval system.

For the 50 worst-performing queries (lowest Recall@10), categorize failures:
  - "lexical_mismatch"    : review uses colloquial/experiential language,
                            product uses technical specs
  - "too_short_query"     : review is very short (< 5 words), no semantic signal
  - "ambiguous_query"     : review could match many products (generic statement)
  - "rare_product"        : product appears < 3 times in training data
  - "wrong_but_reasonable": top result is plausible, just not THE product

Usage:
    python run_error_analysis.py \
        --results results/dual_hardneg/ \
        --data-dir data/ \
        --output-dir results/error_analysis/
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, ".")


def parse_args():
    parser = argparse.ArgumentParser(description="Failure case analysis for retrieval")
    parser.add_argument("--results",    type=str, required=True,
                        help="Directory with per_query_metrics.parquet from best dense model")
    parser.add_argument("--data-dir",   type=str, default="data/")
    parser.add_argument("--output-dir", type=str, default="results/error_analysis/")
    parser.add_argument("--n-failures", type=int, default=50,
                        help="Number of worst failures to analyze")
    parser.add_argument("--min-train-count", type=int, default=3,
                        help="Products below this count are 'rare'")
    return parser.parse_args()


def load_data(data_dir: str, results_dir: str):
    """Load corpus, test set, training data, and per-query results."""
    corpus_df = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
    test_df   = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    train_df  = pd.read_parquet(os.path.join(data_dir, "train.parquet"))

    pq_path   = os.path.join(results_dir, "per_query_metrics.parquet")
    if not os.path.exists(pq_path):
        raise FileNotFoundError(f"Not found: {pq_path}. Run evaluate_dense.py first.")
    pq_df = pd.read_parquet(pq_path)

    return corpus_df, test_df, train_df, pq_df


def categorize_failure(
    query_text: str,
    true_product: str,
    top3_products: list,
    product_train_count: int,
    min_train_count: int,
) -> str:
    """
    Heuristic categorization of a single failure case.

    Priority order:
    1. too_short_query    → review < 5 words
    2. rare_product       → product_train_count < min_train_count
    3. lexical_mismatch   → low word overlap between query and true product
    4. ambiguous_query    → very common words, could match anything
    5. wrong_but_reasonable (default)
    """
    words = query_text.strip().split()
    n_words = len(words)

    # 1. Too short
    if n_words < 5:
        return "too_short_query"

    # 2. Rare product
    if product_train_count < min_train_count:
        return "rare_product"

    # 3. Lexical mismatch — measure word overlap between query and true product
    query_tokens   = set(re.findall(r"\w+", query_text.lower()))
    product_tokens = set(re.findall(r"\w+", true_product.lower()))
    # Remove stop words
    stop = {"the", "a", "an", "is", "it", "was", "and", "or", "for", "to",
            "with", "this", "that", "my", "i", "in", "on", "of", "but", "so",
            "very", "really", "great", "good", "nice", "works", "well", "use"}
    query_tokens   = query_tokens - stop
    product_tokens = product_tokens - stop

    if not query_tokens or not product_tokens:
        return "lexical_mismatch"

    overlap = len(query_tokens & product_tokens) / len(query_tokens)
    if overlap < 0.05:
        return "lexical_mismatch"

    # 4. Ambiguous (very short meaningful content or generic words)
    generic_words = {"great", "good", "awesome", "perfect", "nice", "ok",
                     "works", "love", "hate", "bad", "excellent", "best",
                     "product", "item", "thing", "bought", "received"}
    content_words = query_tokens - generic_words
    if len(content_words) < 2:
        return "ambiguous_query"

    # 5. Default — top result is plausible, just not THE product
    return "wrong_but_reasonable"


def analyze_failures(
    pq_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_failures: int,
    min_train_count: int,
) -> pd.DataFrame:
    """
    Analyze the n_failures worst performing queries.

    Returns a DataFrame with failure details and categories.
    """
    corpus_lookup = dict(zip(corpus_df["product_id"], corpus_df["product_doc"]))
    corpus_title  = {}
    if "product_title" in corpus_df.columns:
        corpus_title = dict(zip(corpus_df["product_id"], corpus_df["product_title"]))

    # Count product appearances in training data
    train_counts = train_df["product_id"].value_counts().to_dict()

    # Find failed queries (hits@10=0 or lowest Recall@10)
    fail_col = "recall@10" if "recall@10" in pq_df.columns else "hits@10"
    failed_df = pq_df[pq_df[fail_col] == 0].copy() if fail_col in pq_df.columns else pq_df.copy()

    # Sort by score ascending (worst first) and take top-n
    if "ndcg@10" in failed_df.columns:
        failed_df = failed_df.sort_values("ndcg@10", ascending=True)
    failed_df = failed_df.head(n_failures)

    print(f"\nAnalyzing {len(failed_df)} failure cases...")

    records = []
    for _, row in failed_df.iterrows():
        query_text = row.get("query_text", "")
        product_id = row.get("product_id", "")
        true_product = corpus_lookup.get(product_id, "UNKNOWN")
        true_title   = corpus_title.get(product_id, product_id)

        # Parse top-3 retrieved products
        retrieved_str = row.get("retrieved_ids", "")
        if isinstance(retrieved_str, str) and retrieved_str:
            retrieved_ids = [p.strip() for p in retrieved_str.split(",") if p.strip()]
        else:
            retrieved_ids = []

        top3 = [corpus_lookup.get(pid, pid) for pid in retrieved_ids[:3]]
        top3_titles = [corpus_title.get(pid, pid) for pid in retrieved_ids[:3]]

        train_count = train_counts.get(product_id, 0)

        category = categorize_failure(
            query_text, true_product, top3,
            product_train_count=train_count,
            min_train_count=min_train_count,
        )

        records.append({
            "query_text":     query_text,
            "true_product_id": product_id,
            "true_product":   true_title,
            "top1_product":   top3_titles[0] if top3_titles else "",
            "top2_product":   top3_titles[1] if len(top3_titles) > 1 else "",
            "top3_product":   top3_titles[2] if len(top3_titles) > 2 else "",
            "train_count":    train_count,
            "category":       category,
            "ndcg@10":        row.get("ndcg@10", 0.0),
            "recall@10":      row.get("recall@10", row.get("hits@10", 0.0)),
        })

    return pd.DataFrame(records)


def print_analysis(failures_df: pd.DataFrame):
    """Print formatted analysis tables."""
    print("\n" + "=" * 70)
    print("FAILURE MODE DISTRIBUTION")
    print("=" * 70)
    category_counts = failures_df["category"].value_counts()
    total = len(failures_df)
    for cat, count in category_counts.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {cat:<25} {count:3d} ({pct:5.1f}%)  {bar}")

    print(f"\n  Total failure cases analyzed: {total}")

    print("\n" + "=" * 70)
    print("SAMPLE FAILURE CASES (5 per category)")
    print("=" * 70)

    for category in failures_df["category"].unique():
        cat_df = failures_df[failures_df["category"] == category].head(3)
        print(f"\n── Category: {category} ──")
        for _, row in cat_df.iterrows():
            print(f"\n  Query   : {row['query_text'][:100]}")
            print(f"  True    : {row['true_product'][:80]}")
            print(f"  Top-1   : {row['top1_product'][:80]}")
            print(f"  Top-2   : {row['top2_product'][:80]}")

    print("\n" + "=" * 70)
    print("ACTIONABLE INSIGHTS")
    print("=" * 70)
    cat_counts = failures_df["category"].value_counts()

    if "lexical_mismatch" in cat_counts:
        pct = 100 * cat_counts["lexical_mismatch"] / len(failures_df)
        print(f"\n  Lexical mismatch ({pct:.0f}%):")
        print("    → Query augmentation / back-translation during training")
        print("    → BM25 hard negatives specifically help this failure mode")

    if "too_short_query" in cat_counts:
        pct = 100 * cat_counts["too_short_query"] / len(failures_df)
        print(f"\n  Too short query ({pct:.0f}%):")
        print("    → Filter out reviews < N words at inference time")
        print("    → Concatenate review title + text as query")

    if "rare_product" in cat_counts:
        pct = 100 * cat_counts["rare_product"] / len(failures_df)
        print(f"\n  Rare product ({pct:.0f}%):")
        print("    → Data augmentation for products with few reviews")
        print("    → Cross-encoder reranker particularly helps here")

    if "ambiguous_query" in cat_counts:
        pct = 100 * cat_counts["ambiguous_query"] / len(failures_df)
        print(f"\n  Ambiguous query ({pct:.0f}%):")
        print("    → Cannot be solved without additional context")
        print("    → User history / session context would help")

    if "wrong_but_reasonable" in cat_counts:
        pct = 100 * cat_counts["wrong_but_reasonable"] / len(failures_df)
        print(f"\n  Wrong but reasonable ({pct:.0f}%):")
        print("    → More training data and larger batch size")
        print("    → Cross-encoder reranker to improve final ranking")

    print()


def main():
    args = parse_args()

    print("=" * 70)
    print("Dense Retrieval Error Analysis")
    print("=" * 70)
    print(f"  Results   : {args.results}")
    print(f"  Failures  : {args.n_failures}")

    # Load data
    corpus_df, test_df, train_df, pq_df = load_data(args.data_dir, args.results)
    print(f"\n  Test queries : {len(pq_df):,}")
    print(f"  Corpus       : {len(corpus_df):,} products")
    print(f"  Training data: {len(train_df):,} pairs")

    # Analyze failures
    failures_df = analyze_failures(
        pq_df, corpus_df, train_df,
        n_failures=args.n_failures,
        min_train_count=args.min_train_count,
    )

    # Print analysis
    print_analysis(failures_df)

    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    failures_path = os.path.join(args.output_dir, "failure_cases.parquet")
    failures_df.to_parquet(failures_path, index=False)
    print(f"Failure cases saved to: {failures_path}")

    # Save category counts CSV
    category_counts = failures_df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]
    category_counts["pct"] = 100 * category_counts["count"] / len(failures_df)
    csv_path = os.path.join(args.output_dir, "error_categories.csv")
    category_counts.to_csv(csv_path, index=False)
    print(f"Category counts saved to: {csv_path}")

    # Print final category table
    print("\n" + "=" * 70)
    print("FINAL ERROR CATEGORY TABLE")
    print("=" * 70)
    print(category_counts.to_string(index=False))
    print("=" * 70)

    print("\n[DONE] Error analysis complete.")
    return failures_df


if __name__ == "__main__":
    main()
