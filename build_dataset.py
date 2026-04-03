"""
build_dataset.py
================
Downloads Amazon Reviews 2023 (Electronics) and builds train/val/test splits
with a deduplicated product corpus.

Usage:
    python build_dataset.py [--data-dir data/] [--n-samples 20000] [--seed 42]
"""

import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Amazon Reviews retrieval dataset")
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="Output directory for parquet files")
    parser.add_argument("--n-samples", type=int, default=100000,
                        help="Number of review-product pairs to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_reviews(n_samples: int, seed: int) -> pd.DataFrame:
    """Load raw Electronics reviews from HuggingFace."""
    print("=" * 60)
    print("Step 1: Loading reviews from HuggingFace...")
    print("=" * 60)
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_Electronics",
            split="full",
            trust_remote_code=True,
        )
        df = ds.to_pandas()
    except Exception as e:
        print(f"[ERROR] Failed to load reviews: {e}")
        raise

    print(f"  Loaded {len(df):,} total reviews")

    # Keep required columns
    needed = ["rating", "title", "text", "asin", "parent_asin", "user_id",
              "timestamp", "helpful_vote", "verified_purchase"]
    available = [c for c in needed if c in df.columns]
    df = df[available].copy()

    # Use 'text' as the review text (renamed to review_text)
    if "text" in df.columns:
        df = df.rename(columns={"text": "review_text"})
    elif "title" in df.columns:
        df = df.rename(columns={"title": "review_text"})

    # Drop rows with missing review text
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip().str.len() > 0]

    # Drop rows with missing product id
    id_col = "parent_asin" if "parent_asin" in df.columns else "asin"
    df = df.dropna(subset=[id_col])
    df = df.rename(columns={id_col: "product_id"})

    print(f"  After cleaning: {len(df):,} reviews")

    # Sample n_samples rows
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    print(f"  Sampled: {len(df):,} review-product pairs")

    return df


def load_metadata(product_ids: list) -> pd.DataFrame:
    """Load product metadata from HuggingFace for the sampled product ids."""
    print("\nStep 2: Loading product metadata from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_Electronics",
            split="full",
            trust_remote_code=True,
        )
        meta_df = ds.to_pandas()
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}")
        raise

    print(f"  Loaded {len(meta_df):,} product metadata entries")

    # Identify the product id column
    if "parent_asin" in meta_df.columns:
        meta_df = meta_df.rename(columns={"parent_asin": "product_id"})
    elif "asin" in meta_df.columns:
        meta_df = meta_df.rename(columns={"asin": "product_id"})

    # Filter to only products we need
    product_set = set(product_ids)
    meta_df = meta_df[meta_df["product_id"].isin(product_set)].copy()
    print(f"  Filtered to {len(meta_df):,} relevant products")

    # Build product document = title + " " + description
    title_col = None
    for c in ["title", "product_title", "name"]:
        if c in meta_df.columns:
            title_col = c
            break

    desc_col = None
    for c in ["description", "features", "details"]:
        if c in meta_df.columns:
            desc_col = c
            break

    if title_col is not None:
        meta_df["product_title"] = meta_df[title_col].fillna("").astype(str)
    else:
        meta_df["product_title"] = ""

    if desc_col is not None:
        def flatten_desc(val):
            if isinstance(val, list):
                return " ".join(str(v) for v in val)
            return str(val) if val is not None else ""
        meta_df["product_description"] = meta_df[desc_col].apply(flatten_desc).fillna("")
    else:
        meta_df["product_description"] = ""

    meta_df["product_doc"] = (
        meta_df["product_title"].str.strip()
        + " "
        + meta_df["product_description"].str.strip()
    ).str.strip()

    # Drop products with empty documents
    meta_df = meta_df[meta_df["product_doc"].str.len() > 0]

    # Keep one entry per product_id (deduplicate)
    meta_df = meta_df.drop_duplicates(subset=["product_id"])

    keep_cols = ["product_id", "product_title", "product_description", "product_doc"]
    keep_cols = [c for c in keep_cols if c in meta_df.columns]
    meta_df = meta_df[keep_cols].reset_index(drop=True)

    print(f"  Final metadata: {len(meta_df):,} unique products")
    return meta_df


def build_splits(df: pd.DataFrame, seed: int):
    """
    Product-level 80/10/10 split.
    No product appears in both train and test.
    """
    print("\nStep 3: Building product-level 80/10/10 splits...")

    # Get unique products
    unique_products = df["product_id"].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_products)

    n = len(unique_products)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)

    train_products = set(unique_products[:n_train])
    val_products   = set(unique_products[n_train : n_train + n_val])
    test_products  = set(unique_products[n_train + n_val:])

    train_df = df[df["product_id"].isin(train_products)].copy().reset_index(drop=True)
    val_df   = df[df["product_id"].isin(val_products)].copy().reset_index(drop=True)
    test_df  = df[df["product_id"].isin(test_products)].copy().reset_index(drop=True)

    print(f"  Unique products — train: {len(train_products)}, "
          f"val: {len(val_products)}, test: {len(test_products)}")
    print(f"  Reviews — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Verify no product overlap
    assert len(train_products & test_products) == 0, "Train/test product overlap!"
    assert len(train_products & val_products) == 0,  "Train/val product overlap!"
    assert len(val_products & test_products) == 0,   "Val/test product overlap!"
    print("  [OK] No product overlap between splits.")

    return train_df, val_df, test_df


def build_corpus(train_df, val_df, test_df, meta_df) -> pd.DataFrame:
    """Build the product corpus from all unique products across splits."""
    print("\nStep 4: Building product corpus...")
    all_product_ids = set(train_df["product_id"]) | set(val_df["product_id"]) | set(test_df["product_id"])
    corpus_df = meta_df[meta_df["product_id"].isin(all_product_ids)].copy().reset_index(drop=True)
    print(f"  Corpus size: {len(corpus_df):,} unique products")
    return corpus_df


def print_statistics(train_df, val_df, test_df, corpus_df):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"  Total review-product pairs : {total:,}")
    print(f"  Train pairs                : {len(train_df):,} ({100*len(train_df)/total:.1f}%)")
    print(f"  Val pairs                  : {len(val_df):,} ({100*len(val_df)/total:.1f}%)")
    print(f"  Test pairs                 : {len(test_df):,} ({100*len(test_df)/total:.1f}%)")
    print(f"  Corpus (unique products)   : {len(corpus_df):,}")

    all_reviews = pd.concat([train_df, val_df, test_df], ignore_index=True)
    lengths = all_reviews["review_text"].str.split().str.len()
    print(f"\n  Avg review length (words)  : {lengths.mean():.1f}")
    print(f"  Median review length       : {lengths.median():.1f}")
    print(f"  Max review length          : {lengths.max()}")
    print(f"  Min review length          : {lengths.min()}")

    doc_lengths = corpus_df["product_doc"].str.split().str.len()
    print(f"\n  Avg product doc length     : {doc_lengths.mean():.1f}")
    print(f"  Median product doc length  : {doc_lengths.median():.1f}")

    # Products with multiple reviews
    vc = all_reviews["product_id"].value_counts()
    print(f"\n  Products with 1 review     : {(vc == 1).sum():,}")
    print(f"  Products with 2-5 reviews  : {((vc >= 2) & (vc <= 5)).sum():,}")
    print(f"  Products with >5 reviews   : {(vc > 5).sum():,}")

    print("\n  Sample review-product pair:")
    row = train_df.iloc[0]
    print(f"    Review  : {row['review_text'][:120]}...")
    print(f"    Product : {corpus_df[corpus_df['product_id']==row['product_id']]['product_doc'].values[0][:120]}...")
    print("=" * 60)


def save_parquets(train_df, val_df, test_df, corpus_df, data_dir: str):
    """Save all splits and corpus as parquet files."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    train_path  = os.path.join(data_dir, "train.parquet")
    val_path    = os.path.join(data_dir, "val.parquet")
    test_path   = os.path.join(data_dir, "test.parquet")
    corpus_path = os.path.join(data_dir, "corpus.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    corpus_df.to_parquet(corpus_path, index=False)

    print(f"\nSaved files:")
    print(f"  {train_path}  ({os.path.getsize(train_path)/1e6:.1f} MB)")
    print(f"  {val_path}    ({os.path.getsize(val_path)/1e6:.1f} MB)")
    print(f"  {test_path}   ({os.path.getsize(test_path)/1e6:.1f} MB)")
    print(f"  {corpus_path} ({os.path.getsize(corpus_path)/1e6:.1f} MB)")


def main():
    args = parse_args()
    set_seed(args.seed)

    print("BLaIR Dataset Builder")
    print(f"  n_samples = {args.n_samples:,}")
    print(f"  seed      = {args.seed}")
    print(f"  data_dir  = {args.data_dir}")

    # Step 1: Load reviews
    reviews_df = load_reviews(args.n_samples, args.seed)

    # Step 2: Load metadata for sampled products
    product_ids = reviews_df["product_id"].unique().tolist()
    meta_df = load_metadata(product_ids)

    # Step 3: Join reviews + metadata, drop rows with missing product docs
    print("\nStep 3: Joining reviews with metadata...")
    df = reviews_df.merge(meta_df[["product_id", "product_doc", "product_title"]],
                          on="product_id", how="inner")
    df = df.dropna(subset=["review_text", "product_doc"])
    df = df[df["review_text"].str.strip().str.len() > 0]
    df = df[df["product_doc"].str.strip().str.len() > 0]
    df = df.reset_index(drop=True)
    print(f"  After join: {len(df):,} pairs with valid review + product doc")

    # Step 4: Product-level splits
    train_df, val_df, test_df = build_splits(df, args.seed)

    # Step 5: Build corpus
    corpus_df = build_corpus(train_df, val_df, test_df, meta_df)

    # Step 6: Print statistics
    print_statistics(train_df, val_df, test_df, corpus_df)

    # Step 7: Save parquets
    save_parquets(train_df, val_df, test_df, corpus_df, args.data_dir)

    print("\n[DONE] Dataset build complete.")

    # Return key stats for notebook use
    return {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "corpus_size": len(corpus_df),
        "avg_review_length": train_df["review_text"].str.split().str.len().mean(),
    }


if __name__ == "__main__":
    main()
