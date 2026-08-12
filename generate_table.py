"""
generate_table.py
=================
Generate the final comparison table across all systems, with p-values from
McNemar's tests. Also outputs a LaTeX-formatted table fragment.

Usage:
    python generate_table.py \
        --results-dir results/ \
        --sig-dir results/significance/ \
        --output-dir results/
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final results comparison table")
    parser.add_argument("--results-dir", type=str, default="results/",
                        help="Directory containing all system results")
    parser.add_argument("--sig-dir",    type=str, default="results/significance/",
                        help="Directory with McNemar JSON files")
    parser.add_argument("--output-dir", type=str, default="results/",
                        help="Output directory for CSV and LaTeX tables")
    return parser.parse_args()


# ── System definitions (in display order) ────────────────────────────────────

SYSTEMS = [
    {
        "label":      "BM25 Okapi",
        "subdir":     "bm25",
        "is_baseline": True,
    },
    {
        "label":      "Dense zero-shot (no FT)",
        "subdir":     "zeroshot",
    },
    {
        "label":      "Bi-encoder, mean, random",
        "subdir":     "biencoder",
    },
    {
        "label":      "Dual encoder, mean, random",
        "subdir":     "dual",
    },
    {
        "label":      "Dual encoder, mean, BM25-neg",
        "subdir":     "dual_hardneg",
    },
    {
        "label":      "Dual encoder, CLS, BM25-neg",
        "subdir":     "dual_cls",
    },
    {
        "label":      "Hybrid BM25 + Dual (RRF)",
        "subdir":     "hybrid",
    },
    {
        "label":      "Hybrid + Cross-encoder",
        "subdir":     "reranker",
    },
]

METRIC_COLS = ["ndcg@10", "recall@10", "mrr", "recall@1"]
META_COLS   = ["n_queries", "corpus_size", "latency_ms"]


def load_metrics(results_dir: str, subdir: str) -> dict:
    """Load metrics.json from a results subdirectory."""
    path = os.path.join(results_dir, subdir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_significance(sig_dir: str) -> dict:
    """
    Load all McNemar JSON files.
    Returns dict: {(label_a, label_b): result_dict}
    """
    sig = {}
    if not os.path.exists(sig_dir):
        return sig

    for path in glob.glob(os.path.join(sig_dir, "*.json")):
        try:
            with open(path) as f:
                res = json.load(f)
            key = (res.get("label_a", ""), res.get("label_b", ""))
            sig[key] = res
        except Exception:
            pass

    return sig


def format_pvalue(p: float, sig_99: bool, sig_95: bool, direction: str) -> str:
    """Format p-value string for the table."""
    if direction == "B_better":
        return f"p={p:.3f} ↓"
    if sig_99:
        return "p<0.01 **"
    elif sig_95:
        return "p<0.05 *"
    elif p < 1.0:
        return f"p={p:.3f}"
    return "—"


def get_pvalue_vs_baseline(label: str, sig_map: dict) -> str:
    """Find p-value comparing this system vs BM25 baseline."""
    for (la, lb), res in sig_map.items():
        if la == label and lb == "BM25 Okapi":
            return format_pvalue(
                res["p_value"], res["significant_99"], res["significant_95"], res["direction"]
            )
        if lb == label and la == "BM25 Okapi":
            # Flip direction
            direction = "B_better" if res["direction"] == "A_better" else \
                        "A_better" if res["direction"] == "B_better" else "tie"
            return format_pvalue(res["p_value"], res["significant_99"], res["significant_95"], direction)
    return "—"


def print_comparison_table(rows: list):
    """Print formatted ASCII comparison table."""
    header = f"{'System':<38} {'NDCG@10':>8} {'R@10':>8} {'MRR':>8} {'R@1':>8} {'ms/q':>8} {'p-value':>14}"
    sep    = "─" * len(header)

    print("\n" + "=" * len(header))
    print("BLaIR-style Dual Encoder Dense Retrieval — Results")
    print("=" * len(header))
    print(header)
    print(sep)

    for row in rows:
        ndcg   = f"{row['ndcg@10']:.4f}" if row["ndcg@10"] is not None else "N/A"
        r10    = f"{row['recall@10']:.4f}" if row["recall@10"] is not None else "N/A"
        mrr    = f"{row['mrr']:.4f}" if row["mrr"] is not None else "N/A"
        r1     = f"{row['recall@1']:.4f}" if row["recall@1"] is not None else "N/A"
        pval   = row.get("p_value_str", "baseline")

        lat  = f"{row['latency_ms']:.1f}" if row.get("latency_ms") is not None else "—"
        print(f"  {row['label']:<36} {ndcg:>8} {r10:>8} {mrr:>8} {r1:>8} {lat:>8} {pval:>14}")

    print(sep)
    print("\n  * p<0.05, ** p<0.01 (McNemar's test, continuity-corrected, N=2000)")
    print("  Baseline: BM25 Okapi")
    print("=" * len(header))


def generate_latex(rows: list) -> str:
    """Generate LaTeX table fragment."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Retrieval results on Amazon Electronics (20k reviews, $\sim$6k product corpus). "
        r"All p-values from McNemar's test (continuity-corrected, $N=2000$). "
        r"$\dagger$ p$<$0.05, $\ddagger$ p$<$0.01 vs BM25 baseline.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccccr}",
        r"\toprule",
        r"System & NDCG@10 & R@10 & MRR & R@1 & $p$-value \\",
        r"\midrule",
    ]

    for i, row in enumerate(rows):
        ndcg = f"{row['ndcg@10']:.4f}" if row["ndcg@10"] is not None else "--"
        r10  = f"{row['recall@10']:.4f}" if row["recall@10"] is not None else "--"
        mrr  = f"{row['mrr']:.4f}" if row["mrr"] is not None else "--"
        r1   = f"{row['recall@1']:.4f}" if row["recall@1"] is not None else "--"

        sig  = row.get("p_value_str", "baseline")
        sig_latex = sig.replace("**", r"$\ddagger$").replace("*", r"$\dagger$")

        if i == 0:
            sig_latex = "baseline"

        label_escaped = row["label"].replace("&", r"\&").replace("_", r"\_")
        lines.append(f"{label_escaped} & {ndcg} & {r10} & {mrr} & {r1} & {sig_latex} \\\\")

        # Add midrule after BM25 (baseline)
        if row.get("is_baseline", False):
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def generate_ablation_table(results_dir: str) -> pd.DataFrame:
    """Build ablation table: pooling + temperature ablations."""
    ablation_systems = [
        ("Dual, mean pool, τ=0.05",  "dual_hardneg"),
        ("Dual, cls pool, τ=0.05",   "dual_cls"),
        ("Dual, mean pool, τ=0.10",  "dual_temp010"),
        ("Dual, mean pool, τ=0.20",  "dual_temp020"),
    ]

    rows = []
    for label, subdir in ablation_systems:
        m = load_metrics(results_dir, subdir)
        if m is None:
            continue
        rows.append({
            "Configuration": label,
            "NDCG@10":    round(m.get("ndcg@10", 0), 4),
            "Recall@10":  round(m.get("recall@10", 0), 4),
            "MRR":        round(m.get("mrr", 0), 4),
            "Recall@1":   round(m.get("recall@1", 0), 4),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    print("=" * 60)
    print("Generating Final Results Table")
    print("=" * 60)

    # Load all system metrics
    sig_map = load_significance(args.sig_dir)
    print(f"  Loaded {len(sig_map)} significance test results")

    rows = []
    missing = []
    for system in SYSTEMS:
        m = load_metrics(args.results_dir, system["subdir"])
        if m is None:
            missing.append(system["label"])
            continue

        row = {
            "label":      system["label"],
            "is_baseline": system.get("is_baseline", False),
            "ndcg@10":    m.get("ndcg@10"),
            "recall@10":  m.get("recall@10"),
            "mrr":        m.get("mrr"),
            "recall@1":   m.get("recall@1"),
            "latency_ms": m.get("latency_ms"),
            "n_queries":  m.get("n_queries"),
            "corpus_size":m.get("corpus_size"),
            "p_value_str": "baseline" if system.get("is_baseline") else
                           get_pvalue_vs_baseline(system["label"], sig_map),
        }
        rows.append(row)
        print(f"  Loaded: {system['label']}")

    if missing:
        print(f"\n  [MISSING] {len(missing)} systems not yet evaluated:")
        for m in missing:
            print(f"    - {m}")

    if not rows:
        print("[ERROR] No results found! Run evaluation scripts first.")
        return

    # Print comparison table
    print_comparison_table(rows)

    # Save CSV
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for row in rows:
        csv_rows.append({
            "system":    row["label"],
            "ndcg@10":   row["ndcg@10"],
            "recall@10": row["recall@10"],
            "mrr":       row["mrr"],
            "recall@1":  row["recall@1"],
            "p_value":   row.get("p_value_str", "—"),
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(args.output_dir, "comparison_table.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to: {csv_path}")

    # Generate ablation table
    ablation_df = generate_ablation_table(args.results_dir)
    if not ablation_df.empty:
        ablation_path = os.path.join(args.output_dir, "ablation_table.csv")
        ablation_df.to_csv(ablation_path, index=False)
        print(f"Ablation table saved to: {ablation_path}")
        print("\nAblation Table:")
        print(ablation_df.to_string(index=False))

    # Generate LaTeX
    latex = generate_latex(rows)
    latex_path = os.path.join(args.output_dir, "table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {latex_path}")
    print("\nLaTeX fragment:")
    print(latex)

    # ── Latency-Accuracy Pareto Plot ─────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat_ndcg = [(r["label"], r.get("latency_ms"), r.get("ndcg@10"))
                for r in rows
                if r.get("latency_ms") is not None and r.get("ndcg@10") is not None]

    if len(lat_ndcg) >= 2:
        colors = ["#e74c3c","#3498db","#2ecc71","#f39c12",
                  "#9b59b6","#1abc9c","#e67e22","#34495e"]
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, (label, lat, ndcg) in enumerate(lat_ndcg):
            ax.scatter(lat, ndcg, s=140, color=colors[i % len(colors)],
                       zorder=5)
            ax.annotate(label, (lat, ndcg),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

        # Identify Pareto-optimal points (not dominated on both axes)
        pareto = []
        for lat_i, ndcg_i, label_i in [(l, n, lb) for lb, l, n in lat_ndcg]:
            dominated = any(lat_j <= lat_i and ndcg_j >= ndcg_i and
                            (lat_j < lat_i or ndcg_j > ndcg_i)
                            for lat_j, ndcg_j, _ in [(l, n, lb) for lb, l, n in lat_ndcg])
            if not dominated:
                pareto.append(label_i)
        if pareto:
            ax.set_title(
                f"Latency vs Accuracy — Pareto Frontier\n"
                f"Pareto-optimal: {', '.join(pareto)}", fontsize=12)
        else:
            ax.set_title("Latency vs Accuracy — Pareto Frontier", fontsize=12)

        ax.set_xlabel("Latency (ms/query)", fontsize=12)
        ax.set_ylabel("NDCG@10", fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pareto_path = os.path.join(args.output_dir, "latency_accuracy_pareto.png")
        plt.savefig(pareto_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Pareto plot saved → {pareto_path}")
        if pareto:
            print(f"Pareto-optimal systems: {', '.join(pareto)}")
    else:
        print("[INFO] Not enough latency data for Pareto plot — run evaluation scripts first")

    print("\n[DONE] Table generation complete.")
    return rows


if __name__ == "__main__":
    main()
