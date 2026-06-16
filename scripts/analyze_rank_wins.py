#!/usr/bin/env python3
"""
Analyze per-query rank CSVs produced by dump_per_query_ranks.py.

Requires only pandas + numpy (no PyTorch).

Usage:
    python scripts/analyze_rank_wins.py \\
        --out_dir outputs/rank_analysis \\
        --method baseline=outputs/additive_umls_baseline/ranks.csv \\
        --method fm=outputs/additive_umls_fm_lr/ranks.csv \\
        --method gat=outputs/gat_umls_rule_attn/ranks.csv

After retraining paper_sum, add:
        --method paper_sum=outputs/additive_umls_paper_sum/ranks.csv \\
        --method paper_sum_fm=outputs/additive_umls_paper_sum_fm/ranks.csv

Outputs (written to --out_dir):
    merged_ranks.csv      per-query table with ranks for every method
    per_relation.csv      per-relation MRR / win-rate breakdown
    summary.txt           human-readable summary printed to stdout + file
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midpoint_rank(L: pd.Series, H: pd.Series) -> pd.Series:
    """Scalar rank = (L + H - 1) / 2 (standard mid-point for tie bands)."""
    return (L + H - 1) / 2.0


def mrr_from_ranks(ranks: pd.Series) -> float:
    return float((1.0 / ranks).mean())


def hit_at(ranks: pd.Series, k: int) -> float:
    return float((ranks <= k).mean())


def aggregate_metrics(ranks: pd.Series, topk: int = 10) -> dict:
    return {
        "MRR":       mrr_from_ranks(ranks),
        "MR":        float(ranks.mean()),
        f"Hit@1":    hit_at(ranks, 1),
        f"Hit@3":    hit_at(ranks, 3),
        f"Hit@{topk}": hit_at(ranks, topk),
    }


def wins_losses_ties(rank_a: pd.Series, rank_b: pd.Series):
    """Return (wins, losses, ties) where 'a wins' means rank_a < rank_b."""
    wins   = int((rank_a < rank_b).sum())
    losses = int((rank_a > rank_b).sum())
    ties   = int((rank_a == rank_b).sum())
    return wins, losses, ties


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self._f = open(filepath, "w")

    def write(self, *lines):
        for line in lines:
            print(line)
            self._f.write(line + "\n")

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare per-query ranks across RulE variants.")
    p.add_argument("--method", action="append", metavar="LABEL=PATH",
                   help="label=path/to/ranks.csv. Repeat for each method. "
                        "First method listed becomes the reference for pairwise comparisons.")
    p.add_argument("--out_dir", default="outputs/rank_analysis",
                   help="Directory for output files.")
    p.add_argument("--topk", type=int, default=10,
                   help="k for Hit@k statistics (default 10).")
    p.add_argument("--top_n_queries", type=int, default=20,
                   help="Number of best/worst queries to show in per-pair flip lists.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.method:
        print("ERROR: provide at least one --method label=path argument.", file=sys.stderr)
        sys.exit(1)

    # Parse label=path pairs
    methods = {}
    for spec in args.method:
        if "=" not in spec:
            print(f"ERROR: bad --method spec '{spec}', expected LABEL=PATH", file=sys.stderr)
            sys.exit(1)
        label, path = spec.split("=", 1)
        methods[label] = path

    method_names = list(methods.keys())
    reference    = method_names[0]

    os.makedirs(args.out_dir, exist_ok=True)
    out = _Tee(os.path.join(args.out_dir, "summary.txt"))

    out.write("=" * 70)
    out.write("  RulE per-query rank analysis")
    out.write("=" * 70)
    out.write(f"  Methods   : {', '.join(method_names)}")
    out.write(f"  Reference : {reference}  (used for pairwise comparisons)")
    out.write(f"  Hit@k     : k={args.topk}")
    out.write("")

    # Load all CSVs -------------------------------------------------------
    dfs = {}
    for label, path in methods.items():
        if not os.path.exists(path):
            out.write(f"WARNING: {label} -> {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        for col in ("L_rule", "H_rule", "L_kge", "H_kge", "L_fused", "H_fused"):
            if col not in df.columns:
                raise ValueError(f"{path} is missing column '{col}'.")
        df["rank_rule"]  = midpoint_rank(df["L_rule"],  df["H_rule"])
        df["rank_kge"]   = midpoint_rank(df["L_kge"],   df["H_kge"])
        df["rank_fused"] = midpoint_rank(df["L_fused"], df["H_fused"])
        dfs[label] = df
        out.write(f"Loaded {label}: {len(df)} queries  ({path})")

    loaded = list(dfs.keys())
    if not loaded:
        out.write("No valid CSVs loaded. Exiting.")
        out.close()
        sys.exit(1)
    out.write("")

    # KGE-only sanity check -----------------------------------------------
    # All models share the same frozen KGE embeddings, so kge-only ranks
    # should be identical across methods.
    out.write("-" * 70)
    out.write("KGE-only rank sanity (should be ~0 across methods)")
    ref_kge = dfs[loaded[0]].set_index(["h", "r", "t"])["rank_kge"]
    for label in loaded[1:]:
        other_kge = dfs[label].set_index(["h", "r", "t"])["rank_kge"]
        aligned   = ref_kge.align(other_kge, join="inner")
        max_diff  = float((aligned[0] - aligned[1]).abs().max())
        out.write(f"  max |rank_kge({loaded[0]}) - rank_kge({label})| = {max_diff:.3f}")
    out.write("")

    # ── Section 1: Per-method aggregate metrics ──────────────────────────
    out.write("=" * 70)
    out.write("  SECTION 1: Per-method aggregate metrics")
    out.write("=" * 70)

    metrics_table = {}
    for label in loaded:
        df = dfs[label]
        metrics_table[label] = {
            "rule-only":  aggregate_metrics(df["rank_rule"],  args.topk),
            "kge-only":   aggregate_metrics(df["rank_kge"],   args.topk),
            "fused":      aggregate_metrics(df["rank_fused"], args.topk),
        }

    for scoring in ("rule-only", "kge-only", "fused"):
        out.write(f"\n  {scoring.upper()}")
        header = f"  {'method':20s}  {'MRR':>7}  {'MR':>8}  {'Hit@1':>7}  {'Hit@3':>7}  {f'Hit@{args.topk}':>8}"
        out.write(header)
        out.write("  " + "-" * (len(header) - 2))
        for label in loaded:
            m = metrics_table[label][scoring]
            out.write(
                f"  {label:20s}  {m['MRR']:7.4f}  {m['MR']:8.1f}"
                f"  {m['Hit@1']:7.4f}  {m['Hit@3']:7.4f}  {m[f'Hit@{args.topk}']:8.4f}"
            )
    out.write("")

    # ── Section 2: Pairwise method comparison (rule-only and fused) ──────
    out.write("=" * 70)
    out.write("  SECTION 2: Pairwise method comparison")
    out.write("=" * 70)

    # Merge all methods on (h, r, t)
    merged = dfs[loaded[0]][["h", "r", "t", "grounded",
                               "rank_rule", "rank_kge", "rank_fused"]].copy()
    merged = merged.rename(columns={
        "rank_rule":  f"rank_rule_{loaded[0]}",
        "rank_kge":   f"rank_kge_{loaded[0]}",
        "rank_fused": f"rank_fused_{loaded[0]}",
        "grounded":   f"grounded_{loaded[0]}",
    })
    for label in loaded[1:]:
        right = dfs[label][["h", "r", "t", "grounded",
                             "rank_rule", "rank_kge", "rank_fused"]].rename(columns={
            "rank_rule":  f"rank_rule_{label}",
            "rank_kge":   f"rank_kge_{label}",
            "rank_fused": f"rank_fused_{label}",
            "grounded":   f"grounded_{label}",
        })
        merged = merged.merge(right, on=["h", "r", "t"], how="outer")

    for scoring in ("rule", "fused"):
        out.write(f"\n  --- {scoring.upper()} scoring ---")
        for label in loaded[1:]:
            col_ref   = f"rank_{scoring}_{reference}"
            col_other = f"rank_{scoring}_{label}"
            if col_ref not in merged.columns or col_other not in merged.columns:
                continue
            valid = merged[[col_ref, col_other]].dropna()
            w, l, t = wins_losses_ties(valid[col_other], valid[col_ref])
            n = len(valid)
            out.write(f"\n  {label} vs {reference}  (n={n})")
            out.write(f"    {label} wins (lower rank) : {w:5d}  ({100*w/n:5.1f}%)")
            out.write(f"    {label} loses             : {l:5d}  ({100*l/n:5.1f}%)")
            out.write(f"    ties                       : {t:5d}  ({100*t/n:5.1f}%)")

            delta = valid[col_other] - valid[col_ref]
            out.write(f"    mean rank delta ({label} - {reference}): {delta.mean():+.2f}")
            out.write(f"    (negative = {label} ranks higher on average)")

            # Top-N biggest improvements by this method
            improved = merged.nsmallest(args.top_n_queries, col_other)[[
                "h", "r", "t", col_ref, col_other
            ]].dropna()
            improved.columns = ["h", "r", "t", f"rank_{reference}", f"rank_{label}"]
            improved["delta"] = improved[f"rank_{label}"] - improved[f"rank_{reference}"]
            # Sort by biggest improvement (most negative delta)
            improved = improved.sort_values("delta").head(args.top_n_queries)
            if not improved.empty:
                out.write(f"\n    Top-{args.top_n_queries} most improved queries (h, r, t, ref_rank, {label}_rank, delta):")
                for _, row in improved.iterrows():
                    out.write(f"      h={int(row['h'])} r={int(row['r'])} t={int(row['t'])}"
                              f"  ref={row[f'rank_{reference}']:.1f}"
                              f"  {label}={row[f'rank_{label}']:.1f}"
                              f"  delta={row['delta']:+.1f}")
    out.write("")

    # ── Section 3: Rule vs KGE win analysis ──────────────────────────────
    out.write("=" * 70)
    out.write("  SECTION 3: Rule vs KGE win analysis")
    out.write("=" * 70)

    # kge-only ranks are identical; use reference method's kge column
    kge_col = f"rank_kge_{loaded[0]}"

    for label in loaded:
        col_rule  = f"rank_rule_{label}"
        col_fused = f"rank_fused_{label}"
        if col_rule not in merged.columns or col_fused not in merged.columns:
            continue

        sub = merged[["h", "r", "t", kge_col, col_rule, col_fused]].dropna()
        n   = len(sub)
        if n == 0:
            continue

        rule_beats_kge  = (sub[col_rule]  < sub[kge_col]).sum()
        fused_beats_kge = (sub[col_fused] < sub[kge_col]).sum()

        # Hit@k rescues: kge was outside top-k, fused brought it in
        rescues = ((sub[kge_col] > args.topk) & (sub[col_fused] <= args.topk)).sum()
        # Regressions: kge was inside top-k, fused pushed it out
        regressions = ((sub[kge_col] <= args.topk) & (sub[col_fused] > args.topk)).sum()

        out.write(f"\n  [{label}]  (n={n})")
        out.write(f"    rule-only beats kge-only          : {rule_beats_kge:5d}  ({100*rule_beats_kge/n:5.1f}%)")
        out.write(f"    fused beats kge-only              : {fused_beats_kge:5d}  ({100*fused_beats_kge/n:5.1f}%)")
        out.write(f"    Hit@{args.topk} rescues (kge miss -> fused hit)  : {rescues:5d}  ({100*rescues/n:5.1f}%)")
        out.write(f"    Hit@{args.topk} regressions (kge hit -> fused miss): {regressions:5d}  ({100*regressions/n:5.1f}%)")

    # Cross-method rule-beats-kge overlap
    out.write("\n  Cross-method rule-beats-kge (rule-only < kge-only):")
    beat_cols = []
    for label in loaded:
        col_rule = f"rank_rule_{label}"
        if col_rule in merged.columns:
            beat_col = f"beats_kge_{label}"
            merged[beat_col] = (merged[col_rule] < merged[kge_col])
            beat_cols.append((label, beat_col))

    if len(beat_cols) >= 2:
        all_beat = merged[[c for _, c in beat_cols]].dropna()
        shared_all  = all_beat.all(axis=1).sum()
        unique_each = {
            label: (all_beat[bc] & ~all_beat[[c for l, c in beat_cols if l != label]].any(axis=1)).sum()
            for label, bc in beat_cols
        }
        out.write(f"    Rules beat KGE in ALL methods     : {int(shared_all)}")
        for label, _ in beat_cols:
            out.write(f"    Unique to {label:20s}       : {int(unique_each[label])}")
    out.write("")

    # ── Section 4: Per-relation breakdown ────────────────────────────────
    out.write("=" * 70)
    out.write("  SECTION 4: Per-relation MRR (rule-only)")
    out.write("=" * 70)

    rel_rows = []
    for label in loaded:
        col_rule = f"rank_rule_{label}"
        if col_rule not in merged.columns:
            continue
        sub = merged[["r", col_rule]].dropna()
        grp = sub.groupby("r")[col_rule]
        rel_mrr = grp.apply(mrr_from_ranks).rename(f"mrr_{label}")
        rel_rows.append(rel_mrr)

    if rel_rows:
        per_rel = pd.concat(rel_rows, axis=1).reset_index()
        per_rel_path = os.path.join(args.out_dir, "per_relation.csv")
        per_rel.to_csv(per_rel_path, index=False)
        out.write(f"\n  Per-relation MRR written to {per_rel_path}")

        # Print the relations with the largest spread (most variation across methods)
        if len(rel_rows) > 1:
            mrr_cols = [f"mrr_{l}" for l in loaded if f"mrr_{l}" in per_rel.columns]
            per_rel["spread"] = per_rel[mrr_cols].max(axis=1) - per_rel[mrr_cols].min(axis=1)
            top_spread = per_rel.nlargest(10, "spread")
            out.write(f"\n  Top-10 relations with largest MRR spread across methods:")
            out.write(f"  {'rel':>6}  " + "  ".join(f"{l:>12}" for l in loaded if f"mrr_{l}" in per_rel.columns) + "  spread")
            for _, row in top_spread.iterrows():
                vals = "  ".join(f"{row.get(f'mrr_{l}', float('nan')):12.4f}" for l in loaded if f"mrr_{l}" in per_rel.columns)
                out.write(f"  {int(row['r']):6d}  {vals}  {row['spread']:.4f}")
    out.write("")

    # ── Save merged table ─────────────────────────────────────────────────
    merged_path = os.path.join(args.out_dir, "merged_ranks.csv")
    merged.to_csv(merged_path, index=False)
    out.write(f"Merged rank table written to {merged_path}")

    out.write("\nDone.")
    out.close()


if __name__ == "__main__":
    main()
