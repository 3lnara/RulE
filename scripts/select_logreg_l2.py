#!/usr/bin/env python3
"""
Pick the best L2 for the logreg-binary aggregator from a set of PER-L2 rank
dumps -- pure rule-only selection, no KGE fusion involved.

Each L2 in --l2_grid is expected to have been fit + evaluated separately
(scripts/rule_logreg_train.py --l2 <v>, then src_additive/main.py
--logreg_binary --dump_ranks) into its own subdirectory

    <logreg_root>/l2_<v>/ranks_valid.csv
    <logreg_root>/l2_<v>/ranks_test.csv

where <v> is the EXACT string token passed on --l2_grid (e.g. "1e-4"), taken
verbatim -- not reformatted -- so it matches whatever directory name the
driving shell script used (bash "1e-5" and Python's float reformatting of
1e-5, e.g. "1e-05", do NOT agree, hence keeping this a plain string). Each
ranks_<split>.csv already has
the per-query filtered (L, H) band trainer.evaluate produced for that L2 (see
src_additive/trainer.py:_dump_ranks), so MRR/Hit@ are recomputed directly from
those columns -- no counts_<split>.pt / kge_<split>.pt / model reload needed.

Selection: L2 is chosen on --select_split (default valid) MRR; the same L2's
metrics are then reported on --report_split (default test).

Outputs (in --logreg_root):
  logreg_l2_sweep.csv       one row per (l2, split): MRR/Hit@/MR/n
  logreg_l2_selection.json  selected l2 + valid/test metrics

Usage (from repo root):
    python scripts/select_logreg_l2.py \\
        --logreg_root outputs/additive_family/logreg \\
        --l2_grid 1e-5 1e-4 1e-3 1e-2 1e-1 1 \\
        --select_split valid --report_split test
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def expectation_metrics(results: List[Tuple[int, int, int, int, int]]) -> Dict[str, float]:
    """Replicate trainer.evaluate's expectation-over-[L, H) metrics."""
    h1 = h3 = h10 = mr = mrr = 0.0
    n = len(results)
    for (_, _, _, L, H) in results:
        span = H - L
        for rank in range(L, H):
            w = 1.0 / span
            if rank <= 1:  h1  += w
            if rank <= 3:  h3  += w
            if rank <= 10: h10 += w
            mr  += w * rank
            mrr += w / rank
    if n == 0:
        return {}
    return {
        "MRR": mrr / n, "MR": mr / n,
        "Hit@1": h1 / n, "Hit@3": h3 / n, "Hit@10": h10 / n,
        "n": n,
    }


def load_ranks_csv(path: str) -> List[Tuple[int, int, int, int, int]]:
    results = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            results.append((int(row["h"]), int(row["r"]), int(row["t"]),
                             int(row["L"]), int(row["H"])))
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logreg_root", required=True,
                   help="Dir holding one l2_<v>/ subdirectory per L2 value.")
    p.add_argument("--l2_grid", type=str, nargs="+", required=True,
                   help="L2 values to compare, as the EXACT string tokens used "
                        "to name each l2_<v> subdirectory (e.g. 1e-5 1e-4 ...).")
    p.add_argument("--select_split", default="valid",
                   help="Split whose MRR is maximised for selection.")
    p.add_argument("--report_split", default="test",
                   help="Split reported at the selected L2.")
    p.add_argument("--splits", nargs="+", default=["valid", "test"],
                   help="Splits to score/write in the sweep CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.logreg_root, exist_ok=True)

    rows = []
    metrics_by_l2 = {}  # l2 token (str) -> {split: metrics_dict}
    for l2 in args.l2_grid:
        variant_dir = os.path.join(args.logreg_root, f"l2_{l2}")
        metrics_by_l2[l2] = {}
        for split in args.splits:
            ranks_path = os.path.join(variant_dir, f"ranks_{split}.csv")
            if not os.path.isfile(ranks_path):
                raise FileNotFoundError(
                    f"L2={l2}: missing {ranks_path}. Run rule_logreg_train.py "
                    f"--l2 {l2} + main.py --logreg_binary --dump_ranks into "
                    f"{variant_dir} first.")
            results = load_ranks_csv(ranks_path)
            m = expectation_metrics(results)
            metrics_by_l2[l2][split] = m
            rows.append({
                "l2": float(l2), "split": split,
                "MRR": round(m["MRR"], 6), "Hit@1": round(m["Hit@1"], 6),
                "Hit@3": round(m["Hit@3"], 6), "Hit@10": round(m["Hit@10"], 6),
                "MR": round(m["MR"], 4), "n": m["n"],
            })
            print(f"  L2={l2:<8} [{split}] MRR={m['MRR']:.6f}  "
                  f"Hit@1={m['Hit@1']:.6f}  Hit@10={m['Hit@10']:.6f}  n={m['n']}")

    if args.select_split not in args.splits:
        raise ValueError(f"--select_split {args.select_split} not in --splits {args.splits}")

    l2_best = max(args.l2_grid, key=lambda l2: metrics_by_l2[l2][args.select_split]["MRR"])
    sel_mrr = metrics_by_l2[l2_best][args.select_split]["MRR"]

    print("\n" + "=" * 70)
    print(f"L2 selection on {args.select_split} MRR:")
    for l2 in args.l2_grid:
        mark = "  <-- selected" if l2 == l2_best else ""
        print(f"    L2={l2:<8}  MRR={metrics_by_l2[l2][args.select_split]['MRR']:.6f}{mark}")
    print(f"SELECTED : L2={l2_best}  ({args.select_split} MRR {sel_mrr:.6f})")

    selection = {
        "selected_l2": float(l2_best),
        "select_split": args.select_split,
        "l2_grid": [float(v) for v in args.l2_grid],
        f"{args.select_split}_metrics": metrics_by_l2[l2_best][args.select_split],
    }
    if args.report_split in metrics_by_l2[l2_best]:
        rep_m = metrics_by_l2[l2_best][args.report_split]
        selection[f"{args.report_split}_metrics"] = rep_m
        print(f"REPORT   on {args.report_split} at L2={l2_best}: "
              f"MRR={rep_m['MRR']:.6f}  Hit@1={rep_m['Hit@1']:.6f}  "
              f"Hit@10={rep_m['Hit@10']:.6f}")
    print("=" * 70)

    sweep_path = os.path.join(args.logreg_root, "logreg_l2_sweep.csv")
    with open(sweep_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["l2", "split", "MRR", "Hit@1",
                                           "Hit@3", "Hit@10", "MR", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {sweep_path}  ({len(rows)} rows)")

    sel_path = os.path.join(args.logreg_root, "logreg_l2_selection.json")
    with open(sel_path, "w") as fh:
        json.dump(selection, fh, indent=2)
    print(f"Wrote {sel_path}\nDone.")


if __name__ == "__main__":
    main()
