#!/usr/bin/env python3
"""
Compute per-rule empirical precision on the LEAVE-ONE-OUT (LOO) design matrix.

This is the apples-to-apples counterpart of scripts/rule_precision_train.py for
comparing precision_binary against the learned logreg_binary aggregator.

WHY A SEPARATE SCRIPT
---------------------------------------------------------------------------
rule_precision_train.py grounds on the FULL train graph (no leave-one-out):

    precision_R = #{(h,t): body fires h->t on full graph, t in gold[r][h]}
                / #{(h,t): body fires h->t on full graph, h in gold[r].keys()}

scripts/rule_logreg_train.py instead fits beta_R on an EDGE-SPECIFIC leave-one-
out design: a positive (h, t) is grounded with exactly the answer edge (h, r, t)
removed (matching src_additive/data.py's edges_to_remove), while negatives use
plain full-graph grounding. Comparing that logreg to the full-graph precision
conflates the weighting method (precision vs fitted coefficient) with the
grounding regime (full-graph vs LOO). The most conspicuous artifact: the
length-1 identity rule (body = head relation) fires on every gold tail via its
own edge under full-graph grounding -> precision 1.0, but contributes nothing
under LOO.

This script computes precision on the SAME LOO design the logreg fit uses, so
the only difference left between the two weightings is the aggregation method.
It literally reuses build_design_matrix() from rule_logreg_train.py, then sets

    precision_R = gold_fired[R] / fired_total[R]

on that LOO design (fired_total = pos+neg rows the rule fires in; gold_fired =
positive rows). Rules that never fire get precision = nan -> 0.0 in-model,
exactly like the full-graph precision.

OUTPUT (schema-compatible with rule_precision_train.py)
---------------------------------------------------------------------------
  rule_precision.pt          tensors indexed by rule_id:
                               precision, support, gold_fired, head, length,
                               w_R_unclamped (nan placeholder), plus a `meta`
                               dict with leave_one_out=True.
  rule_precision_loo.csv      per-rule table (one row per rule)

Load it in-model exactly like the full-graph precision:
    src_additive/main.py --precision_binary --precision_file <out_dir>/rule_precision.pt

Usage (from repo root):
    python scripts/rule_precision_loo_train.py \\
        --data_path data/family \\
        --rule_file data/family/mined_rules.txt \\
        --out_dir   outputs/additive_family/precision_loo \\
        --device    cuda
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bootstrap: reuse the LOO design-matrix builder + gold dict from the logreg
# trainer (single source of truth for the leave-one-out grounding), and
# MinimalGraph/load_rules from dump_rule_counts.py.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

sys.path.insert(0, _SCRIPT_DIR)
import dump_rule_counts as _drc
import rule_logreg_train as _lr

MinimalGraph        = _drc.MinimalGraph
load_rules          = _drc.load_rules
build_gold          = _lr.build_gold
build_design_matrix = _lr.build_design_matrix


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_path", default="data/umls",
                   help="Dataset directory (entities.dict, relations.dict, train.txt, ...).")
    p.add_argument("--rule_file", default=None,
                   help="mined_rules.txt (defaults to <data_path>/mined_rules.txt).")
    p.add_argument("--out_dir", default="outputs/additive_umls_aggregation_2x2/precision_loo",
                   help="Where to write rule_precision.pt and rule_precision_loo.csv.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for the LOO grounding. Falls back to cpu if CUDA "
                        "is unavailable.")
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Process each relation's known-subject heads in blocks "
                        "of this size to bound the [B, N] mask/message memory "
                        "(useful for WN18RR/FB15k). 0 = no chunking (default).")
    p.add_argument("--top_k", type=int, default=20,
                   help="How many highest-precision (support>0) rules to print.")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    rule_file = args.rule_file or os.path.join(args.data_path, "mined_rules.txt")
    dataset   = os.path.basename(os.path.normpath(args.data_path))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  WARNING: --device cuda requested but CUDA is unavailable; "
              "falling back to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"dataset    : {dataset}")
    print(f"data_path  : {args.data_path}")
    print(f"rule_file  : {rule_file}")
    print(f"out_dir    : {args.out_dir}")
    print(f"device     : {device}  chunk_size: {args.chunk_size or 'none'}")
    print("grounding  : LEAVE-ONE-OUT (edge-specific, matches logreg design)")

    # ---- Graph + rules -------------------------------------------------------
    print("\nLoading graph...")
    graph = MinimalGraph(args.data_path)
    NR = graph.relation_size
    N  = graph.entity_size
    print(f"  entities={N}  directed_relations={NR * 2}  "
          f"train_facts={len(graph.train_facts)}")

    print("Loading rules...")
    rules = load_rules(rule_file, NR)
    R = len(rules)
    print(f"  num_rules={R}")

    id_to_body   = {rule[0]: rule[2:] for rule in rules}
    lengths_meta = np.array([len(r[2:]) for r in rules], dtype=np.int64)
    heads_meta   = np.array([r[1]       for r in rules], dtype=np.int64)

    # ---- Gold dict -----------------------------------------------------------
    print("\nBuilding gold dict from train facts...")
    gold = build_gold(graph)
    total_known = sum(len(v) for v in gold.values())
    print(f"  unique (r, h) pairs with >=1 gold tail: {total_known}")

    # ---- LOO design matrix ---------------------------------------------------
    # We only need the per-rule counts (fired_total, gold_fired); the sparse
    # rows/cols/y (the logreg fit inputs) are discarded. Reusing the same
    # builder guarantees the precision below shares the logreg's exact
    # leave-one-out grounding regime.
    print("\nBuilding leave-one-out design matrix (grounding from known heads)...")
    _rows, _cols, _y, fired_total, gold_fired, M = build_design_matrix(
        graph, rules, gold, device=device, chunk_size=args.chunk_size)

    # ---- Precision on the LOO design -----------------------------------------
    support   = fired_total
    precision = np.where(support > 0,
                         gold_fired / support.astype(np.float64),
                         float("nan")).astype(np.float32)

    has_prec   = np.isfinite(precision)
    n_positive = int(has_prec.sum())
    if n_positive > 0:
        pv = precision[has_prec]
        print(f"\n  Rules with support>0 : {n_positive}/{R}")
        print(f"  Precision (support>0)  "
              f"range=[{pv.min():.4f}, {pv.max():.4f}]  "
              f"mean={pv.mean():.4f}  median={np.median(pv):.4f}")
    else:
        print("\n  WARNING: no rule fired on any LOO row. Check data_path/rule_file.")

    # ---- Top-precision rules for inspection ----------------------------------
    if args.top_k > 0 and n_positive > 0:
        # Rank by precision, break ties by support (a precise-but-rare rule is
        # less trustworthy than a precise frequent one).
        order = sorted((g for g in range(R) if support[g] > 0),
                       key=lambda g: (precision[g], support[g]), reverse=True)
        shown = order[:args.top_k]
        print(f"\n  Top {len(shown)} rules by LOO precision:")
        print(f"    {'rule_id':>7}  {'head':>4}  {'len':>3}  {'prec':>7}  "
              f"{'support':>8}  {'gold':>6}  body")
        for gid in shown:
            body = " ".join(str(x) for x in id_to_body.get(gid, []))
            print(f"    {gid:>7}  {int(heads_meta[gid]):>4}  "
                  f"{int(lengths_meta[gid]):>3}  {precision[gid]:>7.3f}  "
                  f"{int(support[gid]):>8}  {int(gold_fired[gid]):>6}  {body}")

    # ---- Save rule_precision.pt (schema-compatible with the full-graph one) --
    # w_R_unclamped is a nan placeholder: the in-model --precision_binary path
    # reads only `precision`, and there is no confidence correlation to report
    # here. `meta.leave_one_out` distinguishes this file from the full-graph
    # rule_precision.pt at a glance.
    out_pt = os.path.join(args.out_dir, "rule_precision.pt")
    torch.save({
        "precision":     torch.tensor(precision,    dtype=torch.float32),
        "support":       torch.tensor(support,      dtype=torch.long),
        "gold_fired":    torch.tensor(gold_fired,   dtype=torch.long),
        "head":          torch.tensor(heads_meta,   dtype=torch.long),
        "length":        torch.tensor(lengths_meta, dtype=torch.long),
        "w_R_unclamped": torch.full((R,), float("nan"), dtype=torch.float32),
        "meta": {
            "dataset":       dataset,
            "num_rules":     R,
            "num_rows":      M,
            "leave_one_out": True,
        },
    }, out_pt)
    print(f"\nSaved {out_pt}")

    # ---- Per-rule CSV --------------------------------------------------------
    out_csv = os.path.join(args.out_dir, "rule_precision_loo.csv")
    fieldnames = ["rule_id", "head", "length", "body",
                  "fired_total", "gold_fired", "precision"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for gid in range(R):
            prec = float(precision[gid])
            body = id_to_body.get(gid, [])
            writer.writerow({
                "rule_id":     gid,
                "head":        int(heads_meta[gid]),
                "length":      int(lengths_meta[gid]),
                "body":        " ".join(str(x) for x in body),
                "fired_total": int(support[gid]),
                "gold_fired":  int(gold_fired[gid]),
                "precision":   f"{prec:.6f}" if math.isfinite(prec) else "nan",
            })
    print(f"Saved {out_csv}  ({R} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
