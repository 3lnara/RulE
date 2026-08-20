#!/usr/bin/env python3
"""
Rebuild and CACHE the leave-one-out logistic-regression design matrix.

rule_logreg_train.py builds this matrix, fits beta, then discards the matrix.
The interpretability analyses (collinearity, bootstrap stability) need the exact
firing matrix the betas were fit on. Because build_design_matrix is deterministic
given (data, rules), rebuilding it locally reproduces that matrix EXACTLY -- this
script rebuilds it, verifies a refit reproduces the saved beta (a parity gate),
and caches the sparse matrix + per-rule meta to design_matrix.pt for reuse.

Small datasets (UMLS/Family, N~3k) build fine on CPU in minutes. This is a
one-time cost; the downstream analyses read the cache, not the graph.

Outputs (in --out_dir, default = --logreg_dir):
  design_matrix.pt   {rows, cols, y [nnz/M], M, R, support, gold_fired,
                      head, length, meta}

Usage (from repo root):
    python scripts/build_design_matrix_cache.py \\
        --data_path  data/family \\
        --rule_file  data/family/mined_rules.txt \\
        --logreg_dir outputs/additive_family/logreg-family
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _SCRIPT_DIR)

import dump_rule_counts as drc
import rule_logreg_train as rlt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True)
    p.add_argument("--rule_file", default=None,
                   help="Defaults to <data_path>/mined_rules.txt.")
    p.add_argument("--logreg_dir", required=True,
                   help="Dir with rule_logreg.pt (for the parity check) and where "
                        "design_matrix.pt is written unless --out_dir is given.")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--chunk_size", type=int, default=0)
    p.add_argument("--parity_l2", type=float, default=None,
                   help="L2 to refit for the parity check (default: selected_l2 "
                        "from logreg_selection.json, else the saved meta l2).")
    p.add_argument("--parity_tol", type=float, default=1e-2,
                   help="Max |refit-saved| beta diff tolerated by the parity gate.")
    p.add_argument("--skip_parity", action="store_true")
    return p.parse_args()


def _dump_parity_diagnostics(diff_vec, beta, ref, fired_total, gold_fired, out_dir,
                             min_support, top_n=20):
    """On a parity-gate failure, dump the rules with the largest |refit-saved| beta
    divergence. A gap concentrated on low-support / (quasi-)separated rules points
    to numerical fragility from weak L2 on an ill-conditioned fit (the coefficient
    is barely pinned down to begin with, so float rounding -- e.g. GPU sparse-matmul
    reduction order -- moves it), not a genuine design-matrix mismatch: those would
    show up as loss/corr divergence too, spread across many well-supported rules."""
    R = diff_vec.numel()
    k = min(top_n, R)
    top_idx = torch.topk(diff_vec, k).indices.tolist()
    rows = []
    print(f"\n  Top {k} rules by |refit-saved| beta divergence:")
    print(f"  {'rule_id':>8} {'diff':>10} {'refit_beta':>12} {'saved_beta':>12} "
          f"{'support':>8} {'gold_fired':>10} {'separated':>10}")
    for i in top_idx:
        support_i = int(fired_total[i])
        gold_i = int(gold_fired[i])
        separated = support_i > 0 and (gold_i == 0 or gold_i == support_i)
        print(f"  {i:>8} {float(diff_vec[i]):>10.4f} {float(beta[i]):>12.4f} "
              f"{float(ref[i]):>12.4f} {support_i:>8} {gold_i:>10} {str(separated):>10}")
        rows.append(dict(rule_id=i, diff=float(diff_vec[i]), refit_beta=float(beta[i]),
                          saved_beta=float(ref[i]), support=support_i,
                          gold_fired=gold_i, separated=separated))
    out_csv = os.path.join(out_dir, "parity_failure_diagnostics.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule_id", "diff", "refit_beta", "saved_beta",
                                            "support", "gold_fired", "separated"])
        w.writeheader(); w.writerows(rows)
    n_sep = sum(1 for r in rows if r["separated"])
    n_low = sum(1 for r in rows if r["support"] < min_support)
    print(f"\n  {n_sep}/{k} divergent rules are (quasi-)separated; "
          f"{n_low}/{k} have support < {min_support}.")
    print(f"  Wrote {out_csv}")


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    out_dir = args.out_dir or args.logreg_dir
    os.makedirs(out_dir, exist_ok=True)
    rule_file = args.rule_file or os.path.join(args.data_path, "mined_rules.txt")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    print(f"data_path : {args.data_path}")
    print(f"rule_file : {rule_file}")
    print(f"out_dir   : {out_dir}")
    print(f"device    : {device}")

    print("\nLoading graph + rules...")
    graph = drc.MinimalGraph(args.data_path)
    rules = drc.load_rules(rule_file, graph.relation_size)
    R = len(rules)
    heads_meta   = np.array([r[1]       for r in rules], dtype=np.int64)
    lengths_meta = np.array([len(r[2:]) for r in rules], dtype=np.int64)
    print(f"  entities={graph.entity_size}  rules={R}  train_facts={len(graph.train_facts)}")

    gold = rlt.build_gold(graph)

    print("\nBuilding leave-one-out design matrix...")
    t0 = time.time()
    rows, cols, y, fired_total, gold_fired, M = rlt.build_design_matrix(
        graph, rules, gold, device=device, chunk_size=args.chunk_size)
    print(f"  done in {time.time()-t0:.1f}s")

    # ---- Parity gate: a refit must reproduce the saved beta ------------------
    saved_pt = os.path.join(args.logreg_dir, "rule_logreg.pt")
    if not args.skip_parity and os.path.isfile(saved_pt):
        saved = torch.load(saved_pt, map_location="cpu")
        l2 = args.parity_l2
        if l2 is None:
            sel_json = os.path.join(args.logreg_dir, "logreg_selection.json")
            if os.path.isfile(sel_json):
                import json
                l2 = float(json.load(open(sel_json))["selected_l2"])
            else:
                l2 = float(saved.get("meta", {}).get("l2", 1e-4))
        print(f"\nParity check: refit at L2={l2:g} vs saved beta...")
        beta, _ = rlt.fit_logreg(rows, cols, y, M, R, device=device,
                                 l2=l2, max_iter=200)
        # saved 'beta' is at primary l2; if selected differs, compare to beta_grid row
        ref = None
        if "beta_grid" in saved and "l2_grid" in saved:
            l2g = [float(v) for v in saved["l2_grid"].tolist()]
            if any(abs(v - l2) <= 1e-12 for v in l2g):
                gi = min(range(len(l2g)), key=lambda i: abs(l2g[i] - l2))
                ref = saved["beta_grid"][gi].float()
        if ref is None:
            ref = saved["beta"].float()
        diff_vec = (beta - ref).abs()
        diff = float(diff_vec.max())
        corr = float(np.corrcoef(beta.numpy(), ref.numpy())[0, 1])
        supp_ok = bool((saved["support"].numpy() == fired_total).all()) \
            if "support" in saved else True
        print(f"  max|refit-saved|={diff:.3e}  corr={corr:.8f}  support_match={supp_ok}")
        if diff > args.parity_tol or not supp_ok:
            print(f"  ERROR: parity gate FAILED (tol={args.parity_tol:g}). The rebuilt "
                  f"matrix does not match the fitted model; aborting.")
            _dump_parity_diagnostics(diff_vec, beta, ref, fired_total, gold_fired,
                                     out_dir, min_support=20)
            sys.exit(1)
        print("  parity OK.")
    else:
        print("\nParity check skipped.")

    out_pt = os.path.join(out_dir, "design_matrix.pt")
    torch.save({
        "rows": rows, "cols": cols, "y": y, "M": int(M), "R": int(R),
        "support": torch.tensor(fired_total, dtype=torch.long),
        "gold_fired": torch.tensor(gold_fired, dtype=torch.long),
        "head": torch.tensor(heads_meta, dtype=torch.long),
        "length": torch.tensor(lengths_meta, dtype=torch.long),
        "meta": {"data_path": args.data_path, "rule_file": rule_file,
                 "num_rows": int(M), "num_rules": int(R)},
    }, out_pt)
    print(f"\nSaved {out_pt}  (M={M}, R={R}, nnz={rows.numel()})")


if __name__ == "__main__":
    main()
