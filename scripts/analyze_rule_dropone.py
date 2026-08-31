#!/usr/bin/env python3
"""Drop-one-rule marginal importance (delta-MRR) for the logreg aggregator.

Pure offline
arithmetic over the dumped counts + selected beta; scoring matches
score_counts_offline / trainer.evaluate. Sign: importance = MRR_baseline -
MRR_without_R (positive => R helps).

Usage:
    python scripts/analyze_rule_dropone.py --logreg_dir outputs/<ds>/rq \\
        --analysis_dir outputs/<ds>/rq --data_path data/<ds> --split valid
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _SCRIPT_DIR)

import score_counts_offline as sco


def mrr_of(L, H):
    """Expected reciprocal rank over the tie span [L, H), matching evaluate."""
    span = H - L
    if span <= 0:
        return 0.0
    s = 0.0
    for rank in range(L, H):
        s += (1.0 / span) / rank
    return s


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logreg_dir", required=True)
    p.add_argument("--analysis_dir", required=True,
                   help="Dir with counts_<split>.pt.")
    p.add_argument("--data_path", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--split", default="valid")
    p.add_argument("--clamp", action="store_true",
                   help="Clamp negative betas to 0 (default: keep, matching "
                        "in-model --logreg_binary).")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    out_dir = args.out_dir or args.logreg_dir

    # Selected beta.
    sel_pt = os.path.join(args.logreg_dir, "rule_logreg_selected.pt")
    lr_pt  = os.path.join(args.logreg_dir, "rule_logreg.pt")
    src = sel_pt if os.path.isfile(sel_pt) else lr_pt
    beta = torch.load(src, map_location="cpu")["beta"].double().numpy()
    if args.clamp:
        beta = np.clip(beta, 0.0, None)
    R = beta.shape[0]

    # Counts + sizes + filter.
    c = torch.load(os.path.join(args.analysis_dir, f"counts_{args.split}.pt"),
                   map_location="cpu")
    N, num_rel = sco._load_entity_rel_sizes(args.data_path)
    hr2ooo = sco.build_hr2ooo(args.data_path, N, num_rel)

    query_h    = c["query_h"].numpy()
    query_r    = c["query_r"].numpy()
    query_gold = c["query_gold"].numpy()
    q_ptr      = c["q_ptr"].numpy()
    rule_id    = c["rule_id"].numpy()
    ent        = c["ent"].numpy()
    Q = query_h.shape[0]

    delta_sum   = np.zeros(R)
    n_q_fired   = np.zeros(R, dtype=np.int64)
    eval_supp   = np.zeros(R, dtype=np.int64)   # entity-firings over eval split
    mrr_base_sum = 0.0

    isO = np.zeros(N, dtype=bool)

    for qi in range(Q):
        h, r, gold = int(query_h[qi]), int(query_r[qi]), int(query_gold[qi])
        a, b = int(q_ptr[qi]), int(q_ptr[qi + 1])
        rid = rule_id[a:b]
        ents = ent[a:b]

        sc = np.zeros(N)
        np.add.at(sc, ents, beta[rid])
        val = sc[gold]

        # Filter set O (known trues incl. gold).
        O = hr2ooo.get(r * N + h, set())
        for t in O:
            isO[t] = True

        grounded = np.unique(ents)
        gcomp = grounded[~isO[grounded]]
        gc = np.sort(sc[gcomp])
        n_gc = gc.shape[0]
        U = N - len(O) - n_gc                      # ungrounded competitors

        ge = (n_gc - np.searchsorted(gc, val, "left"))  + (U if val <= 0 else 0)
        gt = (n_gc - np.searchsorted(gc, val, "right")) + (U if val <  0 else 0)
        Lb, Hb = gt + 1, ge + 2
        mrr_b = mrr_of(Lb, Hb)
        mrr_base_sum += mrr_b

        # Group entries by rule.
        order = np.argsort(rid, kind="stable")
        rid_o = rid[order]; ent_o = ents[order]
        bounds = np.searchsorted(rid_o, np.arange(R + 1))

        # Distinct rules present in this query.
        present = np.unique(rid_o)
        for R_ in present:
            s0, s1 = bounds[R_], bounds[R_ + 1]
            eR = ent_o[s0:s1]
            bR = beta[R_]
            eval_supp[R_] += eR.shape[0]
            n_q_fired[R_] += 1

            gold_fired = bool(np.isin(gold, eR))
            delta_g = bR if gold_fired else 0.0
            new_val = val - delta_g

            eR_comp = eR[~isO[eR]]
            sR = sc[eR_comp]

            ge_all = n_gc - np.searchsorted(gc, new_val, "left")
            gt_all = n_gc - np.searchsorted(gc, new_val, "right")
            new_ge_g = ge_all - int((sR >= new_val).sum()) + int(((sR - bR) >= new_val).sum())
            new_gt_g = gt_all - int((sR >  new_val).sum()) + int(((sR - bR) >  new_val).sum())
            new_ge = new_ge_g + (U if new_val <= 0 else 0)
            new_gt = new_gt_g + (U if new_val <  0 else 0)
            newL, newH = new_gt + 1, new_ge + 2
            delta_sum[R_] += mrr_of(newL, newH) - mrr_b

        for t in O:
            isO[t] = False

    mrr_base = mrr_base_sum / Q
    delta_mrr = delta_sum / Q

    print(f"split={args.split}  Q={Q}  baseline rule-only MRR = {mrr_base:.6f}")
    sel_json = os.path.join(args.logreg_dir, "logreg_selection.json")
    if os.path.isfile(sel_json):
        import json
        j = json.load(open(sel_json))
        ref = j.get("valid_rule_only_MRR")
        if ref is not None and args.split == "valid":
            print(f"  (logreg_selection.json valid_rule_only_MRR = {ref:.6f}; "
                  f"diff = {abs(ref - mrr_base):.2e})")

    out_csv = os.path.join(out_dir, f"rule_dropone_{args.split}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule_id", "beta", "eval_support",
                                           "n_queries_fired", "delta_mrr",
                                           "importance"])
        w.writeheader()
        for r_ in range(R):
            w.writerow(dict(rule_id=r_, beta=f"{beta[r_]:.6f}",
                            eval_support=int(eval_supp[r_]),
                            n_queries_fired=int(n_q_fired[r_]),
                            delta_mrr=f"{delta_mrr[r_]:.6e}",
                            importance=f"{-delta_mrr[r_]:.6e}"))
    print(f"Wrote {out_csv}")

    # Quick top movers for a sanity glance.
    order = np.argsort(delta_mrr)  # most negative delta = most helpful
    print("\nTop 5 most HELPFUL rules (removing them hurts MRR most):")
    for r_ in order[:5]:
        print(f"  rule {r_:6d}  beta={beta[r_]:+.4f}  importance={-delta_mrr[r_]:+.3e}  "
              f"nq={n_q_fired[r_]}")
    print("Top 5 most HARMFUL rules (removing them helps MRR most):")
    for r_ in order[::-1][:5]:
        print(f"  rule {r_:6d}  beta={beta[r_]:+.4f}  importance={-delta_mrr[r_]:+.3e}  "
              f"nq={n_q_fired[r_]}")


if __name__ == "__main__":
    main()
