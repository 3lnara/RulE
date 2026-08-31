#!/usr/bin/env python3
"""Offline (L2 x alpha) sweep + two-stage model selection for the logreg
aggregator fused with the frozen KGE score.

Pure tensor arithmetic over rule_logreg.pt (beta_grid + l2_grid), counts_<split>.pt
and kge_<split>.pt; the combined score is
`sum_R beta_g[R] * 1[rule fired] + alpha * kge`, ranked with the same filtered L/H
expectation as the rest of the repo. Writes
logreg_alpha_sweep_<split>.csv and logreg_selection.json.

Usage:
    python scripts/select_logreg.py --logreg_file outputs/<ds>/rq/rule_logreg.pt \\
        --analysis_dir outputs/<ds>/rq --data_path data/<ds> \\
        --alpha_grid 0 0.5 1 2 3 5 --splits valid test
"""

import argparse
import csv
import json
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _SCRIPT_DIR)

# Reuse the exact filter + metric definitions used by the other offline scorer
# so results are directly comparable to precision_binary etc.
import score_counts_offline as _sco
build_hr2ooo        = _sco.build_hr2ooo
expectation_metrics = _sco.expectation_metrics
_load_sizes         = _sco._load_entity_rel_sizes


def eval_g(counts, kge, w_R, alpha_items, hr2ooo, N):
    """Rank every query for one weight vector w_R across alpha_items = [(ai, alpha)],
    combined score rule(w_R) + alpha*kge (binary activation, like --logreg_binary).
    Returns {ai: metrics_dict}."""
    query_h    = counts["query_h"]
    query_r    = counts["query_r"]
    query_gold = counts["query_gold"]
    q_ptr      = counts["q_ptr"]
    rule_id    = counts["rule_id"]
    ent        = counts["ent"]
    Q = query_h.size(0)

    # KGE is only consulted for alpha != 0. In rule-only mode (kge is None) every
    # requested alpha must be 0, so kge is never touched.
    need_kge = any(a != 0.0 for _, a in alpha_items)
    if need_kge and kge is None:
        raise ValueError("eval_g: alpha != 0 requested but no KGE scores loaded "
                         "(rule-only mode). Pass kge or restrict to alpha=0.")

    results = {ai: [] for ai, _ in alpha_items}
    for qi in range(Q):
        h    = int(query_h[qi]);  r = int(query_r[qi]);  gold = int(query_gold[qi])
        start = int(q_ptr[qi]);   stop = int(q_ptr[qi + 1])

        # Rule score once (reused across all alphas).
        rule_score = torch.zeros(N, dtype=torch.float64)
        if start < stop:
            w_vals = w_R[rule_id[start:stop]].double()          # binary basis
            rule_score.scatter_add_(0, ent[start:stop], w_vals)

        # Filter once (reused across all alphas).
        flag = torch.ones(N, dtype=torch.bool)
        for t in hr2ooo.get(r * N + h, set()):
            flag[t] = False                                     # includes gold itself

        kge_row = kge[qi].double() if need_kge else None
        for ai, alpha in alpha_items:
            score = rule_score if alpha == 0.0 else rule_score + alpha * kge_row
            val   = score[gold].item()
            n_gt  = int((score[flag] > val).sum().item())
            n_ge  = int((score[flag] >= val).sum().item())
            results[ai].append((h, r, gold, n_gt + 1, n_ge + 2))

    return {ai: expectation_metrics(results[ai]) for ai in results}


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline (L2 x alpha) sweep + selection for the logreg "
                    "aggregator fused with KGE.")
    p.add_argument("--logreg_file", required=True,
                   help="rule_logreg.pt with beta_grid + l2_grid "
                        "(scripts/rule_logreg_train.py --l2_grid).")
    p.add_argument("--analysis_dir", required=True,
                   help="Dir holding counts_<split>.pt and kge_<split>.pt.")
    p.add_argument("--data_path", default="data/umls",
                   help="Dataset dir (for entity/relation sizes + hr2ooo filter).")
    p.add_argument("--out_dir", default=None,
                   help="Where to write outputs (default: --analysis_dir).")
    p.add_argument("--alpha_grid", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
                   help="KGE fusion weights to sweep. Include 0 for rule-only. "
                        "Ignored in rule-only mode (--no_kge or missing kge_*.pt).")
    p.add_argument("--no_kge", action="store_true", default=False,
                   help="RULE-ONLY mode: skip KGE fusion entirely. L2 is selected "
                        "on rule-only validation MRR and reported at alpha=0, which "
                        "reproduces the in-model --logreg_binary ranks EXACTLY (the "
                        "in-model KGE fusion is disabled, so a fused number would "
                        "not correspond to any ranks_*.csv). Auto-enabled if "
                        "kge_<split>.pt is absent. KGE fusion is an optional, "
                        "separate experiment; this is the canonical headline path.")
    p.add_argument("--select_split", default="valid",
                   help="Split whose combined MRR is maximised for selection.")
    p.add_argument("--report_split", default="test",
                   help="Split reported at the selected (L2, alpha).")
    p.add_argument("--splits", nargs="+", default=["valid", "test"],
                   help="Splits to score/write in the sweep CSV.")
    p.add_argument("--full_grid", action="store_true", default=False,
                   help="Evaluate the ENTIRE L2 x alpha grid on every split for "
                        "the inspection CSV. Default (off): evaluate only what the "
                        "two-stage selection needs -- the alpha=0 column for the "
                        "L2 stage, then all alphas at the selected L2 -- which is "
                        "~(G+A) instead of G*A evaluations (much faster on "
                        "WN18RR/FB15k). Selection is identical either way.")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    out_dir = args.out_dir or args.analysis_dir
    os.makedirs(out_dir, exist_ok=True)

    N, num_rel = _load_sizes(args.data_path)
    print(f"data_path : {args.data_path}  N={N}  num_rel={num_rel}")

    lr = torch.load(args.logreg_file, map_location="cpu")
    if "beta_grid" in lr:
        beta_grid = lr["beta_grid"].float()                 # [G, R]
        l2_grid   = [float(v) for v in lr["l2_grid"].tolist()]
    else:  # single-beta file: treat as a 1-row grid
        beta_grid = lr["beta"].float().unsqueeze(0)
        l2_grid   = [float(lr.get("meta", {}).get("l2", float("nan")))]
    G, R = beta_grid.shape
    # Per-L2 intercepts (constant per query, rank-invariant; carried into the
    # selected-beta file only so the in-model re-dump is exactly reproducible).
    if "intercept_grid" in lr:
        intercept_grid = [float(v) for v in lr["intercept_grid"].float().tolist()]
    else:
        intercept_grid = [float(lr.get("intercept", 0.0))] * G

    # RULE-ONLY mode: explicit (--no_kge) or forced when any split lacks
    # kge_<split>.pt. Then alpha is pinned to 0 (which reproduces the in-model
    # --logreg_binary ranks exactly) and no KGE is ever loaded.
    have_kge = all(os.path.isfile(os.path.join(args.analysis_dir, f"kge_{s}.pt"))
                   for s in args.splits)
    rule_only_mode = args.no_kge or not have_kge
    if rule_only_mode and not args.no_kge:
        print("rule-only mode: kge_<split>.pt not found for every split -> "
              "skipping KGE fusion (alpha pinned to 0).")

    # alpha=0 (rule-only) is required for the L2 selection stage, so ensure it is
    # in the swept grid (prepended if the user omitted it).
    alphas = [float(a) for a in args.alpha_grid]
    if rule_only_mode:
        if any(a != 0.0 for a in alphas):
            print("rule-only mode: alpha sweep disabled; using alpha=0 only.")
        alphas = [0.0]
    elif 0.0 not in alphas:
        alphas = [0.0] + alphas
        print("alpha_grid: 0.0 prepended (needed for rule-only L2 selection)")
    print(f"logreg    : {G} L2 value(s) {l2_grid}  R={R} rules")
    print(f"alpha_grid: {alphas}  (KGE fusion: {'OFF (rule-only)' if rule_only_mode else 'ON'})")

    # Load counts (+ kge unless rule-only) per split, build filter once per split.
    cache = {}
    for split in args.splits:
        counts = torch.load(os.path.join(args.analysis_dir, f"counts_{split}.pt"),
                            map_location="cpu")
        if counts["rule_id"].numel() and int(counts["rule_id"].max()) >= R:
            raise ValueError(
                f"[{split}] counts reference rule_id >= R ({R}); logreg beta and "
                "counts come from different rule files.")
        if rule_only_mode:
            kge = None
        else:
            kge_d = torch.load(os.path.join(args.analysis_dir, f"kge_{split}.pt"),
                               map_location="cpu")
            kge = kge_d["kge"]
            if kge.size(0) != counts["query_h"].size(0):
                raise ValueError(
                    f"[{split}] kge has {kge.size(0)} queries but counts has "
                    f"{counts['query_h'].size(0)}. Re-dump both from the same run.")
        cache[split] = (counts, kge, build_hr2ooo(args.data_path, N, num_rel))
        print(f"  [{split}] queries={counts['query_h'].size(0)}  "
              f"nnz={counts['rule_id'].numel()}")

    # rows[split] = CSV rows; grid[split][(gi, ai)] = MRR (only computed cells).
    rows = {s: [] for s in args.splits}
    grid = {s: {} for s in args.splits}

    def record(split, gi, ai, m):
        grid[split][(gi, ai)] = m["MRR"]
        rows[split].append({
            "split": split, "l2": l2_grid[gi], "alpha": float(alphas[ai]),
            "MRR": round(m["MRR"], 6), "Hit@1": round(m["Hit@1"], 6),
            "Hit@3": round(m["Hit@3"], 6), "Hit@10": round(m["Hit@10"], 6),
            "MR": round(m["MR"], 4), "n": m["n"],
        })
        print(f"  [{split}] L2={l2_grid[gi]:<8g} alpha={float(alphas[ai]):<5g} "
              f"MRR={m['MRR']:.6f}")

    # Two-stage selection on select_split: L2 on rule-only MRR (alpha=0), then
    # alpha on combined MRR at that L2. Only the consulted cells are evaluated
    # unless --full_grid.
    sel_split = args.select_split
    if sel_split not in cache:
        raise ValueError(f"--select_split {sel_split} not in --splits {args.splits}")
    rep_split = args.report_split
    a0 = alphas.index(0.0)
    all_items = list(enumerate(alphas))

    if args.full_grid:
        for split in args.splits:
            counts, kge, hr2ooo = cache[split]
            for gi in range(G):
                ms = eval_g(counts, kge, beta_grid[gi], all_items, hr2ooo, N)
                for ai in range(len(alphas)):
                    record(split, gi, ai, ms[ai])
    else:
        # Stage 1 inputs: rule-only (alpha=0) for every L2 on the select split.
        counts, kge, hr2ooo = cache[sel_split]
        for gi in range(G):
            ms = eval_g(counts, kge, beta_grid[gi], [(a0, 0.0)], hr2ooo, N)
            record(sel_split, gi, a0, ms[a0])

    # Stage 1: L2 on rule-only (alpha=0) MRR.
    rule_only = {gi: grid[sel_split][(gi, a0)] for gi in range(G)}
    gi_best = max(rule_only, key=rule_only.get)
    l2_best = l2_grid[gi_best]
    rule_only_mrr = rule_only[gi_best]

    if not args.full_grid:
        # Stage 2 inputs: the remaining alphas at the selected L2 (the alpha=0
        # cell was already computed in stage 1 and is reused).
        counts, kge, hr2ooo = cache[sel_split]
        todo = [(ai, a) for ai, a in all_items if (gi_best, ai) not in grid[sel_split]]
        if todo:
            ms = eval_g(counts, kge, beta_grid[gi_best], todo, hr2ooo, N)
            for ai, _ in todo:
                record(sel_split, gi_best, ai, ms[ai])

    # Stage 2: alpha on combined MRR at the selected L2.
    combined = {ai: grid[sel_split][(gi_best, ai)] for ai in range(len(alphas))}
    ai_best = max(combined, key=combined.get)
    alpha_best = float(alphas[ai_best])
    combined_mrr = combined[ai_best]

    # Report split: evaluate only the selected cell (+ rule-only for reference)
    # unless the full grid was requested or it is the same split.
    if not args.full_grid and rep_split in cache and rep_split != sel_split:
        counts, kge, hr2ooo = cache[rep_split]
        items = [(a0, 0.0)] + ([(ai_best, alpha_best)] if ai_best != a0 else [])
        ms = eval_g(counts, kge, beta_grid[gi_best], items, hr2ooo, N)
        for ai, _ in items:
            record(rep_split, gi_best, ai, ms[ai])

    print("\n" + "=" * 70)
    print(f"STAGE 1  L2 on {sel_split} RULE-ONLY (alpha=0) MRR:")
    for gi in range(G):
        mark = "  <-- selected" if gi == gi_best else ""
        print(f"    L2={l2_grid[gi]:<8g}  MRR={rule_only[gi]:.6f}{mark}")
    print(f"STAGE 2  alpha on {sel_split} COMBINED MRR at L2={l2_best:g}:")
    for ai, a in enumerate(alphas):
        mark = "  <-- selected" if ai == ai_best else ""
        print(f"    alpha={a:<5g}  MRR={combined[ai]:.6f}{mark}")
    print(f"SELECTED : L2={l2_best:g} (rule-only MRR {rule_only_mrr:.6f})  "
          f"alpha={alpha_best:g} (combined MRR {combined_mrr:.6f})")

    selection = {
        "selected_l2": l2_best,
        "selected_alpha": alpha_best,
        "select_split": sel_split,
        "l2_selected_on": "rule_only_mrr",
        "alpha_selected_on": "combined_mrr",
        f"{sel_split}_rule_only_MRR": rule_only_mrr,
        f"{sel_split}_combined_MRR": combined_mrr,
        "alpha_grid": alphas,
        "l2_grid": l2_grid,
    }
    if rep_split in grid and (gi_best, ai_best) in grid[rep_split]:
        rep_row = next(r for r in rows[rep_split]
                       if r["l2"] == l2_best and r["alpha"] == alpha_best)
        selection[f"{rep_split}_metrics"] = rep_row
        print(f"REPORT   on {rep_split} at selection: MRR={rep_row['MRR']:.6f}  "
              f"Hit@1={rep_row['Hit@1']:.6f}  Hit@10={rep_row['Hit@10']:.6f}")
    print("=" * 70)

    # Write per-split sweep CSVs (only the evaluated cells unless --full_grid).
    for split in args.splits:
        if not rows[split]:
            continue
        rows[split].sort(key=lambda r: (r["l2"], r["alpha"]))
        path = os.path.join(out_dir, f"logreg_alpha_sweep_{split}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["split", "l2", "alpha", "MRR",
                                               "Hit@1", "Hit@3", "Hit@10", "MR", "n"])
            w.writeheader()
            w.writerows(rows[split])
        print(f"Wrote {path}  ({len(rows[split])} rows)")

    # Selected-L2 weight file: the single beta the pipeline commits to, in the
    # schema src_additive/main.py --logreg_binary loads ('beta' [R] + 'intercept').
    sel_pt = os.path.join(out_dir, "rule_logreg_selected.pt")
    torch.save({
        "beta":      beta_grid[gi_best].clone(),                       # [R] float32
        "intercept": torch.tensor(float(intercept_grid[gi_best]),
                                  dtype=torch.float32),
        "meta": {
            "selected_l2":    l2_best,
            "selected_alpha": alpha_best,
            "rule_only":      rule_only_mode,
            "source":         os.path.abspath(args.logreg_file),
            f"{sel_split}_MRR": combined_mrr,
        },
    }, sel_pt)
    print(f"Wrote {sel_pt}  (L2={l2_best:g}, alpha={alpha_best:g})")

    sel_path = os.path.join(out_dir, "logreg_selection.json")
    with open(sel_path, "w") as fh:
        json.dump(selection, fh, indent=2)
    print(f"Wrote {sel_path}\nDone.")


if __name__ == "__main__":
    main()
