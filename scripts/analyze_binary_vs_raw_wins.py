#!/usr/bin/env python3
"""
Per-query win/lose analysis: where does raw count help vs hurt vs tie vs binary?

All inputs come from the pre-dumped counts (dump_rule_counts.py) and the
pre-scored per-query ranks (score_counts_offline.py). No re-grounding needed.

For each query, binary vs raw "clamp" variants are compared (clamp is the
primary baseline; keep variants reported alongside for completeness).

  rank_delta = midpoint_rank_binary - midpoint_rank_raw
    > 0  -> raw wins   (raw places gold higher than binary)
    = 0  -> tie
    < 0  -> binary wins  (binary places gold higher than raw)

  gold_ppr  = gold_total_paths  / gold_n_rules_fired   (paths per rule)
  comp_ppr  = comp_total_paths  / comp_n_rules_fired

Outputs in --analysis_dir:
  winloss_<split>.csv          per-query table (qi, h, r, gold, rel_name,
                                 outcome, rank_delta, ranks, total_paths,
                                 n_rules_fired, ppr columns)
  ppr_by_outcome_<split>.csv   per-outcome means: n, mean_gold_ppr,
                                 mean_comp_ppr, ppr_adv, n_rules, total_paths,
                                 pct_gold_ppr_gt_comp

Usage (run from repo root):
    python scripts/analyze_binary_vs_raw_wins.py \\
        --analysis_dir outputs/additive_umls_aggregation_2x2/analysis \\
        --data_path    data/umls \\
        --split        valid
"""

import argparse
import csv
import os
from collections import defaultdict

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir",
                   default="outputs/additive_umls_aggregation_2x2/analysis")
    p.add_argument("--data_path",  default="data/umls")
    p.add_argument("--split",      default="valid")
    return p.parse_args()


def midpoint_rank(L: int, H: int) -> float:
    return (L + H - 1) / 2.0


def main():
    args = parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    os.chdir(_root)

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    meta = torch.load(
        os.path.join(args.analysis_dir, "rule_meta.pt"), weights_only=False)

    counts = torch.load(
        os.path.join(args.analysis_dir, f"counts_{args.split}.pt"),
        weights_only=False)
    query_h    = counts["query_h"]
    query_r    = counts["query_r"]
    query_gold = counts["query_gold"]
    q_ptr      = counts["q_ptr"]
    rule_id    = counts["rule_id"]
    ent_arr    = counts["ent"]
    cnt_arr    = counts["cnt"].long()
    Q          = query_h.size(0)

    ent2id = {}
    with open(os.path.join(args.data_path, "entities.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            ent2id[name] = int(idx)
    N = len(ent2id)

    rel_names = {}
    rel2id_dict = {}
    with open(os.path.join(args.data_path, "relations.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            rel_names[int(idx)] = name
            rel2id_dict[name] = int(idx)
    num_rel = len(rel2id_dict)

    # ------------------------------------------------------------------
    # Build hr2ooo filter
    # ------------------------------------------------------------------
    def enc(h, r):
        return r * N + h

    hr2ooo = defaultdict(set)
    for fname in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(args.data_path, fname)) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h = ent2id[parts[0]]
                r = rel2id_dict[parts[1]]
                t = ent2id[parts[2]]
                hr2ooo[enc(h, r)].add(t)
                hr2ooo[enc(t, r + num_rel)].add(h)

    # ------------------------------------------------------------------
    # Load per-query ranks from score_counts_offline
    # ------------------------------------------------------------------
    ranks_path = os.path.join(args.analysis_dir, f"ranks_{args.split}.csv")
    ranks_by_qi = {}
    with open(ranks_path) as f:
        reader = csv.DictReader(f)
        for qi, row in enumerate(reader):
            ranks_by_qi[qi] = {k: int(v) for k, v in row.items()
                               if k.startswith("L_") or k.startswith("H_")}

    # ------------------------------------------------------------------
    # Compute per-query features + delta
    # ------------------------------------------------------------------
    winloss_rows = []

    for qi in range(Q):
        h     = query_h[qi].item()
        r     = query_r[qi].item()
        gold  = query_gold[qi].item()
        s     = q_ptr[qi].item()
        e     = q_ptr[qi + 1].item()

        ent_total  = defaultdict(int)
        ent_nrules = defaultdict(int)

        if s < e:
            r_ids = rule_id[s:e].tolist()
            ents  = ent_arr[s:e].tolist()
            cnts  = cnt_arr[s:e].tolist()
            for rid, ei, c in zip(r_ids, ents, cnts):
                ent_total[ei]  += c
                ent_nrules[ei] += 1

        total_paths_gold   = ent_total.get(gold, 0)
        n_rules_fired_gold = ent_nrules.get(gold, 0)

        known_true  = hr2ooo.get(enc(h, r), set())
        grounded    = set(ent_total.keys())
        competitors = [e for e in grounded if e not in known_true]

        if competitors:
            best_comp = max(competitors, key=lambda e: ent_total[e])
            total_paths_best_comp   = ent_total[best_comp]
            n_rules_fired_best_comp = ent_nrules[best_comp]
        else:
            total_paths_best_comp   = 0
            n_rules_fired_best_comp = 0

        gold_ppr = total_paths_gold / max(1, n_rules_fired_gold)
        comp_ppr = total_paths_best_comp / max(1, n_rules_fired_best_comp)

        rk = ranks_by_qi[qi]
        r_bin      = midpoint_rank(rk["L_binary_clamp"], rk["H_binary_clamp"])
        r_raw      = midpoint_rank(rk["L_raw_clamp"],    rk["H_raw_clamp"])
        r_bin_keep = midpoint_rank(rk["L_binary_keep"],  rk["H_binary_keep"])
        r_raw_keep = midpoint_rank(rk["L_raw_keep"],     rk["H_raw_keep"])

        delta = r_bin - r_raw
        if abs(delta) < 1e-9:
            outcome = "tie"
        elif delta > 0:
            outcome = "raw_wins"
        else:
            outcome = "binary_wins"

        winloss_rows.append({
            "qi":                    qi,
            "h":                     h,
            "r":                     r,
            "gold":                  gold,
            "rel_name":              rel_names.get(r % num_rel, str(r)),
            "outcome":               outcome,
            "rank_delta":            round(delta, 2),
            "rank_binary_clamp":     round(r_bin,      2),
            "rank_raw_clamp":        round(r_raw,      2),
            "rank_binary_keep":      round(r_bin_keep, 2),
            "rank_raw_keep":         round(r_raw_keep, 2),
            "gold_total_paths":      total_paths_gold,
            "gold_n_rules_fired":    n_rules_fired_gold,
            "best_comp_total_paths": total_paths_best_comp,
            "best_comp_n_rules_fired": n_rules_fired_best_comp,
            "gold_ppr":              round(gold_ppr, 4),
            "comp_ppr":              round(comp_ppr, 4),
        })

    # Write per-query CSV
    wl_csv = os.path.join(args.analysis_dir, f"winloss_{args.split}.csv")
    fieldnames = list(winloss_rows[0].keys())
    with open(wl_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(winloss_rows)
    print(f"Wrote {wl_csv}  ({Q} queries)")

    # ------------------------------------------------------------------
    # Summary stats (printed)
    # ------------------------------------------------------------------
    outcomes  = [r["outcome"] for r in winloss_rows]
    n_raw     = outcomes.count("raw_wins")
    n_bin     = outcomes.count("binary_wins")
    n_tie     = outcomes.count("tie")
    deltas    = np.array([r["rank_delta"] for r in winloss_rows])

    print(f"\n=== {args.split.upper()} binary-vs-raw (clamp) ===")
    print(f"  raw wins    : {n_raw:5d}  ({100*n_raw/Q:.1f}%)")
    print(f"  binary wins : {n_bin:5d}  ({100*n_bin/Q:.1f}%)")
    print(f"  ties        : {n_tie:5d}  ({100*n_tie/Q:.1f}%)")
    print(f"  mean(rank_delta): {deltas.mean():.3f}  (positive = raw wins on average)")

    # ------------------------------------------------------------------
    # ppr-by-outcome table
    # ------------------------------------------------------------------
    # Only include rows where both entities had at least one rule fire
    # (guards ppr division and makes the means meaningful)
    valid_rows = [r for r in winloss_rows
                  if r["gold_n_rules_fired"] > 0 and r["best_comp_n_rules_fired"] > 0]

    ppr_outcome_rows = []
    print(f"\n  Per-outcome paths-per-rule (ppr) stats  (n={len(valid_rows)} queries with both gold+comp grounded):")
    print(f"  {'outcome':12s} {'n':>5}  {'gold_ppr':>8}  {'comp_ppr':>8}  "
          f"{'ppr_adv':>8}  {'gold_nr':>7}  {'comp_nr':>7}  "
          f"{'gold_tp':>7}  {'comp_tp':>7}  {'%gppr>cppr':>10}")

    for o in ("raw_wins", "binary_wins", "tie"):
        sub = [r for r in valid_rows if r["outcome"] == o]
        if not sub:
            continue
        gp  = np.mean([r["gold_ppr"]              for r in sub])
        cp  = np.mean([r["comp_ppr"]              for r in sub])
        gnr = np.mean([r["gold_n_rules_fired"]    for r in sub])
        cnr = np.mean([r["best_comp_n_rules_fired"] for r in sub])
        gtp = np.mean([r["gold_total_paths"]       for r in sub])
        ctp = np.mean([r["best_comp_total_paths"]  for r in sub])
        pct = 100 * np.mean([1 if r["gold_ppr"] > r["comp_ppr"] else 0 for r in sub])
        print(f"  {o:12s} {len(sub):5d}  {gp:8.3f}  {cp:8.3f}  "
              f"{gp-cp:8.3f}  {gnr:7.1f}  {cnr:7.1f}  "
              f"{gtp:7.1f}  {ctp:7.1f}  {pct:10.1f}%")
        ppr_outcome_rows.append({
            "outcome":             o,
            "n":                   len(sub),
            "mean_gold_ppr":       round(gp,  4),
            "mean_comp_ppr":       round(cp,  4),
            "ppr_adv":             round(gp - cp, 4),
            "mean_gold_n_rules_fired":    round(gnr, 2),
            "mean_comp_n_rules_fired":    round(cnr, 2),
            "mean_gold_total_paths":      round(gtp, 2),
            "mean_comp_total_paths":      round(ctp, 2),
            "pct_gold_ppr_gt_comp":       round(pct, 2),
        })

    ppr_csv = os.path.join(args.analysis_dir, f"ppr_by_outcome_{args.split}.csv")
    with open(ppr_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ppr_outcome_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ppr_outcome_rows)
    print(f"\n  Wrote {ppr_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
