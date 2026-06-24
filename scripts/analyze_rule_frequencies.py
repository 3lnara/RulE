#!/usr/bin/env python3
"""
Rule shape / frequency analysis + P(gold|count) diagnostic.

For each rule, computes:
  fire_rate      : fraction of its relation's queries it grounds >= 1 entity
  p_count_ge2    : fraction of (query, entity) pairs where count >= 2
  mean_count     : mean path count over grounded (query, entity) pairs
  max_count      : maximum path count observed
  w_R_unclamped  : raw RulE confidence (can be negative)
  length         : body length (1=chain-1, 2=chain-2, ...)
  head           : head relation id

Diagnostic: is raw path-count actually informative about correctness?
  premise test   : P(entity == gold | count bucket) -- raw's core assumption
  confound test  : corr(count, entity degree) -- does count track popularity?
  within-query   : % of queries where gold has the highest total raw count

Outputs in --analysis_dir:
  rules_summary.csv            per-rule table (one row per rule)
  pgold_by_count_<split>.csv   P(gold|count) by count bucket
  pgold_by_degree_<split>.csv  P(gold), mean count by entity degree quartile

Usage (run from repo root):
    python scripts/analyze_rule_frequencies.py \\
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
    p.add_argument("--split",      default="valid",
                   help="Which split's counts to use for frequency / diagnostic stats.")
    return p.parse_args()


def main():
    args = parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    os.chdir(_root)

    os.makedirs(args.analysis_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    meta  = torch.load(
        os.path.join(args.analysis_dir, "rule_meta.pt"), weights_only=False)
    w_R   = meta["w_R_unclamped"].numpy()
    lens  = meta["length"].numpy()
    heads = meta["head"].numpy()
    R     = len(w_R)

    counts   = torch.load(
        os.path.join(args.analysis_dir, f"counts_{args.split}.pt"),
        weights_only=False)
    query_r    = counts["query_r"]
    query_gold = counts["query_gold"]
    q_ptr      = counts["q_ptr"]
    rule_id    = counts["rule_id"]
    ent        = counts["ent"]
    cnt        = counts["cnt"].long()
    Q          = query_r.size(0)

    # ------------------------------------------------------------------
    # Entity degree from train.txt (head+tail occurrences)
    # ------------------------------------------------------------------
    e2id = {}
    with open(os.path.join(args.data_path, "entities.dict")) as f:
        for line in f:
            i, n = line.strip().split("\t"); e2id[n] = int(i)
    N = len(e2id)

    r2id = {}
    with open(os.path.join(args.data_path, "relations.dict")) as f:
        for line in f:
            i, n = line.strip().split("\t"); r2id[n] = int(i)

    deg = np.zeros(N, dtype=np.int64)
    with open(os.path.join(args.data_path, "train.txt")) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            deg[e2id[p[0]]] += 1
            deg[e2id[p[2]]] += 1

    # ------------------------------------------------------------------
    # Per-relation query count (fire_rate denominator)
    # ------------------------------------------------------------------
    q_per_rel = defaultdict(int)
    for r in query_r.tolist():
        q_per_rel[r] += 1

    # ------------------------------------------------------------------
    # Per-rule accumulators + per-triple lists for diagnostic
    # ------------------------------------------------------------------
    rule_fired_queries = defaultdict(int)
    rule_count_values  = defaultdict(list)

    all_counts  = []    # per-(query,rule,entity) count value
    all_isgold  = []    # 1 if entity == gold for this query, else 0
    all_edeg    = []    # entity degree

    top_raw_hit = 0     # queries where gold has max total raw count
    n_queries_grounded = 0

    for qi in range(Q):
        r_val  = query_r[qi].item()
        gold   = query_gold[qi].item()
        s      = q_ptr[qi].item()
        e      = q_ptr[qi + 1].item()
        if s == e:
            continue
        n_queries_grounded += 1

        r_ids = rule_id[s:e].tolist()
        ents  = ent[s:e].tolist()
        cnts  = cnt[s:e].tolist()

        fired_set = set(r_ids)
        for rid in fired_set:
            rule_fired_queries[rid] += 1

        per_ent_total = defaultdict(int)
        for rid, en, c in zip(r_ids, ents, cnts):
            rule_count_values[rid].append(c)
            all_counts.append(c)
            all_isgold.append(1 if en == gold else 0)
            all_edeg.append(int(deg[en]))
            per_ent_total[en] += c

        mx = max(per_ent_total.values())
        if gold in per_ent_total and per_ent_total[gold] == mx:
            top_raw_hit += 1

    all_counts = np.array(all_counts, dtype=np.int64)
    all_isgold = np.array(all_isgold, dtype=np.int8)
    all_edeg   = np.array(all_edeg,   dtype=np.int64)

    # ------------------------------------------------------------------
    # Per-rule summary CSV (rules_summary.csv)
    # ------------------------------------------------------------------
    rows = []
    for rid in range(R):
        r_head  = int(heads[rid])
        n_q_rel = q_per_rel.get(r_head, 0)
        fired   = rule_fired_queries.get(rid, 0)
        fire_rate = fired / n_q_rel if n_q_rel > 0 else 0.0

        cvs = rule_count_values.get(rid, [])
        if cvs:
            arr    = np.array(cvs)
            mean_c = float(arr.mean())
            max_c  = int(arr.max())
            p_ge2  = float((arr >= 2).mean())
        else:
            mean_c = max_c = p_ge2 = 0.0

        rows.append({
            "rule_id":       rid,
            "head":          r_head,
            "length":        int(lens[rid]),
            "w_R_unclamped": round(float(w_R[rid]), 6),
            "fire_rate":     round(fire_rate, 6),
            "p_count_ge2":   round(p_ge2, 6),
            "mean_count":    round(mean_c, 4),
            "max_count":     max_c,
            "n_firings":     len(cvs),
        })

    csv_path = os.path.join(args.analysis_dir, "rules_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["rule_id", "head", "length", "w_R_unclamped",
                        "fire_rate", "p_count_ge2", "mean_count", "max_count",
                        "n_firings"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}  ({len(rows)} rules)")

    # ------------------------------------------------------------------
    # Global count stats (printed)
    # ------------------------------------------------------------------
    print(f"\nGlobal count stats (split={args.split}):")
    print(f"  total (query, rule, entity) triplets with count>0 : {len(all_counts)}")
    print(f"  count==1 : {int((all_counts==1).sum())}  "
          f"({100*(all_counts==1).mean():.1f}%)")
    print(f"  count>=2 : {int((all_counts>=2).sum())}  "
          f"({100*(all_counts>=2).mean():.1f}%)")
    print(f"  max count: {int(all_counts.max())}  mean: {all_counts.mean():.3f}")

    active_rules = sum(1 for r in rows if r["fire_rate"] > 0)
    print(f"  rules with fire_rate>0 : {active_rules}/{R}  "
          f"({100*active_rules/R:.1f}%)")

    # ------------------------------------------------------------------
    # P(gold|count) premise diagnostic
    # ------------------------------------------------------------------
    base_rate = float(all_isgold.mean())
    print(f"\n  base rate P(entity==gold) over all grounded triples = {base_rate:.4f}")
    print(f"\n  [Premise test] P(entity==gold | per-triple count bucket):")
    print(f"  {'bucket':>6}  {'n':>8}  {'P(gold)':>8}  {'mean_deg':>9}")

    BUCKETS = [(1, 1, "=1"), (2, 2, "=2"), (3, 3, "=3"),
               (4, 5, "4-5"), (6, 10, "6-10"), (11, 10**9, "11+")]

    pgold_count_rows = []
    for lo, hi, lab in BUCKETS:
        m = (all_counts >= lo) & (all_counts <= hi)
        if m.sum() == 0:
            continue
        pg  = float(all_isgold[m].mean())
        mdg = float(all_edeg[m].mean())
        print(f"  {lab:>6}  {m.sum():>8d}  {pg:>8.4f}  {mdg:>9.1f}")
        pgold_count_rows.append({
            "bucket": lab, "n": int(m.sum()),
            "p_gold": round(pg, 6), "mean_degree": round(mdg, 2),
        })

    pgc_csv = os.path.join(args.analysis_dir, f"pgold_by_count_{args.split}.csv")
    with open(pgc_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["bucket", "n", "p_gold", "mean_degree"])
        writer.writeheader()
        writer.writerows(pgold_count_rows)
    print(f"\n  Wrote {pgc_csv}")

    # ------------------------------------------------------------------
    # Degree-confound diagnostic
    # ------------------------------------------------------------------
    corr = float(np.corrcoef(all_counts, all_edeg)[0, 1])
    print(f"\n  [Confound test] corr(count, entity_degree) = {corr:.3f}")
    print(f"  Mean count + P(gold) by entity-degree quartile:")
    qs = np.quantile(all_edeg, [0, 0.25, 0.5, 0.75, 1.0])

    pgold_deg_rows = []
    for i in range(4):
        m = (all_edeg >= qs[i]) & (all_edeg <= qs[i + 1])
        mc = float(all_counts[m].mean())
        pg = float(all_isgold[m].mean())
        lab = f"deg[{int(qs[i])}-{int(qs[i+1])}]"
        print(f"    {lab}: mean_count={mc:.3f}  P(gold)={pg:.4f}")
        pgold_deg_rows.append({
            "quartile": i + 1,
            "deg_lo": int(qs[i]), "deg_hi": int(qs[i + 1]),
            "n": int(m.sum()),
            "mean_count": round(mc, 4), "p_gold": round(pg, 6),
        })

    pgd_csv = os.path.join(args.analysis_dir, f"pgold_by_degree_{args.split}.csv")
    with open(pgd_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["quartile", "deg_lo", "deg_hi", "n",
                            "mean_count", "p_gold"])
        writer.writeheader()
        writer.writerows(pgold_deg_rows)
    print(f"\n  Wrote {pgd_csv}")

    # ------------------------------------------------------------------
    # Within-query top-count diagnostic (printed only)
    # ------------------------------------------------------------------
    pct_top = 100 * top_raw_hit / n_queries_grounded if n_queries_grounded else 0.0
    print(f"\n  [Within-query] gold is among the highest total-raw-count entities "
          f"in {pct_top:.1f}% of queries (n={n_queries_grounded})")

    print("\nDone.")


if __name__ == "__main__":
    main()
