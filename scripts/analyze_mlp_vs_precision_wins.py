#!/usr/bin/env python3
"""
Per-query win/loss: original MLP rule scorer vs precision*binary.

Both per-query rank dumps are produced in-model on GPU (HPC) and synced back;
this script is a pure-offline join + analysis (no GPU, no grounding). The two
rank CSVs share the schema `h,r,t,L,H` and are joined on (h, r, t).

Self-contained: this analysis depends ONLY on its three direct inputs and the
dataset files. It does NOT read winloss_<split>.csv, ranks_<split>.csv, or any
other artifact from the binary-vs-raw / offline-scoring scripts. The co-fire
features (gold_n_rules_fired, best_comp_n_rules_fired, query_distinct_rules) are
computed here directly from the grounding counts.

Inputs:
  precision_dir/ranks_<split>.csv      precision*binary ranks (HPC)
  mlp_dir/ranks_mlp_<split>.csv        original MLP ranks (HPC)
  analysis_dir/counts_<split>.pt       per-(query,rule,entity) groundings
                                       (dump_rule_counts.py; the grounding
                                       source for the co-fire features)

  rank_precision = midpoint_rank(L, H)  from precision_binary/ranks_<split>.csv
  rank_mlp       = midpoint_rank(L, H)  from mlp_eval-umls/ranks_mlp_<split>.csv

  delta = rank_precision - rank_mlp
    > 0  -> MLP ranks gold better (lower rank)  -> mlp_wins
    < 0  -> precision ranks gold better         -> precision_wins
    ~ 0  -> near_tie (|delta| <= eps)

Headline question: where the MLP beats precision*binary, do those queries have
more co-firing rules than near-ties, after controlling for entity degree?

Outputs in --analysis_dir:
  mlp_vs_precision_<split>.csv          per-query table
  mlp_vs_precision_summary_<split>.csv  per-outcome feature means
  winrate_by_cofire_<split>.csv         quintile win-rate by co-fire counts
  winrate_by_degree_<split>.csv         quartile win-rate by gold degree
  winrate_cofire_x_degree_<split>.csv   2D co-fire x degree win-rate grid

Usage (run from repo root):
    python scripts/analyze_mlp_vs_precision_wins.py \\
        --analysis_dir  outputs/additive_umls_aggregation_2x2/analysis \\
        --precision_dir outputs/additive_umls_aggregation_2x2/precision_binary \\
        --mlp_dir       outputs/additive_umls_aggregation_2x2/mlp_eval-umls \\
        --data_path     data/umls \\
        --split         valid
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch


# Per-dataset, per-split reference MRRs (from each HPC dump's run.log) used as
# validation gates. Keyed by dataset (data_path basename) so the script stays
# standalone and works across datasets without any external lookup.
REFERENCES = {
    "umls": {
        "precision": {"valid": 0.726304, "test": 0.724881},
        "mlp":       {"valid": 0.805971, "test": 0.805063},
    },
    "family": {
        "precision": {"valid": 0.956762},
        "mlp":       {"valid": 0.972564, "test": 0.974351},
    },
}
MRR_TOL = 2e-3


def expectation_metrics(results):
    """MRR/MR/Hits as an expectation over the tie band [L, H) per query.

    results: iterable of (h, r, t, L, H). Inlined (no import from
    score_counts_offline) so this analysis is fully self-contained.
    """
    h1 = h3 = h10 = mr = mrr = 0.0
    n = 0
    for (_, _, _, L, H) in results:
        n += 1
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
    return {"MRR": mrr / n, "MR": mr / n, "Hit@1": h1 / n,
            "Hit@3": h3 / n, "Hit@10": h10 / n, "n": n}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir",
                   default="outputs/additive_umls_aggregation_2x2/analysis")
    p.add_argument("--precision_dir",
                   default="outputs/additive_umls_aggregation_2x2/precision_binary")
    p.add_argument("--mlp_dir",
                   default="outputs/additive_umls_aggregation_2x2/mlp_eval-umls")
    p.add_argument("--data_path", default="data/umls")
    p.add_argument("--split",     default="valid")
    p.add_argument("--eps", type=float, default=1.0,
                   help="Near-tie band on rank delta (default 1.0).")
    return p.parse_args()


def midpoint_rank(L, H):
    return (L + H - 1) / 2.0


def read_ranks(path):
    """Read an h,r,t,L,H rank CSV -> dict (h,r,t) -> (L, H, midpoint)."""
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h, r, t = int(row["h"]), int(row["r"]), int(row["t"])
            L, H = int(row["L"]), int(row["H"])
            out[(h, r, t)] = (L, H, midpoint_rank(L, H))
    return out


def load_gold_degree(data_path):
    """Entity degree (head+tail occurrences) from train.txt, keyed by ent id."""
    e2id = {}
    with open(os.path.join(data_path, "entities.dict")) as f:
        for line in f:
            i, n = line.strip().split("\t")
            e2id[n] = int(i)
    N = len(e2id)
    deg = np.zeros(N, dtype=np.int64)
    with open(os.path.join(data_path, "train.txt")) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            deg[e2id[p[0]]] += 1
            deg[e2id[p[2]]] += 1
    return deg


def _load_dict(path):
    """Load an id<TAB>name .dict file -> (name->id dict, count)."""
    name2id = {}
    with open(path) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            name2id[name] = int(idx)
    return name2id, len(name2id)


def build_hr2ooo(data_path, e2id, rel2id, N, num_rel):
    """All known-true tails per (h, r), across train/valid/test, plus inverse
    relations. Used to exclude true answers when picking the best competitor.

    Encoding enc(h, r) = r * N + h matches the grounding/eval convention; inverse
    triples are stored under relation id r + num_rel.
    """
    def enc(h, r):
        return r * N + h

    hr2ooo = defaultdict(set)
    for fname in ("train.txt", "valid.txt", "test.txt"):
        path = os.path.join(data_path, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h = e2id[parts[0]]
                r = rel2id[parts[1]]
                t = e2id[parts[2]]
                hr2ooo[enc(h, r)].add(t)
                hr2ooo[enc(t, r + num_rel)].add(h)
    return hr2ooo


def load_query_features(counts_path, data_path):
    """Compute co-fire features per query directly from the grounding counts.

    For each query (qi) the CSR slice [q_ptr[qi], q_ptr[qi+1]) lists
    (rule_id, entity, raw_count) groundings. We derive, keyed by (h, r, gold):
      gold_n_rules_fired      distinct rules grounding a path to the gold tail
      best_comp_n_rules_fired distinct rules for the strongest non-true
                              competitor (max raw paths, excluding known-true)
      query_distinct_rules    distinct rules firing anywhere for the query

    Returns (ordered_keys, feats) where ordered_keys preserves counts qi order.
    """
    try:
        counts = torch.load(counts_path, weights_only=False)
    except TypeError:
        # PyTorch < 1.13 (e.g. local rule_env on 1.11) has no weights_only kwarg
        counts = torch.load(counts_path)
    query_h    = counts["query_h"]
    query_r    = counts["query_r"]
    query_gold = counts["query_gold"]
    q_ptr      = counts["q_ptr"]
    rule_id    = counts["rule_id"]
    ent_arr    = counts["ent"]
    cnt_arr    = counts["cnt"].long()
    Q = query_h.size(0)

    e2id, N        = _load_dict(os.path.join(data_path, "entities.dict"))
    rel2id, num_rel = _load_dict(os.path.join(data_path, "relations.dict"))
    hr2ooo = build_hr2ooo(data_path, e2id, rel2id, N, num_rel)

    def enc(h, r):
        return r * N + h

    ordered_keys = []
    feats = {}
    for qi in range(Q):
        h    = query_h[qi].item()
        r    = query_r[qi].item()
        gold = query_gold[qi].item()
        s    = q_ptr[qi].item()
        e    = q_ptr[qi + 1].item()

        ent_total  = defaultdict(int)   # raw paths per entity
        ent_nrules = defaultdict(int)   # distinct rules per entity
        n_distinct = 0
        if e > s:
            r_ids = rule_id[s:e].tolist()
            ents  = ent_arr[s:e].tolist()
            cnts  = cnt_arr[s:e].tolist()
            for rid, ei, c in zip(r_ids, ents, cnts):
                ent_total[ei]  += c
                ent_nrules[ei] += 1
            n_distinct = len(set(r_ids))

        gold_nr = ent_nrules.get(gold, 0)

        known_true  = hr2ooo.get(enc(h, r), set())
        competitors = [x for x in ent_total if x not in known_true]
        if competitors:
            best_comp = max(competitors, key=lambda x: ent_total[x])
            comp_nr = ent_nrules[best_comp]
        else:
            comp_nr = 0

        key = (h, r, gold)
        ordered_keys.append(key)
        feats[key] = {
            "gold_n_rules_fired":      gold_nr,
            "best_comp_n_rules_fired": comp_nr,
            "query_distinct_rules":    n_distinct,
        }
    return ordered_keys, feats


def quantile_bins(values, n_bins, labels=None):
    """Assign each value to a quantile bin index [0, n_bins).

    Uses unique quantile edges; collapses bins when ties dominate. Returns
    (bin_idx array, list of (lo, hi, label) edges actually used).
    """
    values = np.asarray(values, dtype=np.float64)
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(qs)
    if edges.size < 2:
        # Degenerate: all values identical.
        return np.zeros(values.shape[0], dtype=int), [(edges[0], edges[0], "all")]
    # np.digitize with right=False: bin i = edges[i-1] <= v < edges[i].
    idx = np.clip(np.digitize(values, edges[1:-1], right=False),
                  0, edges.size - 2)
    spans = []
    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        lab = labels[i] if labels and i < len(labels) else f"[{lo:g},{hi:g}]"
        spans.append((lo, hi, lab))
    return idx, spans


def winrate_row(sub_outcomes):
    """Given a list of outcome strings, return (n, pct_mlp, pct_prec, pct_tie)."""
    n = len(sub_outcomes)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    pm = 100 * sum(o == "mlp_wins"       for o in sub_outcomes) / n
    pp = 100 * sum(o == "precision_wins" for o in sub_outcomes) / n
    pt = 100 * sum(o == "near_tie"       for o in sub_outcomes) / n
    return n, pm, pp, pt


def main():
    args = parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    os.chdir(_root)

    split = args.split

    # ------------------------------------------------------------------
    # Load rank dumps
    # ------------------------------------------------------------------
    prec_path = os.path.join(args.precision_dir, f"ranks_{split}.csv")
    mlp_path  = os.path.join(args.mlp_dir,       f"ranks_mlp_{split}.csv")
    print(f"Loading precision ranks : {prec_path}")
    prec = read_ranks(prec_path)
    print(f"Loading MLP ranks       : {mlp_path}")
    mlp = read_ranks(mlp_path)
    print(f"  precision queries={len(prec)}  mlp queries={len(mlp)}")

    # ------------------------------------------------------------------
    # Co-fire features computed directly from the grounding counts.
    # (Only non-rank input; no winloss / offline-scoring artifacts needed.)
    # ------------------------------------------------------------------
    counts_path = os.path.join(args.analysis_dir, f"counts_{split}.pt")
    print(f"Loading counts          : {counts_path}")
    ordered_keys, feats = load_query_features(counts_path, args.data_path)
    Q = len(ordered_keys)
    print(f"  queries={Q}")

    # ------------------------------------------------------------------
    # Gold degree
    # ------------------------------------------------------------------
    deg = load_gold_degree(args.data_path)

    # ------------------------------------------------------------------
    # Merge on (h, r, t) and validate
    # ------------------------------------------------------------------
    keys = set(prec) & set(mlp) & set(feats)
    if len(keys) != len(prec) or len(keys) != len(mlp):
        print(f"  WARNING: key overlap {len(keys)} vs precision {len(prec)} / "
              f"mlp {len(mlp)} / counts {len(feats)}")
    if len(keys) != Q:
        print(f"  WARNING: merged keys {len(keys)} != Q {Q}")

    rows = []
    prec_results = []   # (h, r, t, L, H) for MRR validation
    mlp_results  = []

    for key in ordered_keys:   # stable counts order
        if key not in prec or key not in mlp:
            raise ValueError(f"Missing rank for query {key} in prec/mlp dump")
        h, r, t = key
        Lp, Hp, rp = prec[key]
        Lm, Hm, rm = mlp[key]
        prec_results.append((h, r, t, Lp, Hp))
        mlp_results.append((h, r, t, Lm, Hm))

        ft = feats[key]
        cofire_gap = ft["gold_n_rules_fired"] - ft["best_comp_n_rules_fired"]
        rows.append({
            "h": h, "r": r, "t": t,
            "rank_precision": round(rp, 2),
            "rank_mlp":       round(rm, 2),
            "delta":          round(rp - rm, 2),
            "gold_n_rules_fired":      ft["gold_n_rules_fired"],
            "best_comp_n_rules_fired": ft["best_comp_n_rules_fired"],
            "cofire_gap":              cofire_gap,
            "query_distinct_rules":    ft["query_distinct_rules"],
            "gold_degree":             int(deg[t]) if t < deg.shape[0] else 0,
        })

    # ------------------------------------------------------------------
    # Validation gates (dataset-aware; reference MRRs from each HPC run.log)
    # ------------------------------------------------------------------
    dataset = os.path.basename(os.path.normpath(args.data_path))
    ds_refs = REFERENCES.get(dataset, {})
    print(f"\n=== {split.upper()} validation gates (dataset={dataset}) ===")
    for label, results in (("precision", prec_results), ("mlp", mlp_results)):
        m = expectation_metrics(results)
        ref = ds_refs.get(label, {}).get(split)
        if ref is None:
            print(f"  {label:<10} MRR={m['MRR']:.6f}  (no reference)")
            continue
        delta  = abs(m["MRR"] - ref)
        status = "PASS" if delta < MRR_TOL else "FAIL"
        print(f"  {label:<10} MRR={m['MRR']:.6f}  ref={ref:.6f}  "
              f"delta={delta:.2e}  [{status}]")

    # ------------------------------------------------------------------
    # Outcome classification (continues in the rest of main)
    # ------------------------------------------------------------------
    _classify_and_write(args, split, rows, deg)


def _classify_and_write(args, split, rows, deg):
    eps = args.eps

    # Degree quartile edges (computed over this split's gold entities).
    gold_degs = np.array([r["gold_degree"] for r in rows], dtype=np.float64)
    deg_idx, deg_spans = quantile_bins(gold_degs, 4)

    n_exact_mlp = n_exact_prec = n_exact_tie = 0
    for i, r in enumerate(rows):
        d = r["delta"]
        if d > eps:
            r["outcome"] = "mlp_wins"
        elif d < -eps:
            r["outcome"] = "precision_wins"
        else:
            r["outcome"] = "near_tie"
        r["degree_quartile"] = int(deg_idx[i]) + 1
        # exact-tie (eps=0) sensitivity
        if abs(d) < 1e-9:
            n_exact_tie += 1
        elif d > 0:
            n_exact_mlp += 1
        else:
            n_exact_prec += 1

    n = len(rows)
    outcomes = [r["outcome"] for r in rows]
    n_mlp  = outcomes.count("mlp_wins")
    n_prec = outcomes.count("precision_wins")
    n_tie  = outcomes.count("near_tie")

    print(f"\n=== {split.upper()} MLP vs precision*binary  (eps={eps}) ===")
    print(f"  mlp_wins       : {n_mlp:5d}  ({100*n_mlp/n:.1f}%)")
    print(f"  precision_wins : {n_prec:5d}  ({100*n_prec/n:.1f}%)")
    print(f"  near_tie       : {n_tie:5d}  ({100*n_tie/n:.1f}%)")
    print(f"  [exact-tie sensitivity eps=0] mlp={n_exact_mlp}  "
          f"precision={n_exact_prec}  tie={n_exact_tie}")

    # Write per-query CSV
    per_csv = os.path.join(args.analysis_dir, f"mlp_vs_precision_{split}.csv")
    fieldnames = ["h", "r", "t", "rank_precision", "rank_mlp", "delta",
                  "outcome", "gold_n_rules_fired", "best_comp_n_rules_fired",
                  "cofire_gap", "query_distinct_rules", "gold_degree",
                  "degree_quartile"]
    with open(per_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Wrote {per_csv}  ({n} queries)")

    # ------------------------------------------------------------------
    # Summary by outcome
    # ------------------------------------------------------------------
    print(f"\n  Per-outcome feature means:")
    print(f"  {'outcome':<16}{'n':>6}{'gold_nr':>9}{'comp_nr':>9}"
          f"{'cofire_gap':>11}{'q_distinct':>11}{'gold_deg':>9}")
    summary_rows = []
    for o in ("mlp_wins", "precision_wins", "near_tie"):
        sub = [r for r in rows if r["outcome"] == o]
        if not sub:
            continue
        gnr = np.mean([r["gold_n_rules_fired"]      for r in sub])
        cnr = np.mean([r["best_comp_n_rules_fired"] for r in sub])
        cg  = np.mean([r["cofire_gap"]              for r in sub])
        qd  = np.mean([r["query_distinct_rules"]    for r in sub])
        gd  = np.mean([r["gold_degree"]             for r in sub])
        print(f"  {o:<16}{len(sub):>6}{gnr:>9.2f}{cnr:>9.2f}"
              f"{cg:>11.2f}{qd:>11.2f}{gd:>9.2f}")
        summary_rows.append({
            "outcome": o, "n": len(sub),
            "mean_gold_n_rules_fired":      round(gnr, 3),
            "mean_comp_n_rules_fired":      round(cnr, 3),
            "mean_cofire_gap":              round(cg, 3),
            "mean_query_distinct_rules":    round(qd, 3),
            "mean_gold_degree":             round(gd, 3),
        })
    summ_csv = os.path.join(args.analysis_dir,
                            f"mlp_vs_precision_summary_{split}.csv")
    with open(summ_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Wrote {summ_csv}")

    # ------------------------------------------------------------------
    # Win-rate by co-fire (quintiles on query_distinct_rules and gold_nr)
    # ------------------------------------------------------------------
    cofire_rows = []
    for feat_name in ("query_distinct_rules", "gold_n_rules_fired"):
        vals = np.array([r[feat_name] for r in rows], dtype=np.float64)
        idx, spans = quantile_bins(vals, 5)
        print(f"\n  Win-rate by {feat_name} quintile:")
        print(f"  {'bin':<14}{'range':<18}{'n':>6}{'%mlp':>8}"
              f"{'%prec':>8}{'%tie':>8}")
        for b, (lo, hi, _lab) in enumerate(spans):
            sub = [rows[i]["outcome"] for i in range(len(rows)) if idx[i] == b]
            nb, pm, pp, pt = winrate_row(sub)
            rng = f"[{lo:g}, {hi:g}]"
            print(f"  {('q'+str(b+1)):<14}{rng:<18}{nb:>6}{pm:>8.1f}"
                  f"{pp:>8.1f}{pt:>8.1f}")
            cofire_rows.append({
                "feature": feat_name, "bin": b + 1,
                "lo": round(lo, 3), "hi": round(hi, 3), "n": nb,
                "pct_mlp_wins": round(pm, 2),
                "pct_precision_wins": round(pp, 2),
                "pct_near_tie": round(pt, 2),
            })
    cofire_csv = os.path.join(args.analysis_dir,
                              f"winrate_by_cofire_{split}.csv")
    with open(cofire_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cofire_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cofire_rows)
    print(f"\n  Wrote {cofire_csv}")

    # ------------------------------------------------------------------
    # Win-rate by gold-degree quartile
    # ------------------------------------------------------------------
    print(f"\n  Win-rate by gold-degree quartile:")
    print(f"  {'quartile':<10}{'n':>6}{'%mlp':>8}{'%prec':>8}{'%tie':>8}"
          f"{'mean_cofire':>12}")
    degree_rows = []
    for q in range(1, 5):
        sub_rows = [r for r in rows if r["degree_quartile"] == q]
        sub = [r["outcome"] for r in sub_rows]
        nb, pm, pp, pt = winrate_row(sub)
        mc = np.mean([r["cofire_gap"] for r in sub_rows]) if sub_rows else 0.0
        print(f"  Q{q:<9}{nb:>6}{pm:>8.1f}{pp:>8.1f}{pt:>8.1f}{mc:>12.2f}")
        degree_rows.append({
            "degree_quartile": q, "n": nb,
            "pct_mlp_wins": round(pm, 2),
            "pct_precision_wins": round(pp, 2),
            "pct_near_tie": round(pt, 2),
            "mean_cofire_gap": round(float(mc), 3),
        })
    degree_csv = os.path.join(args.analysis_dir,
                              f"winrate_by_degree_{split}.csv")
    with open(degree_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(degree_rows[0].keys()))
        writer.writeheader()
        writer.writerows(degree_rows)
    print(f"  Wrote {degree_csv}")

    # ------------------------------------------------------------------
    # 2D co-fire x degree win-rate grid (cofire_gap quartiles x degree quartiles)
    # ------------------------------------------------------------------
    cofire_vals = np.array([r["cofire_gap"] for r in rows], dtype=np.float64)
    cofire_idx, cofire_spans = quantile_bins(cofire_vals, 4)
    for i, r in enumerate(rows):
        r["_cofire_q"] = int(cofire_idx[i]) + 1

    print(f"\n  2D pct_mlp_wins  (rows=cofire_gap quartile, cols=degree quartile):")
    _corner = "cofire\\deg"
    header = "  " + f"{_corner:<14}" + "".join(
        f"{('D'+str(q)):>8}" for q in range(1, 5)) + f"{'n_row':>8}"
    print(header)
    grid_rows = []
    for cq in range(1, 5):
        cells = []
        row_total = 0
        for dq in range(1, 5):
            sub = [r["outcome"] for r in rows
                   if r["_cofire_q"] == cq and r["degree_quartile"] == dq]
            nb, pm, _pp, _pt = winrate_row(sub)
            cells.append((nb, pm))
            row_total += nb
            grid_rows.append({
                "cofire_quartile": cq, "degree_quartile": dq,
                "n": nb, "pct_mlp_wins": round(pm, 2),
            })
        cell_str = "".join(f"{pm:>8.1f}" for (_nb, pm) in cells)
        print(f"  {('C'+str(cq)):<14}{cell_str}{row_total:>8}")
    grid_csv = os.path.join(args.analysis_dir,
                            f"winrate_cofire_x_degree_{split}.csv")
    with open(grid_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)
    print(f"  Wrote {grid_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
