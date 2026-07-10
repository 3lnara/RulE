#!/usr/bin/env python3
"""
Co-firing rule interaction analysis: original MLP rule scorer vs precision*binary.

Answers two literal questions, on one split:
  (1) Among the queries where the MLP ranks gold better than precision*binary,
      how many have exactly two rules co-firing versus three-plus?
  (2) Where the MLP wins, do those queries have more co-firing rules than where
      the two scorers tie?

The premise: precision*binary is purely additive over rules (score = sum_R
w_R * 1[rule R fired on the candidate], no bias). It therefore CANNOT represent
cross-rule interaction -- two rules co-firing on the same candidate contribute
exactly their sum. The MLP head can. So if the MLP's edge is interaction, it
should grow with the number of rules co-firing on a candidate, especially on the
gold. A monotone rise in P(mlp_win) across cofire 1 -> 2 -> >=3 is the empirical
signature of cross-rule interaction; the fraction of queries at cofire>=2 is the
ceiling on interaction-reachable performance (the FM-worth-it decision input).

Inputs (three paths + one split). Each path may be the file itself or the
directory that contains it (the conventional filename for the split is appended):
  --counts      counts_<split>.pt              (dump_rule_counts.py CSR groundings)
  --mlp_ranks   ranks_mlp_<split>.csv          (RulE_original trainer, h,r,t,L,H)
  --add_ranks   ranks_<split>.csv              (src_additive precision*binary, h,r,t,L,H)

Both rank CSVs share schema h,r,t,L,H where [L, H) is the filtered tie band and
midpoint_rank = (L + H - 1) / 2 (the convention both trainers' evaluate() use).

Join is ALWAYS on (h, r, t), never on row order. Coverage is printed; a
direction/doubling mismatch (one dump has both head+tail queries, the other only
one direction) silently halves the joined set, so we warn loudly when |joined|
falls well below the smallest input.

delta = rank_add - rank_mlp
  > 0  -> MLP ranks gold higher (lower rank)  -> mlp_win
  < 0  -> precision*binary ranks gold higher  -> add_win
  == 0 -> tie  (exact equality; both midpoints use the identical convention)

Two correctness gates are baked in (see the printed === GATE === blocks):

  * MRR reproduction. From each rank CSV we recompute the exact expectation-MRR
    that evaluate() logs (sum over the [L,H) band). It must match the eval logs
    (precision*binary UMLS test ~ 0.7249, MLP its logged value). A mismatch means
    wrong config / alpha / split or a corrupted dump -- stop and investigate.

  * cofire=0 is the BIAS baseline, not a tie check. The MLP keeps its trained
    self.bias, so a no-rule query scores 0 + bias[gold] and ranks by popularity
    (a real, varying rank). The precision*binary model runs --no_bias, so a
    no-rule query scores exactly 0 and ties with every other ungrounded entity
    (midpoint ~ (N+1)/2). The script TESTS this rather than assuming it: if the
    additive cofire_gold=0 ranks are all buried near (N+1)/2, --no_bias is
    confirmed; if they vary / are small, either the additive carried a bias
    (the baseline interpretation shifts) or the two grounding implementations
    (MinimalGraph in the dump vs data.py in the model) disagree. Because of this
    asymmetry, cofire=0 P(mlp_win) is pure bias and cofire=1 P(mlp_win) is the
    single-rule count nonlinearity -- NEITHER is interaction. The interaction
    claim is specifically the rise across cofire >= 2.

Outputs in --out_dir (default: the counts directory):
  cofire_perquery_<split>.csv        per-query table for later slicing
  cofire_tableA_<split>.csv          among mlp_win: count/% per cofire bucket
  cofire_tableB_<split>.csv          mean/median cofire per outcome group
  cofire_tableC_<split>.csv          per bucket: P(mlp_win/tie/add_win), mean delta
  cofire_significance_<split>.csv    Mann-Whitney U, Spearman, headline numbers

Usage (run from repo root), UMLS aggregation_2x2 example:
    python scripts/analyze_cofire_interaction.py \\
        --counts    outputs/additive_umls_aggregation_2x2/analysis \\
        --mlp_ranks outputs/additive_umls_aggregation_2x2/mlp_eval-umls \\
        --add_ranks outputs/additive_umls_aggregation_2x2/precision_binary \\
        --data_path data/umls \\
        --split     test
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
import torch


# Per-dataset reference MRRs (from each HPC dump's run.log) for the MRR gate.
# Keyed by dataset (data_path basename). Override per run with --ref_mlp_mrr /
# --ref_add_mrr; datasets absent here just print "(no reference)".
REFERENCES = {
    "umls": {
        "mlp":       {"valid": 0.805971, "test": 0.805063},
        "add":       {"valid": 0.726304, "test": 0.724881},
    },
    "family": {
        "mlp":       {"valid": 0.972564, "test": 0.974351},
        "add":       {"valid": 0.956762},
    },
}
MRR_TOL = 2e-3

BUCKETS = ["0", "1", "2", ">=3"]


# ---------------------------------------------------------------------------
# Stats helpers (scipy-free: numpy + math.erfc, normal approximations).
# ---------------------------------------------------------------------------

def _norm_sf(z):
    """Upper-tail standard-normal survival function P(Z > z)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _rankdata(a):
    """Average ranks (1-based), ties shared -- mirrors scipy.stats.rankdata."""
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n, dtype=np.intp)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], n]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x, y):
    """Spearman rho with a large-sample normal-approx two-sided p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan"), n
    rx, ry = _rankdata(x), _rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho = max(-1.0, min(1.0, rho))
    z = abs(rho) * math.sqrt(n - 1)            # normal approximation
    p = 2.0 * _norm_sf(z)
    return rho, p, n


def mannwhitney_u(x, y):
    """Mann-Whitney U for group x vs y with tie-corrected normal-approx p.

    Returns (U_x, p, n_x, n_y). U_x is the U statistic associated with group x;
    U_x > n_x*n_y/2 means x tends to rank higher than y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = x.size, y.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), n1, n2
    allv = np.concatenate([x, y])
    r = _rankdata(allv)
    R1 = float(r[:n1].sum())
    U1 = R1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    # tie correction
    _, counts = np.unique(allv, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))
    sigma2 = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0:
        return U1, float("nan"), n1, n2
    z = (U1 - mu) / math.sqrt(sigma2)
    p = 2.0 * _norm_sf(abs(z))
    return U1, p, n1, n2


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def midpoint_rank(L, H):
    return (L + H - 1) / 2.0


def resolve(path, filename):
    """If `path` is a directory, append `filename`; else use it verbatim."""
    if os.path.isdir(path):
        return os.path.join(path, filename)
    return path


def read_ranks(path):
    """Read an h,r,t,L,H rank CSV -> dict (h,r,t) -> (L, H). Warns on dup keys."""
    out = {}
    dups = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (int(row["h"]), int(row["r"]), int(row["t"]))
            if key in out:
                dups += 1
            out[key] = (int(row["L"]), int(row["H"]))
    if dups:
        print(f"  WARNING: {dups} duplicate (h,r,t) keys in {path} (last kept)")
    return out


def load_cofire(counts_path):
    """Per-query co-fire counts from the grounding CSR.

    For each query the CSR slice [q_ptr[qi], q_ptr[qi+1]) lists (rule_id, ent,
    cnt) groundings. Returns dict (h, r, gold) -> (cofire_total, cofire_gold):
      cofire_total = # distinct rule_id firing anywhere on the query
      cofire_gold  = # distinct rule_id grounding a path to the gold tail
    Empty slice -> (0, 0).
    """
    try:
        counts = torch.load(counts_path, weights_only=False)
    except TypeError:
        counts = torch.load(counts_path)   # torch < 1.13 has no weights_only
    query_h    = counts["query_h"]
    query_r    = counts["query_r"]
    query_gold = counts["query_gold"]
    q_ptr      = counts["q_ptr"]
    rule_id    = counts["rule_id"]
    ent_arr    = counts["ent"]
    Q = query_h.size(0)

    out = {}
    dups = 0
    for qi in range(Q):
        h    = query_h[qi].item()
        r    = query_r[qi].item()
        gold = query_gold[qi].item()
        s    = q_ptr[qi].item()
        e    = q_ptr[qi + 1].item()
        if e > s:
            rids = rule_id[s:e].tolist()
            ents = ent_arr[s:e].tolist()
            cofire_total = len(set(rids))
            cofire_gold  = len({rid for rid, ei in zip(rids, ents) if ei == gold})
        else:
            cofire_total = cofire_gold = 0
        key = (h, r, gold)
        if key in out:
            dups += 1
        out[key] = (cofire_total, cofire_gold)
    if dups:
        print(f"  WARNING: {dups} duplicate (h,r,gold) keys in counts (last kept)")
    return out


def expectation_metrics(lh_iter):
    """Faithful reproduction of trainer.evaluate()'s expectation metrics.

    lh_iter: iterable of (L, H). MRR = mean over queries of
    sum_{rank in [L,H)} (1/rank) / (H-L) -- identical to the trainer loop.
    """
    h1 = h3 = h10 = mr = mrr = 0.0
    n = 0
    for (L, H) in lh_iter:
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


def entity_count(data_path):
    with open(os.path.join(data_path, "entities.dict")) as f:
        return sum(1 for line in f if line.strip())


def bucket(c):
    if c <= 0:  return "0"
    if c == 1:  return "1"
    if c == 2:  return "2"
    return ">=3"


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts",   required=True,
                   help="counts_<split>.pt or the directory containing it.")
    p.add_argument("--mlp_ranks", required=True,
                   help="ranks_mlp_<split>.csv or the directory containing it.")
    p.add_argument("--add_ranks", required=True,
                   help="ranks_<split>.csv (precision*binary) or its directory.")
    p.add_argument("--data_path", default="data/umls",
                   help="Dataset dir (for entities.dict -> N, the tie baseline).")
    p.add_argument("--split",    default="test")
    p.add_argument("--out_dir",  default=None,
                   help="Where to write outputs (default: the counts directory).")
    p.add_argument("--ref_mlp_mrr", type=float, default=None,
                   help="Override the MLP reference MRR for the gate.")
    p.add_argument("--ref_add_mrr", type=float, default=None,
                   help="Override the precision*binary reference MRR for the gate.")
    p.add_argument("--mrr_tol", type=float, default=MRR_TOL)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reporting blocks
# ---------------------------------------------------------------------------

def mrr_gate(label, ranks_dict, ref):
    m = expectation_metrics(ranks_dict.values())
    line = (f"  {label:<14} MRR={m['MRR']:.6f}  Hit@1={m['Hit@1']:.4f}  "
            f"Hit@10={m['Hit@10']:.4f}  (n={m['n']})")
    if ref is None:
        print(line + "  ref=(none)")
        return m["MRR"], None
    delta = abs(m["MRR"] - ref)
    status = "PASS" if delta < MRR_TOL else "FAIL"
    print(line + f"  ref={ref:.6f}  delta={delta:.2e}  [{status}]")
    return m["MRR"], status


def bias_baseline_gate(rows, N):
    """Test the cofire=0 bias asymmetry (additive --no_bias vs MLP popularity)."""
    tie_level = (N + 1) / 2.0
    small_thr = max(10.0, 0.05 * N)
    g0 = [r for r in rows if r["cofire_gold"] == 0]
    print(f"\n=== GATE 2: cofire_gold=0 baseline  (N={N}, tie level (N+1)/2="
          f"{tie_level:.1f}, 'small' rank < {small_thr:.0f}) ===")
    if not g0:
        print("  no cofire_gold=0 queries on this split.")
        return
    n0 = len(g0)
    add_r = np.array([r["rank_add"] for r in g0])
    mlp_r = np.array([r["rank_mlp"] for r in g0])

    # Additive under --no_bias: ungrounded gold scores exactly 0, so it sits in
    # the MIDDLE of the zero-score tie. The band centre (hence the midpoint)
    # averages near (N+1)/2, but filtering known-positives shrinks the candidate
    # set and pulls it below -- so "rank below N/2" is NOT itself suspicious.
    # The real inconsistency is gold ranked NEAR THE TOP (small midpoint AND a
    # narrow tie band): with score 0 that can only happen if gold actually got a
    # nonzero score -> a leftover bias, or a MinimalGraph(dump)-vs-data.py(model)
    # grounding disagreement (counts says cofire=0 but the model scored gold).
    suspects = [r for r in g0 if r["rank_add"] < small_thr and r["span_add"] <= 3]
    print(f"  n(cofire_gold=0) = {n0}")
    print(f"  precision*binary rank_add : mean={add_r.mean():.1f}  "
          f"median={np.median(add_r):.1f}  min={add_r.min():.1f}  "
          f"max={add_r.max():.1f}  std={add_r.std():.1f}")
    print(f"     median vs zero-tie centre (N+1)/2={tie_level:.1f}: "
          f"{'matches' if abs(np.median(add_r)-tie_level) <= 2 else 'OFF'} "
          f"-> ungrounded gold sits in the zero-score tie (supports --no_bias).")
    if not suspects:
        print(f"     -> --no_bias CONSISTENT: no cofire_gold=0 query has gold near "
              f"the top of a narrow band; no rule => no signal.")
    else:
        print(f"     -> WARNING: {len(suspects)} cofire_gold=0 rows have gold ranked "
              f"high in a NARROW band (rank<{small_thr:.0f} & span<=3): "
              f"{[(s['h'], s['r'], s['t'], s['rank_add'], s['span_add']) for s in suspects][:6]}"
              f"{' ...' if len(suspects) > 6 else ''}. Either a leftover bias "
              f"(baseline interpretation shifts) OR the dump (MinimalGraph) and "
              f"model (data.py) groundings disagree. Investigate before the slope.")

    # MLP: expect gold ranked by bias/popularity -> varies, can be small.
    frac_small_mlp = float(np.mean(mlp_r < small_thr))
    print(f"  MLP rank_mlp              : mean={mlp_r.mean():.1f}  "
          f"median={np.median(mlp_r):.1f}  min={mlp_r.min():.1f}  "
          f"max={mlp_r.max():.1f}  std={mlp_r.std():.1f}")
    print(f"     fraction with small rank (< {small_thr:.0f}) = {frac_small_mlp:.3f}"
          f"   (>0 => MLP bias/popularity ranks ungrounded gold; these are NOT "
          f"interaction)")
    n_mlp_win0 = sum(r["outcome"] == "mlp_win" for r in g0)
    print(f"  mlp_win among cofire_gold=0: {n_mlp_win0}/{n0} "
          f"({100*n_mlp_win0/n0:.1f}%) -- attributable to MLP bias, not co-firing.")

    # Informational: cofire_gold>0 but gold still buried in a wide tie. NOT
    # necessarily a mismatch -- if every rule grounding gold has precision 0, the
    # additive score is still 0 and gold legitimately ties in the zero band.
    gpos = [r for r in rows if r["cofire_gold"] > 0]
    if gpos:
        buried = int(np.sum((np.array([r["rank_add"] for r in gpos]) >= N / 2.0)
                            & (np.array([r["span_add"] for r in gpos]) >= N / 4.0)))
        print(f"  info: {buried}/{len(gpos)} cofire_gold>0 rows have gold in a wide "
              f"high tie (rank>=N/2 & span>=N/4) -- expected when gold's rules all "
              f"have precision 0; only alarming if pervasive.")


def analysis_for(rows, cofire_key, out_dir, split, label, writers):
    """Tables A/B/C + significance for one cofire variable (gold or total)."""
    n = len(rows)
    vals = np.array([r[cofire_key] for r in rows], dtype=np.float64)
    deltas = np.array([r["delta"] for r in rows], dtype=np.float64)
    buckets = [bucket(r[cofire_key]) for r in rows]
    outcomes = [r["outcome"] for r in rows]

    print(f"\n{'='*70}\nCO-FIRE VARIABLE: {label}\n{'='*70}")

    # ---- Table A: among mlp_win, count/% per bucket --------------------
    mlp_win_buckets = [b for b, o in zip(buckets, outcomes) if o == "mlp_win"]
    n_mlp_win = len(mlp_win_buckets)
    print(f"\n  TABLE A -- among MLP wins ({n_mlp_win}), distribution over "
          f"{label} buckets:")
    print(f"    {'bucket':<8}{'n':>8}{'% of mlp_win':>16}")
    for b in BUCKETS:
        nb = mlp_win_buckets.count(b)
        pct = 100 * nb / n_mlp_win if n_mlp_win else 0.0
        print(f"    {b:<8}{nb:>8}{pct:>15.1f}%")
        writers["A"].writerow({"cofire_var": label, "bucket": b, "n_mlp_win": nb,
                               "pct_of_mlp_win": round(pct, 2)})
    two = mlp_win_buckets.count("2")
    three_plus = mlp_win_buckets.count(">=3")
    print(f"    -> literal answer: {two} MLP wins at exactly two co-firing rules, "
          f"{three_plus} at three-plus.")

    # ---- Table B: mean/median cofire per outcome -----------------------
    print(f"\n  TABLE B -- {label} per outcome group:")
    print(f"    {'outcome':<10}{'n':>8}{'mean':>10}{'median':>10}")
    for o in ("mlp_win", "tie", "add_win"):
        sub = vals[[i for i in range(n) if outcomes[i] == o]]
        if sub.size:
            mean_c, med_c = float(sub.mean()), float(np.median(sub))
        else:
            mean_c = med_c = 0.0
        print(f"    {o:<10}{sub.size:>8}{mean_c:>10.3f}{med_c:>10.3f}")
        writers["B"].writerow({"cofire_var": label, "outcome": o, "n": int(sub.size),
                               "mean_cofire": round(mean_c, 4),
                               "median_cofire": round(med_c, 4)})

    # ---- Table C: per bucket, P(outcome) + mean delta (the slope) ------
    print(f"\n  TABLE C -- per {label} bucket (THE SLOPE):")
    print(f"    {'bucket':<8}{'n':>7}{'P(mlp_win)':>12}{'P(tie)':>10}"
          f"{'P(add_win)':>12}{'mean_delta':>12}")
    p_mlp_by_bucket = {}
    for b in BUCKETS:
        idx = [i for i in range(n) if buckets[i] == b]
        nb = len(idx)
        if nb:
            p_mlp = sum(outcomes[i] == "mlp_win" for i in idx) / nb
            p_tie = sum(outcomes[i] == "tie"     for i in idx) / nb
            p_add = sum(outcomes[i] == "add_win" for i in idx) / nb
            md = float(deltas[idx].mean())
        else:
            p_mlp = p_tie = p_add = md = 0.0
        p_mlp_by_bucket[b] = (nb, p_mlp)
        print(f"    {b:<8}{nb:>7}{p_mlp:>12.3f}{p_tie:>10.3f}{p_add:>12.3f}"
              f"{md:>12.2f}")
        writers["C"].writerow({"cofire_var": label, "bucket": b, "n": nb,
                               "p_mlp_win": round(p_mlp, 4), "p_tie": round(p_tie, 4),
                               "p_add_win": round(p_add, 4), "mean_delta": round(md, 4)})

    # ---- Significance --------------------------------------------------
    win_vals = vals[[i for i in range(n) if outcomes[i] == "mlp_win"]]
    tie_vals = vals[[i for i in range(n) if outcomes[i] == "tie"]]
    U, p_u, n1, n2 = mannwhitney_u(win_vals, tie_vals)
    rho, p_s, n_s = spearman(vals, deltas)
    direction = ("mlp_win > tie" if (n1 and n2 and U > n1 * n2 / 2)
                 else "mlp_win <= tie")
    print(f"\n  SIGNIFICANCE:")
    print(f"    Mann-Whitney U ({label}, mlp_win vs tie): U={U:.1f}  p={p_u:.3e}  "
          f"[n_win={n1}, n_tie={n2}, {direction}]  (normal approx)")
    print(f"    Spearman({label}, delta): rho={rho:+.4f}  p={p_s:.3e}  "
          f"(n={n_s}, normal approx)")
    writers["S"].writerow({"cofire_var": label,
                           "mannwhitney_U": round(U, 2), "mw_p": p_u,
                           "n_mlp_win": n1, "n_tie": n2, "mw_direction": direction,
                           "spearman_rho": round(rho, 4), "spearman_p": p_s})

    # ---- Headline interpretation (slope, not level) --------------------
    n2_, p2 = p_mlp_by_bucket["2"]
    n3_, p3 = p_mlp_by_bucket[">=3"]
    n1_, p1 = p_mlp_by_bucket["1"]
    n0_, p0 = p_mlp_by_bucket["0"]
    rise = p1 <= p2 <= p3 and (p3 > p1)
    n_ge2 = sum(1 for i in range(n) if vals[i] >= 2)
    frac_ge2 = n_ge2 / n if n else 0.0
    print(f"\n  HEADLINE ({label}):")
    print(f"    intercept (no interaction): P(mlp_win | cofire=0)={p0:.3f} [bias], "
          f"P(mlp_win | cofire=1)={p1:.3f} [single-rule count transform]")
    print(f"    interaction slope: cofire 1->2->>=3  P(mlp_win) = "
          f"{p1:.3f} -> {p2:.3f} -> {p3:.3f}  "
          f"{'(monotone rise: interaction signal)' if rise else '(NOT monotone)'}")
    print(f"    interaction-reachable ceiling: {n_ge2}/{n} = {100*frac_ge2:.1f}% "
          f"of queries have {label} >= 2.")
    return {"label": label, "frac_ge2": frac_ge2, "rise": rise,
            "p0": p0, "p1": p1, "p2": p2, "p3": p3, "rho": rho, "p_s": p_s}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    split = args.split

    counts_path = resolve(args.counts,    f"counts_{split}.pt")
    mlp_path    = resolve(args.mlp_ranks, f"ranks_mlp_{split}.csv")
    add_path    = resolve(args.add_ranks, f"ranks_{split}.csv")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(counts_path))
    os.makedirs(out_dir, exist_ok=True)

    dataset = os.path.basename(os.path.normpath(args.data_path))
    N = entity_count(args.data_path)
    print(f"Dataset={dataset}  split={split}  N(entities)={N}")
    print(f"  counts    : {counts_path}")
    print(f"  mlp_ranks : {mlp_path}")
    print(f"  add_ranks : {add_path}")

    # ---- Load ----------------------------------------------------------
    cofire = load_cofire(counts_path)
    mlp    = read_ranks(mlp_path)
    add    = read_ranks(add_path)

    # ---- Coverage ------------------------------------------------------
    ck, mk, ak = set(cofire), set(mlp), set(add)
    joined = ck & mk & ak
    smallest = min(len(ck), len(mk), len(ak))
    print(f"\n=== COVERAGE (join on (h,r,t)) ===")
    print(f"  |counts|={len(ck)}  |ranks_mlp|={len(mk)}  |ranks_add|={len(ak)}  "
          f"|joined|={len(joined)}")
    print(f"  joined/mlp={len(joined)/len(mk):.3f}  "
          f"joined/add={len(joined)/len(ak):.3f}  "
          f"joined/counts={len(joined)/len(ck):.3f}")
    if len(joined) < 0.9 * smallest:
        print(f"  !!! WARNING: |joined|={len(joined)} is far below the smallest "
              f"input ({smallest}). Likely a direction/doubling mismatch (one dump "
              f"has both head+tail queries, another only one direction). The "
              f"analysis below is restricted to the shared {len(joined)} queries.")

    # ---- GATE 1: MRR reproduction (over the FULL csv, not the join) ----
    refs = REFERENCES.get(dataset, {})
    ref_mlp = args.ref_mlp_mrr if args.ref_mlp_mrr is not None else refs.get("mlp", {}).get(split)
    ref_add = args.ref_add_mrr if args.ref_add_mrr is not None else refs.get("add", {}).get(split)
    print(f"\n=== GATE 1: MRR reproduction (full dump, expectation over [L,H)) ===")
    _, st_mlp = mrr_gate("MLP",             mlp, ref_mlp)
    _, st_add = mrr_gate("precision*binary", add, ref_add)
    if "FAIL" in (st_mlp, st_add):
        print("  !!! MRR gate FAILED -- wrong config/alpha/split or corrupted dump. "
              "Fix before trusting the analysis below.")

    # ---- Build joined rows + classify ----------------------------------
    rows = []
    for key in sorted(joined):
        h, r, t = key
        Lm, Hm = mlp[key]
        La, Ha = add[key]
        rm = midpoint_rank(Lm, Hm)
        ra = midpoint_rank(La, Ha)
        delta = ra - rm                       # >0 => MLP ranks gold higher
        if delta > 0:
            outcome = "mlp_win"
        elif delta < 0:
            outcome = "add_win"
        else:
            outcome = "tie"
        ct, cg = cofire[key]
        rows.append({"h": h, "r": r, "t": t,
                     "rank_mlp": rm, "rank_add": ra, "delta": delta,
                     "outcome": outcome, "cofire_total": ct, "cofire_gold": cg,
                     "span_add": Ha - La, "span_mlp": Hm - Lm})

    n = len(rows)
    n_mlp = sum(r["outcome"] == "mlp_win" for r in rows)
    n_tie = sum(r["outcome"] == "tie"     for r in rows)
    n_add = sum(r["outcome"] == "add_win" for r in rows)
    print(f"\n=== OUTCOMES (delta = rank_add - rank_mlp, exact-equality tie) ===")
    print(f"  mlp_win : {n_mlp:5d}  ({100*n_mlp/n:.1f}%)")
    print(f"  tie     : {n_tie:5d}  ({100*n_tie/n:.1f}%)")
    print(f"  add_win : {n_add:5d}  ({100*n_add/n:.1f}%)")

    # ---- GATE 2: cofire=0 bias baseline --------------------------------
    bias_baseline_gate(rows, N)

    # ---- Per-query CSV -------------------------------------------------
    per_csv = os.path.join(out_dir, f"cofire_perquery_{split}.csv")
    fields = ["h", "r", "t", "rank_mlp", "rank_add", "delta", "outcome",
              "cofire_total", "cofire_gold", "cofire_gold_bucket",
              "cofire_total_bucket"]
    with open(per_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in
                           ("h", "r", "t", "rank_mlp", "rank_add", "delta",
                            "outcome", "cofire_total", "cofire_gold")},
                        "cofire_gold_bucket":  bucket(r["cofire_gold"]),
                        "cofire_total_bucket": bucket(r["cofire_total"])})
    print(f"\nWrote {per_csv}  ({n} queries)")

    # ---- Tables A/B/C + significance, twice (gold, then total) ---------
    fA = open(os.path.join(out_dir, f"cofire_tableA_{split}.csv"), "w", newline="")
    fB = open(os.path.join(out_dir, f"cofire_tableB_{split}.csv"), "w", newline="")
    fC = open(os.path.join(out_dir, f"cofire_tableC_{split}.csv"), "w", newline="")
    fS = open(os.path.join(out_dir, f"cofire_significance_{split}.csv"), "w", newline="")
    writers = {
        "A": csv.DictWriter(fA, fieldnames=["cofire_var", "bucket", "n_mlp_win", "pct_of_mlp_win"]),
        "B": csv.DictWriter(fB, fieldnames=["cofire_var", "outcome", "n", "mean_cofire", "median_cofire"]),
        "C": csv.DictWriter(fC, fieldnames=["cofire_var", "bucket", "n", "p_mlp_win", "p_tie", "p_add_win", "mean_delta"]),
        "S": csv.DictWriter(fS, fieldnames=["cofire_var", "mannwhitney_U", "mw_p", "n_mlp_win", "n_tie", "mw_direction", "spearman_rho", "spearman_p"]),
    }
    for w in writers.values():
        w.writeheader()

    head_gold  = analysis_for(rows, "cofire_gold",  out_dir, split, "cofire_gold",  writers)
    head_total = analysis_for(rows, "cofire_total", out_dir, split, "cofire_total", writers)

    for f in (fA, fB, fC, fS):
        f.close()

    # ---- Final headline (lead with cofire_gold) ------------------------
    print(f"\n{'='*70}\nHEADLINE SUMMARY  (lead: cofire_gold -- interaction acts on the "
          f"rules co-firing\non the same candidate; gold's rules are the cross-terms)"
          f"\n{'='*70}")
    for hd in (head_gold, head_total):
        verdict = ("interaction signal present" if hd["rise"] and hd["rho"] > 0
                   else "no clear interaction signal")
        print(f"  [{hd['label']:<12}] P(mlp_win) 0->1->2->>=3 = "
              f"{hd['p0']:.2f} {hd['p1']:.2f} {hd['p2']:.2f} {hd['p3']:.2f}  | "
              f"Spearman rho={hd['rho']:+.3f}  | cofire>=2 = {100*hd['frac_ge2']:.1f}%  "
              f"-> {verdict}")
    print(f"\n  Read: cofire 0 = bias intercept, cofire 1 = single-rule count "
          f"transform (NOT interaction). The interaction claim is the rise across "
          f"cofire >= 2. The cofire>=2 fraction is the ceiling on what an "
          f"interaction model (e.g. FM) can reach.")
    print("\nDone.")


if __name__ == "__main__":
    main()
