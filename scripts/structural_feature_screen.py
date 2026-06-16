#!/usr/bin/env python3
"""
Structural-feature screen (training-free, embeddings-free).

Follow-up to analyze_grounding_headroom.py. That script showed the gold tail is
usually *grounded* but badly *ranked* among many co-grounded decoys, and that the
re-weightings a GAT can express (binary / degree-norm) recover almost none of the
oracle headroom on WN18RR. This script asks the natural next question:

  Is there a STRUCTURAL property of the grounding paths -- something the scalar
  path-count throws away -- that separates the true target from the decoys, i.e.
  that a path-aware scorer (a GAT) could exploit for a real ranking gain?

It does two complementary things, both in the exact filtered setting used by
trainer.GroundTrainer.evaluate (hr2ooo filtering, tie-averaged ranks, ungrounded
gold -> ~random rank):

 1. DIRECT MRR TEST (decisive). The structural quantities are themselves
    candidate scorers. We rank candidates by each and report filtered MRR vs the
    count baseline, overall and per hubness stratum:
      count   : sum of path mass            (= baseline grounding, attention OFF)
      binary  : # rules that fired to t     (path multiplicity removed)
      norm    : degree-normalised path mass (= GAT attention with uniform weights)
      maxrel  : max single-path reliability (product of 1/in-degree along the best path)
      oracle  : rank gold #1 iff grounded   (ceiling for ANY path re-weighting)

 2. CORRELATION SCREEN (explanatory), reusing feature_validity_screen.py's stats
    (Pearson r, Fisher-z 95% CI, p, Spearman rho, 5-bucket monotonicity). The
    TARGET is corrected for this question:
      struct_gain(q) = RR_maxrel(q) - RR_count(q)   <- where does structure beat the scalar?
      headroom(q)    = RR_oracle(q) - RR_count(q)    <- where is there room at all?
    (NOT rule_advantage = RR_rule - RR_kge, which is a rule-vs-KGE routing target.)
    Features (per grounded-gold query), made RELATIVE to the candidate set where
    ranking-relevant:
      n_paths_gold      log1p(#paths to gold)                 (control; ~ the scalar itself)
      maxrel_margin     reliability(gold) - max reliability(decoy)   (relative; discriminative)
      top_hub_degree    max node-degree on gold's top-reliability path (hub indicator)
      len_spread        path-count-weighted std of rule body lengths to gold
      gold_count_pct    percentile of gold's count among candidates (how the scalar already ranks it)

A report (tables + auto-generated CONCLUSION), a per-feature CSV, and a per-query
CSV are written to outputs/.

Usage:
  python3 scripts/structural_feature_screen.py --dataset wn18rr [--split valid] [--max-per-rel N]
"""

import argparse
import csv
import math
import os
import time
from collections import defaultdict

import numpy as np

# Reuse the validated grounding + filtered-eval machinery.
from analyze_grounding_headroom import (
    build_csr,
    load_dicts,
    read_triples,
)


# --------------------------------------------------------------------------- #
# Stats helpers (mirror feature_validity_screen.py; numpy-only, scipy optional).
# --------------------------------------------------------------------------- #
try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pearson_r(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((xc * yc).sum() / denom)


def _rankdata(x):
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return order.astype(np.float64)


def spearman_r(x, y):
    return pearson_r(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def fisher_ci_p(r, n):
    """Two-sided p and 95% CI for a correlation via Fisher z-transform."""
    if not math.isfinite(r) or n < 4:
        return float("nan"), float("nan"), float("nan")
    if abs(r) >= 1.0:
        return 0.0, r, r
    zr = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    p = 2.0 * (1.0 - _norm_cdf(abs(zr / se)))
    lo = math.tanh(zr - 1.96 * se)
    hi = math.tanh(zr + 1.96 * se)
    return p, lo, hi


def correlation_report(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    r = pearson_r(x, y)
    p, lo, hi = fisher_ci_p(r, n)
    rho = spearman_r(x, y)
    if _scipy_stats is not None and n >= 3:
        try:
            sr, sp = _scipy_stats.pearsonr(x, y)
            if math.isfinite(sr):
                r, p = float(sr), float(sp)
                _, lo, hi = fisher_ci_p(r, n)
            srho, _ = _scipy_stats.spearmanr(x, y)
            if math.isfinite(srho):
                rho = float(srho)
        except Exception:
            pass
    return dict(n=n, r=r, r2=(r * r if math.isfinite(r) else float("nan")),
                p=p, ci_lo=lo, ci_hi=hi, rho=rho)


def is_monotone(seq, tol=1e-9):
    vals = [v for v in seq if math.isfinite(v)]
    if len(vals) < 2:
        return False
    inc = all(vals[i + 1] >= vals[i] - tol for i in range(len(vals) - 1))
    dec = all(vals[i + 1] <= vals[i] + tol for i in range(len(vals) - 1))
    return inc or dec


def quantile_bucket_means(feat, target, num_buckets):
    """Mean target per quantile bucket of feat (edges de-duplicated for ties)."""
    feat = np.asarray(feat, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = feat.size
    if n == 0:
        return [], []
    qs = np.linspace(0, 1, num_buckets + 1)[1:-1]
    edges = np.unique(np.quantile(feat, qs)) if qs.size else np.array([])
    if edges.size == 0:
        return [(f"[{feat.min():.4g},{feat.max():.4g}]", n)], [target.mean()]
    bid = np.digitize(feat, edges, right=False)
    labels, means = [], []
    for b in range(bid.max() + 1):
        idx = np.where(bid == b)[0]
        if idx.size == 0:
            continue
        labels.append((f"[{feat[idx].min():.4g},{feat[idx].max():.4g}]", idx.size))
        means.append(float(target[idx].mean()))
    return labels, means


# --------------------------------------------------------------------------- #
# Path-aware grounding for ONE rule body from a single head.
# --------------------------------------------------------------------------- #
def propagate_struct(head, body, indptr, dst, indeg):
    """Return (idx, count, norm_mass, max_reliability) per reached entity.

    count          : # paths (unit edge weights) -> the scalar grounding.
    norm_mass      : degree-normalised mass (divide by in-degree each hop) =
                     uniform-attention path mass.
    max_reliability: max over paths of prod_hops 1/in-degree(node) = the
                     probability of the single most reliable path (paths through
                     low-degree, specific intermediates score higher than through hubs).
    """
    idx = np.array([head], dtype=np.int64)
    cnt = np.array([1.0], dtype=np.float64)
    nrm = np.array([1.0], dtype=np.float64)
    rel = np.array([1.0], dtype=np.float64)
    for b in body:
        ip, dd = indptr[b], dst[b]
        deg = ip[idx + 1] - ip[idx]
        if int(deg.sum()) == 0:
            return (np.empty(0, np.int64),) + (np.empty(0, np.float64),) * 3
        starts = ip[idx]
        pos = np.concatenate([np.arange(starts[i], starts[i] + deg[i]) for i in range(idx.size)])
        targets = dd[pos]
        w_cnt = np.repeat(cnt, deg)
        w_nrm = np.repeat(nrm, deg)
        w_rel = np.repeat(rel, deg)
        uniq, inv = np.unique(targets, return_inverse=True)
        agg_cnt = np.bincount(inv, weights=w_cnt)
        agg_nrm = np.bincount(inv, weights=w_nrm)
        agg_rel = np.zeros(uniq.size, dtype=np.float64)
        np.maximum.at(agg_rel, inv, w_rel)
        d_in = indeg[b][uniq].astype(np.float64)
        d_in[d_in == 0] = 1.0
        idx, cnt, nrm, rel = uniq, agg_cnt, agg_nrm / d_in, agg_rel / d_in
    return idx, cnt, nrm, rel


def top_path_intermediates(head, gold, body, indptr, dst, indeg):
    """Intermediate nodes on the max-reliability path head -> ... -> gold.

    Layered DP with backpointers (the predecessor maximising reliability at each
    node/hop). Returns the list of intermediate nodes (excluding head and gold),
    or None if gold is unreachable under this body.
    """
    cur = {head: 1.0}
    parents = [dict() for _ in body]
    for k, b in enumerate(body):
        ip, dd = indptr[b], dst[b]
        nxt = {}
        par = parents[k]
        deg_in = indeg[b]
        for s, rs in cur.items():
            for posn in range(ip[s], ip[s + 1]):
                d = int(dd[posn])
                di = deg_in[d] if deg_in[d] > 0 else 1
                val = rs / di
                if d not in nxt or val > nxt[d]:
                    nxt[d] = val
                    par[d] = s
        cur = nxt
        if not cur:
            return None
    if gold not in cur:
        return None
    node = gold
    inter = []
    for k in range(len(body) - 1, -1, -1):
        p = parents[k].get(node)
        if p is None:
            break
        if k > 0:  # k==0 -> predecessor is head, not an intermediate
            inter.append(p)
        node = p
    return inter


# --------------------------------------------------------------------------- #
# Filtered rank (tie-averaged midpoint), matching feature_validity_screen.
# --------------------------------------------------------------------------- #
def filtered_rank(D, gold, fset, N, gold_grounded):
    if not gold_grounded:
        return (N + 1) / 2.0
    val = D.get(gold, 0.0)
    if val <= 0:
        return (N + 1) / 2.0
    gt = eq = 0
    for e, de in D.items():
        if e == gold or e in fset:
            continue
        if de > val:
            gt += 1
        elif de == val:
            eq += 1
    L = gt + 1
    H = gt + eq + 2
    return (L + H - 1) / 2.0


STRATA = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"),
          (21, 100, "21-100"), (101, 10**9, "101+")]


def stratum_of(n_comp):
    n = max(n_comp, 1)
    for lo, hi, key in STRATA:
        if lo <= n <= hi:
            return key
    return "101+"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="wn18rr")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--max-per-rel", type=int, default=0, help="cap queries per relation (0=all)")
    ap.add_argument("--num-buckets", type=int, default=5)
    ap.add_argument("--promising-r", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    base = os.path.join(args.data_root, args.dataset)
    ent2id, rel2id = load_dicts(base)
    N = len(ent2id)
    num_rel = len(rel2id)
    id2rel = {v: k for k, v in rel2id.items()}

    train = read_triples(os.path.join(base, "train.txt"), ent2id, rel2id)
    valid = read_triples(os.path.join(base, "valid.txt"), ent2id, rel2id)
    test = read_triples(os.path.join(base, "test.txt"), ent2id, rel2id)

    # node total degree (hub measure): incident train facts per entity
    node_deg = np.zeros(N, dtype=np.int64)
    for h, r, t in train:
        node_deg[h] += 1
        node_deg[t] += 1

    def enc(h, r):
        return r * N + h

    hr2ooo = defaultdict(set)
    for split in (train, valid, test):
        for h, r, t in split:
            hr2ooo[enc(h, r)].add(t)
            hr2ooo[enc(t, r + num_rel)].add(h)

    raw = valid if args.split == "valid" else test
    queries_by_rel = defaultdict(list)
    for h, r, t in raw:
        queries_by_rel[r].append((h, r, t))
        queries_by_rel[r + num_rel].append((t, r + num_rel, h))

    rules_by_head = defaultdict(list)
    with open(os.path.join(base, "mined_rules.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            toks = [int(x) for x in p]
            rules_by_head[toks[0]].append(toks[1:])

    indptr, dst, indeg = build_csr(N, num_rel, train)

    # ----- accumulators -----
    schemes = ["count", "binary", "norm", "maxrel", "oracle"]
    mrr = {s: 0.0 for s in schemes}
    h1 = {s: 0.0 for s in schemes}
    strat_mrr = {s: defaultdict(float) for s in schemes}
    strat_n = defaultdict(int)
    strat_cov = defaultdict(float)
    n_queries = 0
    n_covered = 0

    feat_names = ["n_paths_gold", "maxrel_margin", "top_hub_degree",
                  "len_spread", "gold_count_pct"]
    feats = {k: [] for k in feat_names}
    tgt_struct_gain = []
    tgt_headroom = []
    perquery_rows = []

    t0 = time.time()
    for qrel, qlist in sorted(queries_by_rel.items()):
        rules = rules_by_head.get(qrel, [])
        if args.max_per_rel and len(qlist) > args.max_per_rel:
            qlist = random.sample(qlist, args.max_per_rel)

        for (h, r, gold) in qlist:
            n_queries += 1
            score_sum = defaultdict(float)
            score_norm = defaultdict(float)
            rule_hit = defaultdict(int)
            max_rel = defaultdict(float)
            gold_lens, gold_counts = [], []
            best_rel_gold, best_body = -1.0, None

            for body in rules:
                idx, cnt, nrm, rel = propagate_struct(h, body, indptr, dst, indeg)
                if idx.size == 0:
                    continue
                for e, cv, nv, rv in zip(idx.tolist(), cnt.tolist(), nrm.tolist(), rel.tolist()):
                    score_sum[e] += cv
                    score_norm[e] += nv
                    rule_hit[e] += 1
                    if rv > max_rel[e]:
                        max_rel[e] = rv
                # gold-specific structural bookkeeping
                gi = np.searchsorted(idx, gold)
                if gi < idx.size and idx[gi] == gold:
                    gold_lens.append(len(body))
                    gold_counts.append(float(cnt[gi]))
                    if rel[gi] > best_rel_gold:
                        best_rel_gold = float(rel[gi])
                        best_body = body

            grounded = set(score_sum.keys())
            gold_grounded = (gold in grounded) and (score_sum[gold] > 0)
            if gold_grounded:
                n_covered += 1

            comp = [e for e in grounded if e not in hr2ooo.get(enc(h, r), set())]
            n_comp = len(comp) + (1 if gold_grounded else 0)
            skey = stratum_of(n_comp)
            strat_n[skey] += 1
            if gold_grounded:
                strat_cov[skey] += 1.0

            fset = hr2ooo.get(enc(h, r), set())
            score_binary = {e: float(rule_hit[e]) for e in grounded}
            ranks = {
                "count": filtered_rank(score_sum, gold, fset, N, gold_grounded),
                "binary": filtered_rank(score_binary, gold, fset, N, gold_grounded),
                "norm": filtered_rank(score_norm, gold, fset, N, gold_grounded),
                "maxrel": filtered_rank(max_rel, gold, fset, N, gold_grounded),
                "oracle": 1.0 if gold_grounded else (N + 1) / 2.0,
            }
            for s in schemes:
                rr = 1.0 / ranks[s]
                mrr[s] += rr
                h1[s] += 1.0 if ranks[s] <= 1.0 else 0.0
                strat_mrr[s][skey] += rr

            # ----- structural features (grounded-gold queries only) -----
            if gold_grounded:
                rr_count = 1.0 / ranks["count"]
                rr_maxrel = 1.0 / ranks["maxrel"]
                rr_oracle = 1.0 / ranks["oracle"]
                struct_gain = rr_maxrel - rr_count
                headroom = rr_oracle - rr_count

                n_paths = math.log1p(score_sum[gold])
                decoy_rel = max((max_rel[e] for e in comp if e != gold), default=0.0)
                maxrel_margin = max_rel[gold] - decoy_rel
                inter = top_path_intermediates(h, gold, best_body, indptr, dst, indeg) \
                    if best_body is not None else []
                top_hub = float(max((node_deg[n] for n in inter), default=0))
                if gold_counts and len(set(gold_lens)) > 1:
                    w = np.asarray(gold_counts)
                    L = np.asarray(gold_lens, dtype=np.float64)
                    m = (w * L).sum() / w.sum()
                    len_spread = float(math.sqrt((w * (L - m) ** 2).sum() / w.sum()))
                else:
                    len_spread = 0.0
                # percentile of gold's count among candidates (0..1)
                cand_counts = np.fromiter((score_sum[e] for e in comp), dtype=np.float64) \
                    if comp else np.array([score_sum[gold]])
                gold_count_pct = float((cand_counts < score_sum[gold]).mean())

                feats["n_paths_gold"].append(n_paths)
                feats["maxrel_margin"].append(maxrel_margin)
                feats["top_hub_degree"].append(top_hub)
                feats["len_spread"].append(len_spread)
                feats["gold_count_pct"].append(gold_count_pct)
                tgt_struct_gain.append(struct_gain)
                tgt_headroom.append(headroom)
                perquery_rows.append(dict(
                    rel=id2rel.get(qrel % num_rel, str(qrel)) + ("_inv" if qrel >= num_rel else ""),
                    n_comp=n_comp, rr_count=rr_count, rr_maxrel=rr_maxrel,
                    struct_gain=struct_gain, headroom=headroom,
                    n_paths_gold=n_paths, maxrel_margin=maxrel_margin,
                    top_hub_degree=top_hub, len_spread=len_spread,
                    gold_count_pct=gold_count_pct))

    dt = time.time() - t0

    # ----- build report -----
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out(f"\n{'='*78}")
    out(f"STRUCTURAL-FEATURE SCREEN | dataset={args.dataset} | split={args.split}")
    out(f"{'='*78}")
    out(f"queries={n_queries}  grounded_gold(coverage)={n_covered/n_queries:.4f}  "
        f"rules={sum(len(v) for v in rules_by_head.values())}  ({dt:.1f}s)")
    if args.max_per_rel:
        out(f"(subsampled: max {args.max_per_rel} queries per relation)")

    out(f"\n[1] DIRECT MRR TEST -- does a structural re-scorer beat the count baseline?")
    out(f"{'scheme':<10}{'MRR':>9}{'Hit@1':>9}{'dMRR vs count':>15}")
    base_mrr = mrr["count"] / n_queries
    for s in schemes:
        m = mrr[s] / n_queries
        d = m - base_mrr
        tag = "" if s == "count" else f"{d:+.4f}"
        out(f"{s:<10}{m:>9.4f}{h1[s]/n_queries:>9.4f}{tag:>15}")

    out(f"\n   by hubness (#grounded competitors), MRR:")
    out(f"{'stratum':<9}{'#q':>7}{'cov':>8}{'count':>9}{'binary':>9}{'norm':>9}"
        f"{'maxrel':>9}{'oracle':>9}")
    for _, _, k in STRATA:
        c = strat_n.get(k, 0)
        if c == 0:
            continue
        row = f"{k:<9}{c:>7}{strat_cov.get(k,0.0)/c:>8.3f}"
        for s in schemes:
            row += f"{strat_mrr[s].get(k,0.0)/c:>9.4f}"
        out(row)

    out(f"\n[2] CORRELATION SCREEN (grounded-gold queries; N={len(tgt_struct_gain)})")
    out(f"    target A = struct_gain (RR_maxrel - RR_count): where structure beats the scalar")
    out(f"    target B = headroom    (RR_oracle - RR_count): where ANY re-weighting has room")
    feat_rows = []
    for tname, target in (("struct_gain", tgt_struct_gain), ("headroom", tgt_headroom)):
        out(f"\n  vs {tname}:")
        out(f"  {'feature':>16}{'r':>8}{'95% CI':>18}{'p':>10}{'rho':>8}"
            f"{'monotone':>10}{'flag':>6}")
        out("  " + "-" * 76)
        for fname in feat_names:
            x = feats[fname]
            if len(x) < 4 or len(set(x)) < 2:
                out(f"  {fname:>16}{'n/a':>8}")
                continue
            rep = correlation_report(x, target)
            _, bmeans = quantile_bucket_means(x, target, args.num_buckets)
            mono = is_monotone(bmeans)
            promising = (math.isfinite(rep["r"]) and abs(rep["r"]) >= args.promising_r
                         and mono and not (rep["ci_lo"] <= 0 <= rep["ci_hi"]))
            ci = f"[{rep['ci_lo']:+.3f},{rep['ci_hi']:+.3f}]"
            out(f"  {fname:>16}{rep['r']:>+8.3f}{ci:>18}{rep['p']:>10.2e}"
                f"{rep['rho']:>+8.3f}{('yes' if mono else 'no'):>10}"
                f"{('YES' if promising else ''):>6}")
            feat_rows.append(dict(target=tname, feature=fname, r=rep["r"],
                                  r2=rep["r2"], p=rep["p"], ci_lo=rep["ci_lo"],
                                  ci_hi=rep["ci_hi"], rho=rep["rho"],
                                  monotone=mono, promising=promising))

    # ----- auto conclusion -----
    out(f"\n{'-'*78}")
    out("CONCLUSION")
    out(f"{'-'*78}")
    d_maxrel = mrr["maxrel"] / n_queries - base_mrr
    d_binary = mrr["binary"] / n_queries - base_mrr
    d_norm = mrr["norm"] / n_queries - base_mrr
    gap_oracle = mrr["oracle"] / n_queries - base_mrr
    best_struct = max(d_maxrel, d_binary, d_norm)
    captured = (best_struct / gap_oracle * 100.0) if gap_oracle > 1e-9 else float("nan")
    out(f"- Oracle headroom over count (MRR)         : {gap_oracle:+.4f}")
    out(f"- Best structural re-scorer gain over count: {best_struct:+.4f} "
        f"({captured:.1f}% of the headroom)")
    out(f"  (maxrel {d_maxrel:+.4f} | binary {d_binary:+.4f} | norm {d_norm:+.4f})")
    promising_any = [fr for fr in feat_rows if fr["promising"]]
    if best_struct >= 0.01:
        out(f"- VERDICT: a path-structural scorer DOES improve ranking here "
            f"(+{best_struct:.4f} MRR). Worth pursuing a path-aware grounding.")
    else:
        out(f"- VERDICT: NO structural re-scorer meaningfully beats the scalar count "
            f"({best_struct:+.4f} MRR), despite {gap_oracle:+.4f} of oracle headroom.")
        out(f"  The headroom is per-instance disambiguation that path structure alone "
            f"does not supply -> a GAT over these groundings is not the lever.")
    if promising_any:
        names = ", ".join(sorted({fr["feature"] for fr in promising_any}))
        out(f"- Usable routing signal (|r|>={args.promising_r}, monotone, CI excludes 0): {names}")
    else:
        out(f"- No structural feature clears the routing bar (|r|>={args.promising_r}, "
            f"monotone, CI excludes 0): structure does not even predict where the "
            f"scalar fails.")

    # ----- write outputs -----
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"structural_screen_{args.dataset}_{args.split}"
    report_path = os.path.join(args.out_dir, stem + ".txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    feat_csv = os.path.join(args.out_dir, stem + "_features.csv")
    if feat_rows:
        with open(feat_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(feat_rows[0].keys()))
            w.writeheader()
            w.writerows(feat_rows)
    pq_csv = os.path.join(args.out_dir, stem + "_perquery.csv")
    if perquery_rows:
        with open(pq_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(perquery_rows[0].keys()))
            w.writeheader()
            w.writerows(perquery_rows)

    print(f"\nSaved:\n  {report_path}\n  {feat_csv}\n  {pq_csv}")


if __name__ == "__main__":
    main()
