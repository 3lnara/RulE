#!/usr/bin/env python3
"""
Grounding-headroom analysis: can rule-path re-weighting (e.g. a GAT) help?

This reimplements RulE's rule grounding in pure NumPy (no trained embeddings,
no GPU) and ranks candidate tails under four scoring schemes, in the exact
filtered setting used by trainer.GroundTrainer.evaluate (hr2ooo filtering,
expectation-based ranks, ungrounded gold -> uniform random rank).

Schemes (all aggregate over the rules whose head == query relation):
  sum   : score(t) = sum_rules  #paths(h->t via rule body)        [= baseline grounding, attention OFF]
  binary: score(t) = sum_rules  1[#paths(h->t) > 0]               [within-rule path multiplicity removed]
  norm  : score(t) = sum_rules  degree-normalised path mass       [= GAT attention with *uniform* weights;
                                                                    isolates the per-target softmax normalisation]
  oracle: rank gold #1 whenever it is grounded at all             [= ceiling for ANY path re-weighting, incl. a perfect GAT]

Key facts this exposes:
  * coverage (fraction of queries whose gold has >=1 grounding) == oracle MRR ceiling,
    and it is identical for sum/binary/norm (re-weighting cannot create groundings).
  * gap(sum -> oracle) = the most a better intra-grounding weighting could ever add.
  * sum vs binary  -> does raw path multiplicity carry signal?
  * sum vs norm    -> does the per-target normalisation a GAT *forces* help or hurt?
  * stratified by #grounded competitors ("hubness") -> is there headroom on hub-like queries?

Usage:
  python3 scripts/analyze_grounding_headroom.py --dataset wn18rr [--max-per-rel N] [--seed 0]
"""

import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np


def load_dicts(base):
    ent2id, rel2id = {}, {}
    with open(os.path.join(base, "entities.dict")) as f:
        for line in f:
            i, name = line.rstrip("\n").split("\t")
            ent2id[name] = int(i)
    with open(os.path.join(base, "relations.dict")) as f:
        for line in f:
            i, name = line.rstrip("\n").split("\t")
            rel2id[name] = int(i)
    return ent2id, rel2id


def read_triples(path, ent2id, rel2id):
    out = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            out.append((ent2id[p[0]], rel2id[p[1]], ent2id[p[2]]))
    return out


def build_csr(num_nodes, num_rel, train):
    """CSR out-adjacency for every directed relation (forward r and inverse r+num_rel).

    Mirrors KnowledgeGraph: forward fact (h,r,t) -> edge h->t under r and t->h under r+num_rel.
    Returns indptr[d], dst[d], indeg[d] (full in-degree per target node) for d in [0, 2*num_rel).
    """
    D = 2 * num_rel
    src_lists = [[] for _ in range(D)]
    dst_lists = [[] for _ in range(D)]
    for h, r, t in train:
        src_lists[r].append(h); dst_lists[r].append(t)
        ri = r + num_rel
        src_lists[ri].append(t); dst_lists[ri].append(h)

    indptr = [None] * D
    dst = [None] * D
    indeg = [None] * D
    for d in range(D):
        s = np.asarray(src_lists[d], dtype=np.int64)
        t = np.asarray(dst_lists[d], dtype=np.int64)
        order = np.argsort(s, kind="stable")
        s, t = s[order], t[order]
        ip = np.zeros(num_nodes + 1, dtype=np.int64)
        np.add.at(ip, s + 1, 1)
        np.cumsum(ip, out=ip)
        indptr[d] = ip
        dst[d] = t
        deg = np.zeros(num_nodes, dtype=np.int64)
        if t.size:
            np.add.at(deg, t, 1)
        indeg[d] = deg
    return indptr, dst, indeg


def propagate(head, body, indptr, dst, indeg, normalize):
    """Grounding of a single rule body from a single head node.

    Returns (ent_idx, value) sparse vector of path mass reaching each entity.
    normalize=True divides by the per-target in-degree after every hop
    (the GAT's per-target softmax with uniform weights).
    """
    cur_idx = np.array([head], dtype=np.int64)
    cur_val = np.array([1.0], dtype=np.float64)
    for b in body:
        ip, dd = indptr[b], dst[b]
        deg = cur_idx[1:] - cur_idx[:-1] if False else (ip[cur_idx + 1] - ip[cur_idx])
        total = int(deg.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.float64)
        # gather neighbour edge positions for the whole frontier
        starts = ip[cur_idx]
        pos = np.concatenate([np.arange(starts[i], starts[i] + deg[i]) for i in range(cur_idx.size)])
        targets = dd[pos]
        weights = np.repeat(cur_val, deg)
        uniq, inv = np.unique(targets, return_inverse=True)
        agg = np.bincount(inv, weights=weights)
        if normalize:
            agg = agg / indeg[b][uniq]
        cur_idx, cur_val = uniq, agg
    return cur_idx, cur_val


def expectation_rank_metrics(L, H, N):
    """Replicate evaluate(): expectation over tied ranks in [L, H)."""
    if H <= L:
        H = L + 1
    span = H - L
    h1 = h3 = h10 = mrr = 0.0
    for rank in range(L, H):
        if rank <= 1:
            h1 += 1.0 / span
        if rank <= 3:
            h3 += 1.0 / span
        if rank <= 10:
            h10 += 1.0 / span
        mrr += (1.0 / rank) / span
    return h1, h3, h10, mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wn18rr")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--max-per-rel", type=int, default=0, help="cap queries per relation (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    base = os.path.join(args.data_root, args.dataset)
    ent2id, rel2id = load_dicts(base)
    N = len(ent2id)
    num_rel = len(rel2id)

    train = read_triples(os.path.join(base, "train.txt"), ent2id, rel2id)
    valid = read_triples(os.path.join(base, "valid.txt"), ent2id, rel2id)
    test = read_triples(os.path.join(base, "test.txt"), ent2id, rel2id)

    # hr2ooo filter (train+valid+test, both directions); encode_hr = r*N + h
    def enc(h, r):
        return r * N + h
    hr2ooo = defaultdict(set)
    for split in (train, valid, test):
        for h, r, t in split:
            hr2ooo[enc(h, r)].add(t)
            hr2ooo[enc(t, r + num_rel)].add(h)

    # queries from the chosen split, both directions (matches *_facts construction)
    raw = valid if args.split == "valid" else test
    queries_by_rel = defaultdict(list)
    for h, r, t in raw:
        queries_by_rel[r].append((h, r, t))
        queries_by_rel[r + num_rel].append((t, r + num_rel, h))

    # rules grouped by head relation: line = head body...
    rules_by_head = defaultdict(list)
    with open(os.path.join(base, "mined_rules.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            toks = [int(x) for x in p]
            rules_by_head[toks[0]].append(toks[1:])

    indptr, dst, indeg = build_csr(N, num_rel, train)

    # accumulators
    schemes = ["sum", "binary", "norm", "oracle"]
    agg = {s: dict(h1=0.0, h3=0.0, h10=0.0, mrr=0.0) for s in schemes}
    n_queries = 0
    n_covered = 0
    # hubness strata by #grounded competitors (after filtering)
    strata = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 10**9)]
    strat_keys = ["1", "2-5", "6-20", "21-100", "101+"]
    strat_count = defaultdict(int)
    strat_mrr = {s: defaultdict(float) for s in schemes}
    strat_cov = defaultdict(float)

    t0 = time.time()
    for qrel, qlist in sorted(queries_by_rel.items()):
        rules = rules_by_head.get(qrel, [])
        if args.max_per_rel and len(qlist) > args.max_per_rel:
            qlist = random.sample(qlist, args.max_per_rel)
        if not rules:
            # no rules for this relation: nothing is ever grounded
            for (h, r, gold) in qlist:
                n_queries += 1
                for s in schemes:
                    h1, h3, h10, mrr = expectation_rank_metrics(1, N + 1, N)
                    agg[s]["h1"] += h1; agg[s]["h3"] += h3; agg[s]["h10"] += h10; agg[s]["mrr"] += mrr
                strat_count["1"] += 1
            continue

        for (h, r, gold) in qlist:
            n_queries += 1
            score_sum = defaultdict(float)
            score_norm = defaultdict(float)
            rule_hit = defaultdict(int)
            for body in rules:
                idx, val = propagate(h, body, indptr, dst, indeg, normalize=False)
                for e, v in zip(idx.tolist(), val.tolist()):
                    score_sum[e] += v
                    rule_hit[e] += 1
                idn, vn = propagate(h, body, indptr, dst, indeg, normalize=True)
                for e, v in zip(idn.tolist(), vn.tolist()):
                    score_norm[e] += v

            fset = hr2ooo.get(enc(h, r), set())
            grounded = set(score_sum.keys())
            gold_grounded = gold in grounded and score_sum[gold] > 0
            if gold_grounded:
                n_covered += 1

            # competitors = grounded entities, not known-true (gold handled separately)
            comp = [e for e in grounded if e not in fset]
            n_comp = len(comp) + (1 if gold_grounded else 0)
            skey = next(k for (lo, hi), k in zip(strata, strat_keys) if lo <= max(n_comp, 1) <= hi)
            strat_count[skey] += 1
            if gold_grounded:
                strat_cov[skey] += 1.0

            score_binary = {e: rule_hit[e] for e in grounded}

            for s in schemes:
                if s == "oracle":
                    if gold_grounded:
                        L, H = 1, 2  # rank 1
                    else:
                        L, H = 1, N + 1
                    h1, h3, h10, mrr = expectation_rank_metrics(L, H, N)
                else:
                    D = score_sum if s == "sum" else (score_binary if s == "binary" else score_norm)
                    if gold_grounded and D.get(gold, 0.0) > 0:
                        val = D[gold]
                        gt = eq = 0
                        for e in comp:
                            de = D.get(e, 0.0)
                            if de > val:
                                gt += 1
                            elif de == val:
                                eq += 1
                        L = gt + 1
                        H = gt + eq + 2
                        h1, h3, h10, mrr = expectation_rank_metrics(L, H, N)
                    else:
                        h1, h3, h10, mrr = expectation_rank_metrics(1, N + 1, N)
                agg[s]["h1"] += h1; agg[s]["h3"] += h3; agg[s]["h10"] += h10; agg[s]["mrr"] += mrr
                if s != "oracle":
                    strat_mrr[s][skey] += mrr
                else:
                    strat_mrr[s][skey] += mrr

    dt = time.time() - t0
    print(f"\n=== {args.dataset.upper()} [{args.split}] grounding headroom ===")
    print(f"queries={n_queries}  rules={sum(len(v) for v in rules_by_head.values())}  "
          f"coverage(grounded gold)={n_covered/n_queries:.4f}  ({dt:.1f}s)")
    print(f"{'scheme':<8}{'MRR':>9}{'Hit@1':>9}{'Hit@3':>9}{'Hit@10':>9}")
    for s in schemes:
        print(f"{s:<8}{agg[s]['mrr']/n_queries:>9.4f}{agg[s]['h1']/n_queries:>9.4f}"
              f"{agg[s]['h3']/n_queries:>9.4f}{agg[s]['h10']/n_queries:>9.4f}")

    print(f"\n--- by hubness (#grounded competitors), MRR ---")
    print(f"{'stratum':<8}{'#q':>7}{'cov':>8}{'sum':>9}{'binary':>9}{'norm':>9}{'oracle':>9}")
    for k in strat_keys:
        c = strat_count.get(k, 0)
        if c == 0:
            continue
        cov = strat_cov.get(k, 0.0) / c
        row = f"{k:<8}{c:>7}{cov:>8.3f}"
        for s in schemes:
            row += f"{strat_mrr[s].get(k,0.0)/c:>9.4f}"
        print(row)


if __name__ == "__main__":
    main()
