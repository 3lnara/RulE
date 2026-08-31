#!/usr/bin/env python3
"""Offline scoring helpers for the counts dumped by dump_rule_counts.py.

Helper module only (no CLI): build_hr2ooo (filtered-candidate mask),
score_queries (per-rule weighted sum -> filtered L/H ranks) and
expectation_metrics (the [L, H) expectation MRR / Hit@k). Imported by
select_logreg.py and analyze_rule_dropone.py.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


# hr2ooo filter, mirroring data.py: enc(h, r) = r * N + h.
def build_hr2ooo(data_path: str, N: int, num_rel: int) -> Dict[int, set]:
    """Build hr2ooo from all three splits + both directions."""
    hr2ooo: Dict[int, set] = defaultdict(set)

    def enc(h, r):
        return r * N + h

    ent2id = {}
    rel2id = {}
    with open(os.path.join(data_path, "entities.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            ent2id[name] = int(idx)
    with open(os.path.join(data_path, "relations.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            rel2id[name] = int(idx)

    for fname in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(data_path, fname)) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h = ent2id[parts[0]]
                r = rel2id[parts[1]]
                t = ent2id[parts[2]]
                hr2ooo[enc(h, r)].add(t)
                hr2ooo[enc(t, r + num_rel)].add(h)

    return hr2ooo


# ---------------------------------------------------------------------------
# Core scoring helper (importable by analyze_binary_vs_raw_wins.py)
# ---------------------------------------------------------------------------

def score_queries(
    query_h: torch.Tensor,
    query_r: torch.Tensor,
    query_gold: torch.Tensor,
    q_ptr: torch.Tensor,
    rule_id: torch.Tensor,
    ent: torch.Tensor,
    cnt: torch.Tensor,
    w_R: torch.Tensor,
    basis: str,
    clamp: bool,
    hr2ooo: Dict[int, set],
    N: int,
) -> List[Tuple[int, int, int, int, int]]:
    """Score all queries and return per-query (h, r, gold, L, H) list.

    basis : 'binary' or 'raw'
    clamp : whether to zero negative w_R
    Returns list of (h, r, gold, L, H) — L/H are the filtered rank range.
    """
    w_eff = w_R.clamp(min=0.0) if clamp else w_R.clone()

    def enc(h, r):
        return r * N + h

    results = []
    Q = query_h.size(0)
    for qi in range(Q):
        h     = query_h[qi].item()
        r     = query_r[qi].item()
        gold  = query_gold[qi].item()
        start = q_ptr[qi].item()
        stop  = q_ptr[qi + 1].item()

        # Build score vector (float64 for precision)
        score = torch.zeros(N, dtype=torch.float64)
        if start < stop:
            r_ids  = rule_id[start:stop]
            ents   = ent[start:stop]
            cnts   = cnt[start:stop].long()
            w_vals = w_eff[r_ids].double()
            if basis == "binary":
                vals = w_vals            # weight * 1
            else:
                vals = w_vals * cnts.double()   # weight * count
            score.scatter_add_(0, ents, vals)

        # flag = True for entities not in hr2ooo (competitors); gold excluded too,
        # mirroring ValidDataset/TestDataset and the model eval.
        known_true = hr2ooo.get(enc(h, r), set())
        flag = torch.ones(N, dtype=torch.bool)
        for t in known_true:
            flag[t] = False   # includes gold itself

        val  = score[gold].item()
        n_gt = int((score[flag] > val).sum().item())
        n_ge = int((score[flag] >= val).sum().item())
        L    = n_gt + 1
        H    = n_ge + 2     # = (count >= val) + 2, matches trainer.evaluate

        results.append((h, r, gold, L, H))

    return results


def expectation_metrics(
    results: List[Tuple[int, int, int, int, int]]
) -> Dict[str, float]:
    """Replicate trainer.evaluate expectation over [L, H)."""
    h1 = h3 = h10 = mr = mrr = 0.0
    n = len(results)
    for (_, _, _, L, H) in results:
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
    return {
        "MRR":    mrr  / n,
        "MR":     mr   / n,
        "Hit@1":  h1   / n,
        "Hit@3":  h3   / n,
        "Hit@10": h10  / n,
        "n":      n,
    }


def _load_entity_rel_sizes(data_path: str):
    """Return (N, num_rel) from entities.dict and relations.dict."""
    ent2id = {}
    with open(os.path.join(data_path, "entities.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            ent2id[name] = int(idx)
    rel2id = {}
    with open(os.path.join(data_path, "relations.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            rel2id[name] = int(idx)
    return len(ent2id), len(rel2id)

