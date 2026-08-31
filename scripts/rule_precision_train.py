#!/usr/bin/env python3
"""Compute per-rule empirical train PCA precision.

precision_R = #{(h,t): body grounds h->t and t is a true r-tail of h} over
#{(h,t): body grounds h->t and h has >=1 true r-tail}. Needs no checkpoint;
writes rule_precision.pt.

Usage:
    python scripts/rule_precision_train.py --data_path data/<ds> \\
        --rule_file data/<ds>/mined_rules.txt --out_dir outputs/<ds>/rq
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bootstrap: reuse MinimalGraph and load_rules from dump_rule_counts.py.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

sys.path.insert(0, _SCRIPT_DIR)
import dump_rule_counts as _drc
MinimalGraph = _drc.MinimalGraph
load_rules   = _drc.load_rules


# gold[r][h] = set of true train tails for directed relation r, head h.
def build_gold(graph: MinimalGraph) -> dict:
    """gold[r][h] = set of true train tails (train_facts already has both directions)."""
    gold: dict = defaultdict(lambda: defaultdict(set))
    for h, r, t in graph.train_facts:
        gold[r][h].add(t)
    return gold


def compute_precision(
    graph: MinimalGraph,
    rules: list,
    gold: dict,
    device: torch.device = None,
    chunk_size: int = None,
) -> tuple:
    """Return (fired_total, gold_fired) int64 arrays by rule_id (denominator = firings
    from known-subject heads, numerator = those landing on a true tail)."""
    if device is None:
        device = torch.device("cpu")
    R = len(rules)
    fired_total = np.zeros(R, dtype=np.int64)
    gold_fired  = np.zeros(R, dtype=np.int64)

    # Index rules by head relation.
    relation2rules: dict = defaultdict(list)
    for rule in rules:
        gid    = rule[0]
        r_head = rule[1]
        body   = rule[2:]
        relation2rules[r_head].append((gid, r_head, body))

    N = graph.entity_size

    t0 = time.time()
    for r, rule_list in sorted(relation2rules.items()):
        known_h = sorted(gold[r].keys())
        if not known_h:
            continue
        B = len(known_h)
        step = B if (chunk_size is None or chunk_size <= 0) else chunk_size

        for blk in range(0, B, step):
            block_h = known_h[blk:blk + step]
            Bk = len(block_h)
            H_t = torch.tensor(block_h, dtype=torch.long, device=device)

            # gold_mask[i, j] = True iff j in gold[r][block_h[i]]
            gold_mask = torch.zeros(Bk, N, dtype=torch.bool, device=device)
            for i, h in enumerate(block_h):
                for t in gold[r][h]:
                    gold_mask[i, t] = True

            for gid, r_head, body in rule_list:
                with torch.no_grad():
                    counts = graph.grounding(H_t, r_head, body, None)  # [Bk, N]
                    fired  = counts > 0                                  # [Bk, N]

                fired_total[gid] += int(fired.sum().item())
                gold_fired[gid]  += int((fired & gold_mask).sum().item())

    elapsed = time.time() - t0
    print(f"  Grounding done in {elapsed:.1f}s  "
          f"  rules with support>0: {int((fired_total > 0).sum())}/{R}")

    return fired_total, gold_fired


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_path",    default="data/umls",
                   help="Dataset directory (entities.dict, relations.dict, train.txt, ...).")
    p.add_argument("--rule_file",    default=None,
                   help="mined_rules.txt (defaults to <data_path>/mined_rules.txt).")
    p.add_argument("--out_dir",      default="outputs/additive_umls_aggregation_2x2/analysis",
                   help="Where to write outputs.")
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda"],
                   help="Device for grounding (cuda scales to WN18RR/FB15k).")
    p.add_argument("--chunk_size",   type=int, default=0,
                   help="Head-block size to bound [B, N] mask memory; 0 = no chunking.")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    rule_file = args.rule_file or os.path.join(args.data_path, "mined_rules.txt")
    dataset   = os.path.basename(os.path.normpath(args.data_path))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  WARNING: --device cuda requested but CUDA is unavailable; "
              "falling back to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"dataset   : {dataset}")
    print(f"data_path : {args.data_path}")
    print(f"rule_file : {rule_file}")
    print(f"out_dir   : {args.out_dir}")
    print(f"device    : {device}  chunk_size: {args.chunk_size or 'none'}")

    print("\nLoading graph...")
    graph = MinimalGraph(args.data_path)
    NR = graph.relation_size
    N  = graph.entity_size
    print(f"  entities={N}  directed_relations={NR * 2}  train_facts={len(graph.train_facts)}")

    print("Loading rules...")
    rules = load_rules(rule_file, NR)
    R = len(rules)
    print(f"  num_rules={R}")

    # gid == list position, so these align with precision[gid].
    lengths_meta = np.array([len(r[2:]) for r in rules], dtype=np.int64)
    heads_meta   = np.array([r[1]       for r in rules], dtype=np.int64)

    print("\nBuilding gold dict from train facts...")
    gold = build_gold(graph)
    print(f"  unique (r, h) pairs with >=1 gold tail: {sum(len(v) for v in gold.values())}")

    print("\nComputing PCA precision (grounding from known-subject heads)...")
    fired_total, gold_fired = compute_precision(
        graph, rules, gold, device=device, chunk_size=args.chunk_size)

    support   = fired_total
    precision = np.where(support > 0,
                         gold_fired / support.astype(np.float64),
                         float("nan")).astype(np.float32)

    has_prec = np.isfinite(precision)
    if has_prec.any():
        pv = precision[has_prec]
        print(f"  Rules with support>0 : {int(has_prec.sum())}/{R}  "
              f"precision range=[{pv.min():.4f}, {pv.max():.4f}] mean={pv.mean():.4f}")
    else:
        print("  WARNING: no rule fired from any known subject. Check data_path and rule_file.")

    out_pt = os.path.join(args.out_dir, "rule_precision.pt")
    torch.save({
        "precision":  torch.tensor(precision,    dtype=torch.float32),
        "support":    torch.tensor(support,      dtype=torch.long),
        "gold_fired": torch.tensor(gold_fired,   dtype=torch.long),
        "head":       torch.tensor(heads_meta,   dtype=torch.long),
        "length":     torch.tensor(lengths_meta, dtype=torch.long),
    }, out_pt)
    print(f"\nSaved {out_pt}\n\nDone.")


if __name__ == "__main__":
    main()
