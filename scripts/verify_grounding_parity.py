#!/usr/bin/env python3
"""Assert the offline grounding stack agrees with the real in-model KG.

The offline aggregators (rule_logreg_train.py, rule_precision_train.py,
dump_rule_counts.py) reimplement the graph as MinimalGraph with torch.scatter_add_
so they run without torch_scatter. This
imports the real data.KnowledgeGraph and checks, on sampled forward and inverse
head relations, exact integer path-count equality for graph parity, rule
alignment (rule_features[:, 0] vs load_rules order), plain grounding, and
leave-one-out grounding. All mismatches are fatal.

Usage:
    python scripts/verify_grounding_parity.py --data_path data/<ds>
"""

import argparse
import os
import sys
import random
from collections import defaultdict

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src_additive")
for _p in (_SRC_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as _data_mod                       # REAL KnowledgeGraph (torch_scatter)
import dump_rule_counts as _drc                # MinimalGraph + load_rules
import rule_logreg_train as _rlt               # offline ground() with LOO

KnowledgeGraph = _data_mod.KnowledgeGraph
RuleDataset = _data_mod.RuleDataset
MinimalGraph = _drc.MinimalGraph
load_rules = _drc.load_rules
ground = _rlt.ground
build_gold = _rlt.build_gold


class Report:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def ok(self, msg):
        self.checks += 1
        print(f"    PASS  {msg}")

    def fail(self, msg, detail=""):
        self.checks += 1
        self.failures.append(msg)
        print(f"    FAIL  {msg}")
        if detail:
            for line in str(detail).splitlines():
                print(f"          {line}")

    def check(self, cond, msg, detail=""):
        self.ok(msg) if cond else self.fail(msg, detail)
        return cond


# ---------------------------------------------------------------------------
# Check 0: graph parity
# ---------------------------------------------------------------------------

def check_graph_parity(kg, mg, rep):
    print("\n[0] Graph parity (sizes, adjacency orientation, gold dict)")

    rep.check(kg.entity_size == mg.entity_size,
              f"entity_size {kg.entity_size} == {mg.entity_size}")
    rep.check(kg.relation_size == mg.relation_size,
              f"relation_size {kg.relation_size} == {mg.relation_size}")

    NR = mg.relation_size

    # Real adjacency: index[0] = dst (t), index[1] = src (h)  [see data.propagate]
    # Minimal adjacency: (src, dst) = (h, t)
    # Compare as edge multisets, so an orientation flip cannot hide.
    bad, checked = [], 0
    for r in range(NR * 2):
        real_index = kg.relation2adjacency[r][0]        # LongTensor [2, E]
        real_pairs = sorted(zip(real_index[1].tolist(),  # src
                                real_index[0].tolist()))  # dst
        m_src, m_dst = mg.relation2adjacency[r]
        min_pairs = sorted(zip(m_src.tolist(), m_dst.tolist()))
        checked += len(real_pairs)
        if real_pairs != min_pairs:
            bad.append(r)
    rep.check(not bad,
              f"directed edge multisets match on all {NR * 2} relations "
              f"({checked} edges)",
              f"mismatched relations: {bad[:10]}")

    # A flipped orientation would still pass if every relation were symmetric.
    # Assert the test actually has teeth: some relation is NOT symmetric.
    asym = 0
    for r in range(NR * 2):
        s, d = mg.relation2adjacency[r]
        fwd = set(zip(s.tolist(), d.tolist()))
        rev = set(zip(d.tolist(), s.tolist()))
        if fwd and fwd != rev:
            asym += 1
    rep.check(asym > 0,
              f"{asym}/{NR * 2} relations are asymmetric (orientation check has teeth)",
              "every relation is symmetric; an orientation flip is undetectable")

    # gold dict (offline) must reproduce hr2o (in-model train-only tails)
    gold = build_gold(mg)
    mism = []
    for hr_index, tails in kg.hr2o.items():
        h, r = kg.decode_hr(hr_index)
        if sorted(set(tails)) != sorted(gold[r][h]):
            mism.append((h, r))
    n_gold_keys = sum(len(v) for v in gold.values())
    rep.check(not mism and n_gold_keys == len(kg.hr2o),
              f"build_gold reproduces hr2o exactly ({len(kg.hr2o)} (h, r) keys)",
              f"mismatched keys: {mism[:10]}  "
              f"offline keys={n_gold_keys} vs hr2o={len(kg.hr2o)}")
    return gold


# ---------------------------------------------------------------------------
# Check 1: rule alignment
# ---------------------------------------------------------------------------

def check_rule_alignment(rule_file, NR, rules, rep):
    print("\n[1] Rule alignment (rule_features[:, 0] vs load_rules order)")

    # RuleDataset is the exact provenance of what main.py hands to set_rules:
    #     rules = [rule[0] for rule in ruleset.rules]
    # and set_rules writes rule_features[i] = rules[i] + padding.
    ruleset = RuleDataset(NR, rule_file, negative_sample_size=1)
    model_rules = [r[0] for r in ruleset.rules]

    rep.check(len(model_rules) == len(rules),
              f"rule count matches: {len(model_rules)} == {len(rules)}",
              "the trainer's beta.size(0) != R guard would also catch this")

    n = min(len(model_rules), len(rules))
    gid_bad, body_bad = [], []
    for i in range(n):
        m_gid, m_head, m_body = model_rules[i][0], model_rules[i][1], model_rules[i][2:]
        o_gid, o_head, o_body = rules[i][0], rules[i][1], rules[i][2:]
        if m_gid != o_gid:
            gid_bad.append((i, m_gid, o_gid))
        # This is the check that matters: a shifted id has the same count and
        # the same arange() of gids, but a DIFFERENT body at that gid.
        if (m_head, list(m_body)) != (o_head, list(o_body)):
            body_bad.append((i, (m_head, m_body), (o_head, o_body)))

    rep.check(not gid_bad, f"rule ids agree at every position ({n} rules)",
              f"first mismatches (pos, model_gid, offline_gid): {gid_bad[:5]}")
    rep.check(not body_bad,
              f"(head, body) identical at every rule id ({n} rules)",
              f"first mismatches: {body_bad[:3]}")

    gids = [r[0] for r in model_rules]
    rep.check(gids == list(range(len(gids))),
              "rule_features[:, 0] == arange(R) (beta[gids] is the identity map)")
    return model_rules


# ---------------------------------------------------------------------------
# Checks 2 & 3: grounding parity, with and without leave-one-out
# ---------------------------------------------------------------------------

def pick_relations(rules, gold, NR, num_relations, rng):
    """Sample head relations that have both rules and gold facts.

    Deliberately draws from forward (r < NR) and inverse (r >= NR) relations:
    an orientation flip on the adjacency is invisible if you only ever probe
    one direction.
    """
    r2rules = defaultdict(list)
    for rule in rules:
        r2rules[rule[1]].append(rule)

    eligible = [r for r in r2rules if gold.get(r)]
    fwd = sorted(r for r in eligible if r < NR)
    inv = sorted(r for r in eligible if r >= NR)

    half = max(1, num_relations // 2)
    pick = rng.sample(fwd, min(half, len(fwd))) if fwd else []
    pick += rng.sample(inv, min(num_relations - len(pick), len(inv))) if inv else []
    if not pick and eligible:
        pick = rng.sample(eligible, min(num_relations, len(eligible)))
    return sorted(pick), r2rules


def check_grounding(kg, mg, rules, gold, rep, num_relations, max_heads,
                    max_rules, seed):
    rng = random.Random(seed)
    NR = mg.relation_size
    picks, r2rules = pick_relations(rules, gold, NR, num_relations, rng)

    print(f"\n[2] Grounding parity, full graph (no LOO)  "
          f"relations={picks}")

    n_cells = 0
    mismatch_min, mismatch_off = [], []
    nonzero_total = 0

    for r in picks:
        heads = sorted(gold[r].keys())
        if len(heads) > max_heads:
            heads = rng.sample(heads, max_heads)
        heads = sorted(heads)
        H = torch.tensor(heads, dtype=torch.long)

        rl = r2rules[r]
        if len(rl) > max_rules:
            rl = rng.sample(rl, max_rules)

        for rule in rl:
            r_head, body = rule[1], rule[2:]
            real = kg.grounding(H, r_head, body, None).long()          # [B, N]
            mini = mg.grounding(H, r_head, body, None).long()
            offl = ground(mg, H, body, torch.device("cpu")).long()

            n_cells += real.numel()
            nonzero_total += int((real > 0).sum())
            if not torch.equal(real, mini):
                mismatch_min.append((r, rule[0]))
            if not torch.equal(real, offl):
                mismatch_off.append((r, rule[0]))

    rep.check(nonzero_total > 0,
              f"sampled groundings are non-trivial ({nonzero_total} nonzero cells)",
              "all counts are zero; the comparison proves nothing")
    rep.check(not mismatch_min,
              f"MinimalGraph.grounding == KnowledgeGraph.grounding "
              f"({n_cells} count cells, exact int equality)",
              f"first mismatches (r, rule_id): {mismatch_min[:5]}")
    rep.check(not mismatch_off,
              f"offline ground() == KnowledgeGraph.grounding "
              f"({n_cells} count cells, exact int equality)",
              f"first mismatches (r, rule_id): {mismatch_off[:5]}")

    # ---- [3] leave-one-out parity -------------------------------------------
    print(f"\n[3] Leave-one-out parity (edges_to_remove vs drop_dst)")

    n_cells = 0
    mismatch_loo = []
    differs_from_plain = 0

    for r in picks:
        facts = [(h, t) for h in sorted(gold[r]) for t in sorted(gold[r][h])]
        if len(facts) > max_heads:
            facts = rng.sample(facts, max_heads)
        facts = sorted(facts)
        H = torch.tensor([h for h, _ in facts], dtype=torch.long)
        T = torch.tensor([t for _, t in facts], dtype=torch.long)

        # Exactly how TrainDataset.__getitem__ builds edges_to_remove.
        etr = torch.tensor(
            [kg.relation2ht2index[r][kg.encode_ht(h, t)] for h, t in facts],
            dtype=torch.long)

        rl = r2rules[r]
        if len(rl) > max_rules:
            rl = rng.sample(rl, max_rules)

        for rule in rl:
            r_head, body = rule[1], rule[2:]
            real = kg.grounding(H, r_head, body, etr).long()
            offl = ground(mg, H, body, torch.device("cpu"),
                          r_query=r, drop_dst=T).long()
            plain = kg.grounding(H, r_head, body, None).long()

            n_cells += real.numel()
            if not torch.equal(real, offl):
                mismatch_loo.append((r, rule[0]))
            if not torch.equal(real, plain):
                differs_from_plain += 1

    rep.check(differs_from_plain > 0,
              f"LOO actually removes paths on {differs_from_plain} "
              f"(relation, rule) pairs",
              "LOO never changed a count; drop_dst/edges_to_remove is a no-op "
              "here and the check is vacuous")
    rep.check(not mismatch_loo,
              f"offline ground(drop_dst) == KnowledgeGraph.grounding("
              f"edges_to_remove)  ({n_cells} count cells, exact int equality)",
              f"first mismatches (r, rule_id): {mismatch_loo[:5]}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", default="data/umls")
    p.add_argument("--rule_file", default=None)
    p.add_argument("--num_relations", type=int, default=6,
                   help="How many head relations to probe (split forward/inverse).")
    p.add_argument("--max_heads", type=int, default=64,
                   help="Max query heads (check 2) / gold facts (check 3) per relation.")
    p.add_argument("--max_rules", type=int, default=40,
                   help="Max rules per sampled head relation.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    rule_file = args.rule_file or os.path.join(args.data_path, "mined_rules.txt")

    torch.manual_seed(args.seed)
    print(f"data_path : {args.data_path}")
    print(f"rule_file : {rule_file}")

    print("\nLoading REAL KnowledgeGraph (src_additive/data.py, torch_scatter)...")
    kg = KnowledgeGraph(args.data_path)
    print("Loading MinimalGraph (offline reimplementation)...")
    mg = MinimalGraph(args.data_path)
    rules = load_rules(rule_file, mg.relation_size)
    print(f"  entities={mg.entity_size}  relations={mg.relation_size}  "
          f"rules={len(rules)}")

    rep = Report()
    gold = check_graph_parity(kg, mg, rep)
    check_rule_alignment(rule_file, mg.relation_size, rules, rep)
    check_grounding(kg, mg, rules, gold, rep, args.num_relations,
                    args.max_heads, args.max_rules, args.seed)

    print("\n" + "=" * 74)
    if rep.failures:
        print(f"FAILED  {len(rep.failures)}/{rep.checks} checks:")
        for f in rep.failures:
            print(f"  - {f}")
        print("=" * 74)
        return 1
    print(f"ALL {rep.checks} CHECKS PASSED -- offline grounding agrees with the KG.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
