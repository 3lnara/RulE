#!/usr/bin/env python3
"""
Verify the raw path-count for a single (query, rule, entity) triple by
explicitly enumerating every grounding path in the train graph.

This is an *independent* re-derivation of what dump_rule_counts.py's grounding
does via matrix message-passing. If the number of enumerated paths equals the
stored `cnt`, the count is confirmed to be "number of distinct rule-body paths
from the query head to that entity".

Grounding semantics being reproduced (dump_rule_counts.py):
  - one-hot at the query head h, then push mass along each body relation.
  - a body relation id r < NR is the forward edge h->t of relation r;
    a body relation id r >= NR is the INVERSE edge t->h of relation r-NR.
  - the mass that lands on entity e after the whole body = # of walks
    h --b1--> * --b2--> * ... --bk--> e  (walks, i.e. paths counted with
    multiplicity; intermediate nodes may repeat across different paths).

Usage (run from repo root):
    python scripts/verify_rule_paths.py \\
        --analysis_dir outputs/additive_umls_aggregation_2x2/analysis \\
        --data_path    data/umls \\
        --split        valid \\
        --qi 3                      # which query row
        # optional: --rule_id 9614 --target_entity 78
        # (defaults: the highest-count (rule,entity) row of that query)
"""
import argparse
import os
from collections import defaultdict

import torch


def load_dicts(data_path):
    e2id, r2id = {}, {}
    with open(os.path.join(data_path, "entities.dict")) as f:
        for line in f:
            i, n = line.strip().split("\t"); e2id[n] = int(i)
    with open(os.path.join(data_path, "relations.dict")) as f:
        for line in f:
            i, n = line.strip().split("\t"); r2id[n] = int(i)
    id2e = {v: k for k, v in e2id.items()}
    id2r = {v: k for k, v in r2id.items()}
    return e2id, r2id, id2e, id2r, len(r2id)


def load_rules(rule_file):
    """Same gid convention as dump_rule_counts.load_rules: sequential over
    lines with >= 2 tokens.  Returns dict gid -> (r_head, [body...])."""
    rules = {}
    gid = 0
    with open(rule_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ints = [int(p) for p in parts]
            rules[gid] = (ints[0], ints[1:])   # (r_head, body)
            gid += 1
    return rules


def rel_name(rid, id2r, NR):
    if rid < NR:
        return f"{id2r.get(rid, rid)}"
    return f"{id2r.get(rid - NR, rid - NR)}^-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir",
                    default="outputs/additive_umls_aggregation_2x2/analysis")
    ap.add_argument("--data_path", default="data/umls")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--rule_file", default=None,
                    help="mined rules file; default <data_path>/mined_rules.txt")
    ap.add_argument("--qi", type=int, default=3)
    ap.add_argument("--rule_id", type=int, default=None)
    ap.add_argument("--target_entity", type=int, default=None)
    ap.add_argument("--max_print", type=int, default=30,
                    help="max explicit paths to print for the target entity")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, ".."))
    os.chdir(root)

    rule_file = args.rule_file or os.path.join(args.data_path, "mined_rules.txt")

    e2id, r2id, id2e, id2r, NR = load_dicts(args.data_path)
    rules = load_rules(rule_file)

    # --- adjacency: src -> list[dst] for forward (r) and inverse (r+NR) ------
    adj = [defaultdict(list) for _ in range(2 * NR)]
    with open(os.path.join(args.data_path, "train.txt")) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            h, r, t = e2id[p[0]], r2id[p[1]], e2id[p[2]]
            adj[r][h].append(t)          # forward  h -> t
            adj[r + NR][t].append(h)     # inverse  t -> h

    # --- stored counts ------------------------------------------------------
    c = torch.load(os.path.join(args.analysis_dir, f"counts_{args.split}.pt"))
    q_ptr = c["q_ptr"]; rid = c["rule_id"]; ent = c["ent"]; cnt = c["cnt"].long()
    query_h = c["query_h"]; query_r = c["query_r"]; query_gold = c["query_gold"]

    qi = args.qi
    h = query_h[qi].item(); r_head = query_r[qi].item(); gold = query_gold[qi].item()
    s, e = q_ptr[qi].item(), q_ptr[qi + 1].item()

    print("=" * 70)
    print(f"QUERY  qi={qi}")
    print(f"  head entity : {h:>4}  ({id2e[h]})")
    print(f"  relation    : {r_head:>4}  ({rel_name(r_head, id2r, NR)})   <- query is (head, rel, ?)")
    print(f"  gold tail   : {gold:>4}  ({id2e[gold]})")
    print("=" * 70)

    rows = list(zip(rid[s:e].tolist(), ent[s:e].tolist(), cnt[s:e].tolist()))

    # pick the (rule, entity) row to verify
    if args.rule_id is None or args.target_entity is None:
        rule_sel, ent_sel, cnt_sel = max(rows, key=lambda x: x[2])
        if args.rule_id is not None:
            rule_sel = args.rule_id
        if args.target_entity is not None:
            ent_sel = args.target_entity
    else:
        rule_sel, ent_sel = args.rule_id, args.target_entity
    # stored count for the (rule_sel, ent_sel) pair
    stored = {(ri, en): ct for ri, en, ct in rows}
    cnt_sel = stored.get((rule_sel, ent_sel), 0)

    r_head_rule, body = rules[rule_sel]
    body_names = " , ".join(rel_name(b, id2r, NR) for b in body)
    print(f"\nRULE {rule_sel}:  head={r_head_rule} ({rel_name(r_head_rule, id2r, NR)})")
    print(f"  body (len {len(body)}): [{body_names}]")
    assert r_head_rule == r_head, "rule head must match query relation"

    # --- enumerate every walk from h following the body relations ----------
    # frontier: list of paths, each path = [ (node), (rel,node), (rel,node)...]
    paths = [[h]]                       # each path is a list of node ids
    for b in body:
        new_paths = []
        for pth in paths:
            u = pth[-1]
            for v in adj[b].get(u, []):
                new_paths.append(pth + [v])
        paths = new_paths

    # group by end entity
    by_end = defaultdict(list)
    for pth in paths:
        by_end[pth[-1]].append(pth)

    enum_count = len(by_end.get(ent_sel, []))

    print(f"\nTARGET entity {ent_sel} ({id2e.get(ent_sel, ent_sel)})"
          f"{'  <-- GOLD' if ent_sel == gold else ''}")
    print(f"  stored cnt (from counts_{args.split}.pt) : {cnt_sel}")
    print(f"  enumerated distinct paths                : {enum_count}")
    print(f"  MATCH: {enum_count == cnt_sel}")

    # print each explicit path
    print(f"\n  explicit paths (showing up to {args.max_print}):")
    for k, pth in enumerate(by_end.get(ent_sel, [])[:args.max_print]):
        seg = f"{id2e[pth[0]]}"
        for b, node in zip(body, pth[1:]):
            seg += f"  --{rel_name(b, id2r, NR)}-->  {id2e[node]}"
        print(f"   [{k+1:>2}] {seg}")

    # sanity: total paths == sum of grounding counts for this rule on this query
    rule_rows_stored = {en: ct for ri, en, ct in rows if ri == rule_sel}
    total_enum = sum(len(v) for v in by_end.values())
    total_store = sum(rule_rows_stored.values())
    print(f"\n  [rule-level check] sum of paths over ALL entities: "
          f"enumerated={total_enum}  stored={total_store}  "
          f"MATCH={total_enum == total_store}")


if __name__ == "__main__":
    main()
