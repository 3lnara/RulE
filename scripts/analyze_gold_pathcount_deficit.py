#!/usr/bin/env python3
"""
Weight-free path-count head-to-head: on the rules GOLD actually fires, how often
does a competitor reach gold via MORE paths (higher count) than gold itself?

This forgets scores/weights entirely (no w_R, no binary-vs-raw) and looks only
at raw path multiplicity.  For every rule R that gold activates we compare
gold's path count on R against the path counts other candidate entities receive
on the SAME rule R (like-for-like: a competitor that does not fire R at all is
not a comparison, it is just absence of that evidence).  This is the pure-count
version of the qi=857 story: gold fires r9692 once, competitor id=36 fires it
21x -> gold is "out-multiplied" on that rule.

Definitions (per query q, gold g, filtered competitor set C = entities NOT in
hr2ooo(h,r); gold and other true tails excluded, mirroring trainer.evaluate):
  count_R[e]   = # paths of rule R reaching entity e   (>= 1 when R fires to e)
  For a rule R with count_R[g] > 0:
    co-firers  = { c in C : count_R[c] > 0 }          (competitors also firing R)
    R is "gold-exclusive" if co-firers is empty (no competitor fires it).
    Otherwise R is "contested" and classified by count_R[g] vs the co-firers:
      OUT-MULTIPLIED   count_R[g] <  max_c count_R[c]   (a competitor out-fires gold)
      TIED-TOP         count_R[g] == max_c count_R[c]   (gold ties the best, none above)
      DOMINATES        count_R[g] >  max_c count_R[c]   (gold has the most paths)

Also reported: the pairwise view (each (gold-rule, co-firing competitor) cell),
and an "overall quantity" view (total paths summed over each entity's rules).

Outputs
-------
  Prints the corpus tallies (per-rule, pairwise, per-query, overall-quantity).
  Writes analysis/gold_pathcount_deficit_<split>.csv  (one row per grounded gold).

Usage (from repo root):
    python scripts/analyze_gold_pathcount_deficit.py --split valid
    python scripts/analyze_gold_pathcount_deficit.py --split test
"""

import argparse
import csv
import os
from collections import defaultdict

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir",
                   default="outputs/additive_umls_aggregation_2x2/analysis")
    p.add_argument("--data_path", default="data/umls")
    p.add_argument("--split", default="valid")
    p.add_argument("--top", type=int, default=10,
                   help="How many worst out-multiplied queries to print.")
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


def torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def load_relations(data_path):
    id2rel = {}
    with open(os.path.join(data_path, "relations.dict")) as f:
        for line in f:
            i, name = line.strip().split("\t")
            id2rel[int(i)] = name
    return id2rel, len(id2rel)


def build_hr2ooo(data_path, N, num_rel):
    ent2id, rel2id = {}, {}
    with open(os.path.join(data_path, "entities.dict")) as f:
        for line in f:
            i, name = line.strip().split("\t")
            ent2id[name] = int(i)
    with open(os.path.join(data_path, "relations.dict")) as f:
        for line in f:
            i, name = line.strip().split("\t")
            rel2id[name] = int(i)
    hr2ooo = defaultdict(set)
    for fname in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(data_path, fname)) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h = ent2id[parts[0]]; r = rel2id[parts[1]]; t = ent2id[parts[2]]
                hr2ooo[r * N + h].add(t)
                hr2ooo[(r + num_rel) * N + t].add(h)
    return hr2ooo


def rel_name(r, id2rel, num_rel):
    return id2rel[r] if r < num_rel else id2rel[r - num_rel] + "^-1"


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main():
    args = parse_args()
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    os.chdir(_root)

    id2rel, num_rel = load_relations(args.data_path)
    N = 0
    with open(os.path.join(args.data_path, "entities.dict")) as f:
        N = sum(1 for _ in f)
    hr2ooo = build_hr2ooo(args.data_path, N, num_rel)

    counts = torch_load(os.path.join(args.analysis_dir,
                                     f"counts_{args.split}.pt"))
    query_h    = counts["query_h"]
    query_r    = counts["query_r"]
    query_gold = counts["query_gold"]
    q_ptr      = counts["q_ptr"]
    rule_id    = counts["rule_id"]
    ent_arr    = counts["ent"]
    cnt_arr    = counts["cnt"].long()
    Q = query_h.numel()

    # per-rule (gold-activated rule instance) tallies
    T = E = C = O = Ti = D = 0
    # pairwise (gold-rule, co-firing competitor) tallies
    P = p_lower = p_equal = p_higher = 0
    # per-query / overall-quantity tallies
    q_grounded = 0
    q_out_any = q_out_majority = 0
    q_dom_any = q_dom_majority = 0                 # vice versa: gold outnumbers
    q_tot_lower = q_tot_tied = q_tot_higher = 0   # gold total paths vs best comp

    rows = []
    for qi in range(Q):
        h    = int(query_h[qi]); r = int(query_r[qi]); gold = int(query_gold[qi])
        s    = int(q_ptr[qi]);   e = int(q_ptr[qi + 1])
        if s >= e:
            continue

        ecounts  = defaultdict(dict)      # ent -> {rule: count}
        rule2ent = defaultdict(list)      # rule -> [(ent, count)]
        tot_paths = defaultdict(int)      # ent -> total paths over all its rules
        for rid, en, c in zip(rule_id[s:e].tolist(),
                              ent_arr[s:e].tolist(),
                              cnt_arr[s:e].tolist()):
            ecounts[en][rid] = c
            rule2ent[rid].append((en, c))
            tot_paths[en] += c

        known = hr2ooo.get(r * N + h, set())    # includes gold + other trues
        if gold not in ecounts:                 # gold ungrounded -> no rule to compare
            continue
        q_grounded += 1

        n_contested = n_out = n_dom = 0
        worst_rule = worst_gap = worst_comp = None
        best_rule = best_gap = best_comp = None
        for R, g_cnt in ecounts[gold].items():
            cofire = [(c, cc) for c, cc in rule2ent[R]
                      if c != gold and c not in known]
            T += 1
            if not cofire:
                E += 1
                continue
            C += 1
            n_contested += 1
            counts_c = [cc for _, cc in cofire]
            max_c = max(counts_c)
            # pairwise
            for c, cc in cofire:
                P += 1
                if g_cnt < cc:   p_lower  += 1
                elif g_cnt == cc: p_equal += 1
                else:             p_higher += 1
            # per-rule verdict
            if g_cnt < max_c:
                O += 1; n_out += 1
                gap = max_c - g_cnt
                if worst_gap is None or gap > worst_gap:
                    worst_gap = gap; worst_rule = R
                    worst_comp = max(cofire, key=lambda t: t[1])[0]
            elif g_cnt == max_c:
                Ti += 1
            else:                                  # gold strictly dominates
                D += 1; n_dom += 1
                gap = g_cnt - max_c
                if best_gap is None or gap > best_gap:
                    best_gap = gap; best_rule = R
                    best_comp = max(cofire, key=lambda t: t[1])[0]

        out_any = n_out >= 1
        out_majority = n_contested > 0 and n_out > n_contested / 2.0
        q_out_any += int(out_any)
        q_out_majority += int(out_majority)
        q_dom_any += int(n_dom >= 1)
        q_dom_majority += int(n_contested > 0 and n_dom > n_contested / 2.0)

        # overall-quantity view: gold's total paths vs best competitor's total
        comp_tot = [tot_paths[c] for c in tot_paths if c != gold and c not in known]
        g_tot = tot_paths[gold]
        best_comp_tot = max(comp_tot) if comp_tot else 0
        if g_tot < best_comp_tot:   q_tot_lower  += 1
        elif g_tot == best_comp_tot: q_tot_tied  += 1
        else:                        q_tot_higher += 1

        rows.append({
            "qi": qi, "h": h, "r": r,
            "rel_name": rel_name(r, id2rel, num_rel), "gold": gold,
            "n_gold_rules": len(ecounts[gold]),
            "n_contested": n_contested,
            "n_out_multiplied": n_out,
            "frac_out_multiplied": round(n_out / n_contested, 3) if n_contested else "",
            "gold_total_paths": g_tot,
            "best_comp_total_paths": best_comp_tot,
            "gold_total_lower": int(g_tot < best_comp_tot),
            "worst_rule": worst_rule if worst_rule is not None else "",
            "worst_rule_gold_count": ecounts[gold].get(worst_rule, "") if worst_rule is not None else "",
            "worst_rule_comp_count": (worst_gap + ecounts[gold][worst_rule]) if worst_rule is not None else "",
            "worst_rule_comp_id": worst_comp if worst_comp is not None else "",
            "n_dominate": n_dom,
            "frac_dominate": round(n_dom / n_contested, 3) if n_contested else "",
            "best_rule": best_rule if best_rule is not None else "",
            "best_rule_gold_count": ecounts[gold].get(best_rule, "") if best_rule is not None else "",
            "best_rule_comp_count": (ecounts[gold][best_rule] - best_gap) if best_rule is not None else "",
            "best_rule_comp_id": best_comp if best_comp is not None else "",
        })

    # ---- report ----
    print("=" * 76)
    print(f"{args.data_path}  split={args.split}   (weight-free path-count view)")
    print(f"On gold's activated rules: does a competitor reach gold via MORE paths?")
    print("=" * 76)
    print(f"Grounded-gold queries: {q_grounded}")

    print(f"\n[A] Per gold-activated rule instance (query x gold-rule):  total={T}")
    print(f"  gold-exclusive (no competitor fires that rule):  {E:6d}  ({pct(E,T):5.1f}%)")
    print(f"  contested      (>=1 competitor co-fires):        {C:6d}  ({pct(C,T):5.1f}%)")
    print(f"    among contested rules:")
    print(f"      gold OUT-MULTIPLIED (a competitor has MORE paths): "
          f"{O:6d}  ({pct(O,C):5.1f}% of contested)")
    print(f"      gold tied for top:                                "
          f"{Ti:6d}  ({pct(Ti,C):5.1f}% of contested)")
    print(f"      gold DOMINATES (most paths among co-firers):      "
          f"{D:6d}  ({pct(D,C):5.1f}% of contested)")

    print(f"\n[B] Pairwise (gold-rule x co-firing competitor):  total pairs={P}")
    print(f"  gold has LOWER count than the competitor: {p_lower:7d}  ({pct(p_lower,P):5.1f}%)")
    print(f"  gold equal:                               {p_equal:7d}  ({pct(p_equal,P):5.1f}%)")
    print(f"  gold has HIGHER count:                    {p_higher:7d}  ({pct(p_higher,P):5.1f}%)")

    print(f"\n[C] Per-query (grounded gold):")
    print(f"  gold out-multiplied on >=1 of its contested rules: "
          f"{q_out_any:5d}  ({pct(q_out_any,q_grounded):5.1f}%)")
    print(f"  gold out-multiplied on the MAJORITY of its rules:  "
          f"{q_out_majority:5d}  ({pct(q_out_majority,q_grounded):5.1f}%)")
    print(f"  --- vice versa (gold outnumbers competitors) ---")
    print(f"  gold DOMINATES on >=1 of its contested rules:      "
          f"{q_dom_any:5d}  ({pct(q_dom_any,q_grounded):5.1f}%)")
    print(f"  gold DOMINATES on the MAJORITY of its rules:       "
          f"{q_dom_majority:5d}  ({pct(q_dom_majority,q_grounded):5.1f}%)")

    print(f"\n[D] Overall quantity (total paths summed over each entity's rules):")
    print(f"  gold's total paths LOWER than best competitor's:   "
          f"{q_tot_lower:5d}  ({pct(q_tot_lower,q_grounded):5.1f}%)")
    print(f"  tied:                                              "
          f"{q_tot_tied:5d}  ({pct(q_tot_tied,q_grounded):5.1f}%)")
    print(f"  gold's total paths HIGHER (most paths overall):    "
          f"{q_tot_higher:5d}  ({pct(q_tot_higher,q_grounded):5.1f}%)")

    # worst out-multiplied queries
    ranked = sorted([x for x in rows if x["n_out_multiplied"] > 0],
                    key=lambda x: (x["frac_out_multiplied"], x["n_out_multiplied"]),
                    reverse=True)
    print(f"\nMost out-multiplied queries (highest fraction of rules where a "
          f"competitor out-fires gold):")
    print(f"  {'qi':>5} {'rel':<14} {'gold':>4} {'#rules':>6} {'#cont':>5} "
          f"{'#out':>4} {'frac':>5} | worst: {'rule':>7} gold/comp {'comp_id':>7}")
    for x in ranked[:args.top]:
        print(f"  {x['qi']:>5} {x['rel_name'][:14]:<14} {x['gold']:>4} "
              f"{x['n_gold_rules']:>6} {x['n_contested']:>5} "
              f"{x['n_out_multiplied']:>4} {x['frac_out_multiplied']:>5.2f} | "
              f"{('r'+str(x['worst_rule'])):>13} "
              f"{x['worst_rule_gold_count']}/{x['worst_rule_comp_count']:<4} "
              f"{str(x['worst_rule_comp_id']):>7}")

    # vice versa: most dominant queries
    ranked_dom = sorted([x for x in rows if x["n_dominate"] > 0],
                        key=lambda x: (x["frac_dominate"], x["n_dominate"]),
                        reverse=True)
    print(f"\nMost DOMINANT queries (highest fraction of rules where gold "
          f"out-fires every competitor):")
    print(f"  {'qi':>5} {'rel':<14} {'gold':>4} {'#rules':>6} {'#cont':>5} "
          f"{'#dom':>4} {'frac':>5} | best:  {'rule':>7} gold/comp {'comp_id':>7}")
    for x in ranked_dom[:args.top]:
        print(f"  {x['qi']:>5} {x['rel_name'][:14]:<14} {x['gold']:>4} "
              f"{x['n_gold_rules']:>6} {x['n_contested']:>5} "
              f"{x['n_dominate']:>4} {x['frac_dominate']:>5.2f} | "
              f"{('r'+str(x['best_rule'])):>13} "
              f"{x['best_rule_gold_count']}/{x['best_rule_comp_count']:<4} "
              f"{str(x['best_rule_comp_id']):>7}")

    # ---- CSV ----
    out_csv = args.out_csv or os.path.join(
        args.analysis_dir, f"gold_pathcount_deficit_{args.split}.csv")
    rows.sort(key=lambda x: (x["frac_out_multiplied"] if x["frac_out_multiplied"] != "" else -1,
                             x["n_out_multiplied"]), reverse=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}  ({len(rows)} grounded-gold queries)")


if __name__ == "__main__":
    main()
