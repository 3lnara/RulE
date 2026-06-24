#!/usr/bin/env python3
"""
Per-query, per-rule drill-down for binary-vs-raw winning queries.

For sampled queries where raw count won (or where binary won), shows the
activated rules and exactly how much score mass each rule contributed under
raw vs binary aggregation — for the gold entity and the decisive competitor.

All inputs come from the pre-dumped counts + winloss CSV; no re-grounding.

Score definitions (clamp weights: w = max(0, w_R_unclamped)):
    count_R[e]        path multiplicity of rule R reaching entity e
    raw_contrib    = w * count_R[e]
    binary_contrib = w * 1[count_R[e] > 0]
    score_raw[e]    = sum_R raw_contrib
    score_binary[e] = sum_R binary_contrib
    extra_raw_mass  = score_raw[e] - score_binary[e]
                    = sum_R w * (count_R[e] - 1)  over activated rules

Outputs:
  Prints a breakdown per sampled query.
  Writes analysis/inspect_<split>.csv   (one row per (query, entity_role, rule)).
  With --plot: saves per-query figures under --fig_dir (default analysis/figs/):
    inspect_q{qi}_{outcome}_fanout.png   Plot 1 - selectivity view
    inspect_q{qi}_{outcome}_counts.png   Plot 2 - raw vs binary tradeoff
    inspect_q{qi}_{outcome}_scatter.png  Plot 3 - all-competitor outcome view

Usage (run from repo root):
    python scripts/inspect_winning_queries.py --split valid --outcome both --n 3
    python scripts/inspect_winning_queries.py --split valid --outcome raw_wins \\
        --n 5 --sort delta --top_rules 15 --plot
"""

import argparse
import csv
import os
import random
from collections import defaultdict

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir",
                   default="outputs/additive_umls_aggregation_2x2/analysis")
    p.add_argument("--data_path", default="data/umls")
    p.add_argument("--split", default="valid")
    p.add_argument("--outcome", default="both",
                   choices=["raw_wins", "binary_wins", "both"])
    p.add_argument("--n", type=int, default=3,
                   help="Number of sample queries per outcome.")
    p.add_argument("--sort", default="delta", choices=["delta", "random"],
                   help="'delta' = most decisive; 'random' = random sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top_rules", type=int, default=12,
                   help="Max per-rule rows to PRINT per entity (CSV gets all).")
    p.add_argument("--csv_max_comp", type=int, default=0,
                   help="Max competitors (ranked by raw score) to write to CSV "
                        "per query. 0 = ALL competitors.")
    p.add_argument("--plot2_ncomp", type=int, default=3,
                   help="Number of top competitors (by raw score) to draw in "
                        "Plot 2 alongside the gold.")
    p.add_argument("--out_csv", default=None,
                   help="Defaults to analysis/inspect_<split>.csv")
    p.add_argument("--plot", action="store_true",
                   help="Save per-query Plot 1 (fan-out), Plot 2 (counts), "
                        "Plot 3 (scatter).")
    p.add_argument("--fig_dir", default=None,
                   help="Directory for figures. Defaults to <analysis_dir>/figs.")
    return p.parse_args()


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


def save_plot1_fanout(qi, outcome, rel, per_rule_ents, gold, fig_dir, cap=30):
    """Plot 1: per-rule fan-out (selectivity), gold-grounding rules highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # per_rule_ents: rule_id -> list of (ent, cnt)
    rule_data = []
    for rid, lst in per_rule_ents.items():
        fanout = len(lst)
        grounds_gold = any(en == gold for en, _ in lst)
        rule_data.append((fanout, rid, grounds_gold))
    rule_data.sort(reverse=True)
    rule_data = rule_data[:cap]

    fanouts       = [d[0] for d in rule_data]
    rule_ids      = [d[1] for d in rule_data]
    grounds_golds = [d[2] for d in rule_data]
    x_pos         = list(range(len(fanouts)))
    colors        = ["#E8834C" if g else "#4C9BE8" for g in grounds_golds]

    fig, ax = plt.subplots(figsize=(max(6, len(fanouts) * 0.45 + 1), 4))
    ax.bar(x_pos, fanouts, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Rule id (sorted by fan-out, descending)")
    ax.set_ylabel("Fan-out (# entities grounded)")
    ax.set_title(f"qi={qi} [{outcome}]  rel={rel}\n"
                 f"Selectivity: per-rule fan-out  "
                 f"(orange = also grounds gold)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"r{r}" for r in rule_ids], rotation=90, fontsize=7)
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E8834C", label="grounds gold"),
        Patch(facecolor="#4C9BE8", label="does not ground gold"),
    ], fontsize=8)
    fig.tight_layout()
    path = os.path.join(fig_dir, f"inspect_q{qi}_{outcome}_fanout.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_plot2_counts(qi, outcome, rel, per_rule_ents, gold, comps, escores,
                      w_clamp, bodies, id2rel, num_rel, fig_dir, cap=12):
    """Plot 2: per-rule path counts for gold vs the top-K competitors.

    Bars above the dashed y=1 line are the 'extra raw mass' that raw activation
    adds on top of binary. Showing several competitors (not just the single top
    one) exposes the count-1 crowd that a one-competitor view would hide.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    entities = [gold] + [c for c in comps if c is not None and c != gold]
    if not entities:
        return

    # rule_id -> {ent: count}, keeping only rules that ground a shown entity
    rule_counts = {}
    for rid, lst in per_rule_ents.items():
        d = dict(lst)
        if any(en in d for en in entities):
            rule_counts[rid] = {en: d.get(en, 0) for en in entities}
    if not rule_counts:
        return

    sorted_rules = sorted(rule_counts.keys(),
                          key=lambda r: max(rule_counts[r].values()),
                          reverse=True)[:cap]

    n = len(sorted_rules)
    m = len(entities)
    x_pos = np.arange(n)
    bar_w = 0.8 / m

    # gold = blue; competitors = distinct warm hues (comp1 = top raw)
    comp_palette = ["#E8834C", "#C0392B", "#8E44AD", "#16A085",
                    "#7F8C8D", "#2C3E50"]
    colors = ["#4C9BE8"] + [comp_palette[i % len(comp_palette)]
                            for i in range(m - 1)]

    fig, ax = plt.subplots(figsize=(max(7, n * 0.95 + 1.5), 4.8))
    for i, en in enumerate(entities):
        offset  = (i - (m - 1) / 2.0) * bar_w
        counts  = [rule_counts[r][en] for r in sorted_rules]
        raw_s   = escores.get(en, (0.0, 0.0))[0]
        lab     = (f"gold id={en} (raw={raw_s:.1f})" if i == 0
                   else f"comp{i} id={en} (raw={raw_s:.1f})")
        ax.bar(x_pos + offset, counts, width=bar_w, label=lab,
               color=colors[i], alpha=0.85)

    ax.axhline(1, color="black", linestyle="--", linewidth=1.0,
               label="binary threshold (count → 1)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"r{r}" for r in sorted_rules],
                       rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("Path count")
    ax.set_xlabel("Rule")
    ax.set_title(f"qi={qi} [{outcome}]  rel={rel}\n"
                 f"Raw vs binary tradeoff: gold vs top-{m - 1} competitors "
                 f"per-rule counts\n"
                 f"(bars above the dashed line are 'extra raw mass')")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(fig_dir, f"inspect_q{qi}_{outcome}_counts.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_plot3_scatter(qi, outcome, rel, escores, gold, known_true,
                       gold_raw, gold_bin, n_beat_bin, n_beat_raw, fig_dir):
    """Plot 3: every grounded competitor as a point (x=binary score, y=raw score).

    Reading guide:
      * The dotted diagonal is raw == binary: an entity lands there iff ALL its
        fired rules had count 1 (the "count-1 crowd"). Multiplicity pushes a
        point straight UP, off the diagonal -> that is the extra mass raw adds.
      * Orange dashed lines mark the gold's binary score (vertical) and raw score
        (horizontal). Points to the RIGHT of the vertical line outrank gold under
        binary; points ABOVE the horizontal line outrank gold under raw.
      * raw_wins  => fewer points above the horizontal line than right of the
        vertical line (raw lifts gold over the crowd).
        binary_wins => a hub flies far above the diagonal and past the horizontal
        line (raw inflates a competitor over gold).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cb, cr = [], []
    for en, (raw, bn) in escores.items():
        if en in known_true or en == gold:
            continue
        if raw <= 0 or bn <= 0:          # cannot place on log axes; uninteresting
            continue
        cb.append(bn)
        cr.append(raw)
    if not cb:
        return
    cb = np.asarray(cb)
    cr = np.asarray(cr)

    # shared log-log limits (equal axes so the y=x diagonal is the true 45 deg
    # line and "count-1" entities sit exactly on it)
    vals = [cb.min(), cr.min()]
    if gold_raw > 0:
        vals.append(gold_raw)
    if gold_bin > 0:
        vals.append(gold_bin)
    lo = max(min(vals), 1e-2) / 1.3
    hi = float(max(cb.max(), cr.max(), gold_raw, gold_bin, 1.0)) * 1.3

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    # diagonal: raw == binary (all counts == 1)
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", linewidth=1.0,
            zorder=1, label="raw = binary (all counts 1)")
    # gold reference lines -> quadrants
    if gold_bin > 0:
        ax.axvline(gold_bin, color="#E8834C", linestyle="--", linewidth=0.9, zorder=2)
    if gold_raw > 0:
        ax.axhline(gold_raw, color="#E8834C", linestyle="--", linewidth=0.9, zorder=2)
    # competitors: group identical (binary, raw) scores so the count-1 crowd
    # shows as one big labelled bubble instead of a single overplotted dot
    from collections import Counter
    groups = Counter((round(float(b), 4), round(float(r), 4))
                     for b, r in zip(cb, cr))
    gx = np.array([k[0] for k in groups])
    gy = np.array([k[1] for k in groups])
    gn = np.array([groups[k] for k in groups], dtype=float)
    sizes = 22.0 + 30.0 * np.sqrt(gn - 1.0)   # singleton small, crowd large
    ax.scatter(gx, gy, s=sizes, alpha=0.5, color="#4C9BE8", zorder=3,
               edgecolor="white", linewidth=0.4,
               label=f"competitors (n={len(cb)}; bubble = #entities)")
    for (x, y), n in groups.items():
        if n >= 3:
            ax.annotate(str(n), (x, y), textcoords="offset points",
                        xytext=(7, -3), fontsize=8, color="#2b6ca3", zorder=4)
    # gold
    if gold_raw > 0 and gold_bin > 0:
        ax.scatter([gold_bin], [gold_raw], s=190, marker="*", color="#E8834C",
                   edgecolor="black", linewidth=0.6, zorder=5, label="gold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("binary score   (sum of w over fired rules)")
    ax.set_ylabel("raw score   (sum of w * count)")
    ax.set_title(f"qi={qi} [{outcome}]  rel={rel}\n"
                 f"All competitors: binary vs raw score (gold = star)\n"
                 f"beat gold:  binary={n_beat_bin}  ->  raw={n_beat_raw}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(fig_dir, f"inspect_q{qi}_{outcome}_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    args = parse_args()
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    os.chdir(_root)

    fig_dir = args.fig_dir or os.path.join(args.analysis_dir, "figs")
    if args.plot:
        os.makedirs(fig_dir, exist_ok=True)

    id2rel, num_rel = load_relations(args.data_path)

    def body_names(body):
        return " AND ".join(
            (id2rel[b] if b < num_rel else id2rel[b - num_rel] + "^-1")
            for b in body)

    # ---- load dump + meta ----
    meta = torch.load(os.path.join(args.analysis_dir, "rule_meta.pt"),
                      weights_only=False)
    heads   = meta["head"].tolist()
    bodies  = meta["bodies"]
    lengths = meta["length"].tolist()
    w_keep  = meta["w_R_unclamped"]
    w_clamp = w_keep.clamp(min=0.0)

    rules_per_rel = defaultdict(int)
    for h in heads:
        rules_per_rel[h] += 1

    counts = torch.load(os.path.join(args.analysis_dir, f"counts_{args.split}.pt"),
                        weights_only=False)
    query_h    = counts["query_h"]
    query_r    = counts["query_r"]
    query_gold = counts["query_gold"]
    q_ptr      = counts["q_ptr"]
    rule_id    = counts["rule_id"]
    ent_arr    = counts["ent"]
    cnt_arr    = counts["cnt"].long()

    N = 0
    with open(os.path.join(args.data_path, "entities.dict")) as f:
        N = sum(1 for _ in f)
    hr2ooo = build_hr2ooo(args.data_path, N, num_rel)

    # ---- load winloss rows ----
    wl_path = os.path.join(args.analysis_dir, f"winloss_{args.split}.csv")
    wl_rows = []
    with open(wl_path) as f:
        for row in csv.DictReader(f):
            wl_rows.append(row)

    def pick(outcome):
        cand = [r for r in wl_rows if r["outcome"] == outcome]
        if args.sort == "delta":
            cand.sort(key=lambda r: abs(float(r["rank_delta"])), reverse=True)
        else:
            random.Random(args.seed).shuffle(cand)
        return cand[:args.n]

    selected = []
    if args.outcome in ("raw_wins", "both"):
        selected += pick("raw_wins")
    if args.outcome in ("binary_wins", "both"):
        selected += pick("binary_wins")

    # ---- helpers ----
    def query_rule_ents(qi):
        """Return dict rule_id -> list of (ent, cnt) for this query."""
        s = q_ptr[qi].item(); e = q_ptr[qi + 1].item()
        per_rule = defaultdict(list)
        if s < e:
            for rid, en, c in zip(rule_id[s:e].tolist(),
                                   ent_arr[s:e].tolist(),
                                   cnt_arr[s:e].tolist()):
                per_rule[rid].append((en, c))
        return per_rule

    def query_entity_rules(qi):
        """Return dict ent -> list of (rule_id, count) for this query."""
        s = q_ptr[qi].item(); e = q_ptr[qi + 1].item()
        per_ent = defaultdict(list)
        if s < e:
            for rid, en, c in zip(rule_id[s:e].tolist(),
                                   ent_arr[s:e].tolist(),
                                   cnt_arr[s:e].tolist()):
                per_ent[en].append((rid, c))
        return per_ent

    def entity_scores(per_ent):
        """ent -> (raw_score, binary_score) under clamp weights."""
        out = {}
        for en, lst in per_ent.items():
            raw = sum(w_clamp[rid].item() * c     for rid, c in lst)
            bn  = sum(w_clamp[rid].item()          for rid, c in lst)
            out[en] = (raw, bn)
        return out

    def entity_breakdown_rows(qi, en, role, per_ent, escores):
        """Per-rule contribution rows for one entity, sorted by raw_contrib."""
        raw_s, bin_s = escores.get(en, (0.0, 0.0))
        rows = []
        for rid, c in per_ent.get(en, []):
            wk = w_keep[rid].item(); wc = w_clamp[rid].item()
            rows.append({
                "qi": qi, "role": role, "entity_id": en,
                "entity_raw_score": round(raw_s, 4),
                "entity_bin_score": round(bin_s, 4),
                "rule_id": rid,
                "length": lengths[rid], "count": c,
                "w_keep": round(wk, 4), "w_clamp": round(wc, 4),
                "raw_contrib":    round(wc * c, 4),
                "binary_contrib": round(wc,     4),
                "body": body_names(bodies[rid]),
            })
        rows.sort(key=lambda d: d["raw_contrib"], reverse=True)
        return rows

    # ---- iterate over selected queries ----
    csv_rows = []
    for row in selected:
        qi      = int(row["qi"])
        h       = int(row["h"])
        r       = int(row["r"])
        gold    = int(row["gold"])
        rel     = row["rel_name"]
        outcome = row["outcome"]

        per_ent      = query_entity_rules(qi)
        per_rule_ents = query_rule_ents(qi)
        escores      = entity_scores(per_ent)

        known_true  = hr2ooo.get(r * N + h, set())
        competitors = [en for en in escores if en not in known_true]

        gold_raw, gold_bin = escores.get(gold, (0.0, 0.0))

        top_comp_raw = max(competitors, key=lambda e: escores[e][0], default=None)
        top_comp_bin = max(competitors, key=lambda e: escores[e][1], default=None)

        n_beat_raw = sum(1 for e in competitors if escores[e][0] > gold_raw)
        n_beat_bin = sum(1 for e in competitors if escores[e][1] > gold_bin)

        n_activated = len(per_rule_ents)
        n_rules_rel = rules_per_rel.get(r, 0)

        print("=" * 78)
        print(f"qi={qi}  relation={rel} (r={r})  head={h}  gold={gold}  "
              f"OUTCOME={outcome}")
        print(f"  rank  binary_clamp={float(row['rank_binary_clamp']):.1f}  "
              f"raw_clamp={float(row['rank_raw_clamp']):.1f}  "
              f"delta={float(row['rank_delta']):+.1f}   "
              f"(delta>0 => raw better)")
        print(f"  activated rules (fired anywhere) = {n_activated} / "
              f"{n_rules_rel} rules for this relation")
        print(f"  GOLD   id={gold}:  raw_score={gold_raw:7.3f}  "
              f"binary_score={gold_bin:7.3f}  extra_raw_mass={gold_raw-gold_bin:7.3f}  "
              f"(competitors beating gold: binary={n_beat_bin}, raw={n_beat_raw})")
        if top_comp_raw is not None:
            cr_raw, cr_bin = escores[top_comp_raw]
            print(f"  TOPCOMP(raw)  id={top_comp_raw}:  raw_score={cr_raw:7.3f}  "
                  f"binary_score={cr_bin:7.3f}")
        if top_comp_bin is not None and top_comp_bin != top_comp_raw:
            cb_raw, cb_bin = escores[top_comp_bin]
            print(f"  TOPCOMP(bin)  id={top_comp_bin}:  raw_score={cb_raw:7.3f}  "
                  f"binary_score={cb_bin:7.3f}")

        # Competitors ranked by raw score (comp1 = top raw competitor)
        comps_by_raw = sorted(competitors, key=lambda e: escores[e][0],
                              reverse=True)

        # Console: concise per-rule tables for gold + top raw competitor only
        for role, en in [("gold", gold), ("top_comp_raw", top_comp_raw)]:
            if en is None:
                continue
            brows = entity_breakdown_rows(qi, en, role, per_ent, escores)
            print(f"\n  --- {role} (entity {en}): {len(brows)} activated rules "
                  f"(showing top {min(args.top_rules, len(brows))} by raw_contrib) ---")
            print(f"    {'rule':>6} {'cnt':>4} {'w_clmp':>7} {'raw':>8} "
                  f"{'bin':>7}  body")
            for d in brows[:args.top_rules]:
                print(f"    {d['rule_id']:>6} {d['count']:>4} {d['w_clamp']:>7.2f} "
                      f"{d['raw_contrib']:>8.2f} {d['binary_contrib']:>7.2f}  "
                      f"{d['body']}")
        print()

        # CSV: gold + ALL competitors (or top --csv_max_comp), ranked by raw score
        csv_rows.extend(entity_breakdown_rows(qi, gold, "gold", per_ent, escores))
        n_comp_csv = (len(comps_by_raw) if args.csv_max_comp <= 0
                      else min(args.csv_max_comp, len(comps_by_raw)))
        for rank, en in enumerate(comps_by_raw[:n_comp_csv], start=1):
            csv_rows.extend(
                entity_breakdown_rows(qi, en, f"comp{rank}", per_ent, escores))

        if args.plot:
            save_plot1_fanout(qi, outcome, rel, per_rule_ents, gold, fig_dir)
            save_plot2_counts(qi, outcome, rel, per_rule_ents, gold,
                              comps_by_raw[:args.plot2_ncomp], escores,
                              w_clamp, bodies, id2rel, num_rel, fig_dir)
            save_plot3_scatter(qi, outcome, rel, escores, gold, known_true,
                               gold_raw, gold_bin, n_beat_bin, n_beat_raw, fig_dir)

    # ---- write CSV ----
    out_csv = args.out_csv or os.path.join(args.analysis_dir,
                                           f"inspect_{args.split}.csv")
    if csv_rows:
        fieldnames = ["qi", "role", "entity_id", "entity_raw_score",
                      "entity_bin_score", "rule_id", "length", "count",
                      "w_keep", "w_clamp", "raw_contrib", "binary_contrib", "body"]
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Wrote {out_csv}  ({len(csv_rows)} per-rule rows across "
              f"{len(selected)} queries)")
    else:
        print("No queries selected (check --outcome / --split).")


if __name__ == "__main__":
    main()
