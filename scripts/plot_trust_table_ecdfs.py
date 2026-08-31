#!/usr/bin/env python3
"""ECDF comparison of four rule_trust_table.csv diagnostics (support, minority
count / EPV, max_jaccard, VIF) across datasets: one figure, four subplots, one
ECDF curve per dataset. Axis scales are per-subplot (support symlog, minority and
VIF log, jaccard linear); VIF is right-censored at VIF_CAP so a few numerically
pathological values do not dominate the axis.

Usage:
    python scripts/plot_trust_table_ecdfs.py \\
        --trust_table <ds>:outputs/<ds>/rq/rule_trust_table.csv \\
        --out outputs/<ds>/rq/rule_trust_ecdfs.png
"""

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical palette: dataviz skill reference palette, slots 1-3 (blue/orange/
# aqua) -- the only three-slot subset validated all-pairs, both light and dark
# modes (worst-pair CVD deltaE 9.2 light / 9.4 dark; normal-vision 24.0 / 20.9).
DEFAULT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
DATASETS = [
    ("Family", "outputs/additive_family/logreg-family/rule_trust_table.csv", "#2a78d6"),
    ("UMLS", "outputs/additive_umls_aggregation_2x2/logreg-umls/rule_trust_table.csv", "#eb6834"),
    ("WN18RR", "outputs/additive_wn18rr/logreg-wn18rr-rq3/rule_trust_table.csv", "#1baf7a"),
]

GATE_COLOR = "#9a9285"
VIF_CAP = 1000.0
VIF_GATE = 10.0
EPV_GATE = 10.0
OUT_PATH = "outputs/rule_trust_ecdfs.png"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trust_table", action="append", metavar="LABEL:PATH",
                   help="A dataset's rule_trust_table.csv as 'Label:path'. Repeat "
                        "per dataset. Colours cycle through the built-in palette. "
                        "Omit entirely to use the three built-in dataset paths.")
    p.add_argument("--out", default=OUT_PATH,
                   help="Output image path (a sibling .pdf is also written). "
                        "Default: %(default)s")
    return p.parse_args()


def resolve_datasets(trust_table_args):
    if not trust_table_args:
        return DATASETS
    out = []
    for i, entry in enumerate(trust_table_args):
        if ":" not in entry:
            raise SystemExit(f"--trust_table {entry!r}: expected 'LABEL:PATH'")
        label, path = entry.split(":", 1)
        out.append((label, path, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]))
    return out


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_metrics(path):
    rows = read_rows(path)
    N = len(rows)
    support, minority, jaccard, vif = [], [], [], []
    n_vif_unknown = n_vif_inf = 0

    for r in rows:
        s = int(r["support"])
        support.append(s)

        # true_fired is the current column name; gold_fired the pre-rename one.
        gf = r.get("true_fired", r.get("gold_fired", ""))
        true_fired = int(gf) if gf not in ("", "nan") else None
        separated = (s > 0 and true_fired is not None and
                     (true_fired == 0 or true_fired == s))
        if not separated:
            ms = r["minority_support"]
            if ms not in ("", "nan"):
                minority.append(int(ms))

        mj = r["max_jaccard"]
        if mj not in ("", "nan"):
            jaccard.append(float(mj))

        vif_cell = r["vif"]
        if vif_cell == "nan":
            n_vif_unknown += 1
        elif vif_cell != "":
            v = float(vif_cell)
            if math.isinf(v):
                n_vif_inf += 1
                vif.append(VIF_CAP)
            else:
                vif.append(min(v, VIF_CAP))

    return dict(N=N, support=np.array(support, dtype=float),
                minority=np.array(minority, dtype=float),
                jaccard=np.array(jaccard, dtype=float),
                vif=np.array(vif, dtype=float),
                n_vif_unknown=n_vif_unknown, n_vif_inf=n_vif_inf)


def ecdf_xy(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c3b6")
    ax.spines["bottom"].set_color("#c9c3b6")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color="#e4e0d4", linewidth=0.8, zorder=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("cumulative fraction of rules")
    ax.tick_params(colors="#4a463f")


def gate_line(ax, x, label=None, label_side="right", style=(0, (3, 2))):
    """Vertical reference. Dashed (default) = a chosen gate threshold;
    dotted = a pile-up/censoring point that the data itself defines."""
    ax.axvline(x, color=GATE_COLOR, linestyle=style, linewidth=1.1, zorder=1)
    if label is None:
        return
    dx = 3 if label_side == "right" else -3
    ha = "left" if label_side == "right" else "right"
    ax.annotate(label, xy=(x, 0), xycoords=("data", "axes fraction"),
                xytext=(dx, 6), textcoords="offset points", annotation_clip=False,
                color=GATE_COLOR, fontsize=8.2, ha=ha, va="bottom", rotation=90)


# Gate labels sit on the curve, so each carries an opaque plate.
LABEL_BBOX = dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.92)


def pct(p, decimals=0):
    """Percentages that don't lie: a 0.2% share must not render as '0%'."""
    if 0 < p < 0.01:
        return f"{100*p:.1f}%"
    return f"{100*p:.{decimals}f}%"


def pcts(ps):
    """Format a GROUP of shares, raising precision if rounding would collapse
    two genuinely different values onto the same string (e.g. 62.2% and 61.6%
    both landing on '62%', which reads as a data error rather than a tie)."""
    out = [pct(p) for p in ps]
    if len(set(out)) < len(out):
        out = [pct(p, decimals=1) for p in ps]
    return out


def declutter(ys, min_gap):
    """Nudge a group of same-x label y-positions apart (ascending) so curves
    that cross near the annotated x don't print their labels on top of each
    other. Only the TEXT moves; the anchor dot stays at the true y and a
    connector is drawn whenever the two differ."""
    order = np.argsort(ys)
    adj = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        i, prev = order[k], order[k - 1]
        if adj[i] - adj[prev] < min_gap:
            adj[i] = adj[prev] + min_gap
    return adj


def label_group_at_x(ax, x, entries, side="right", min_gap=0.075):
    """Annotate where each dataset's ECDF meets the vertical reference at `x`.

    entries: list of (color, y, text) where y is the TRUE curve height at x.
    Every annotation in this figure is built the same way, so the four
    reference lines read identically:
      * a dot marks the exact (x, y) crossing,
      * the text sits beside it with a white halo (legible over any curve),
      * a thin connector appears only when decluttering moved the text.
    """
    # Enough clearance to clear the reference line and any near-vertical jump
    # sitting on it.
    dx = 11 if side == "right" else -11
    ha = "left" if side == "right" else "right"
    adj_ys = declutter([y for _, y, _ in entries], min_gap=min_gap)
    for (color, y, text), y_adj in zip(entries, adj_ys):
        if abs(y_adj - y) > 1e-9:
            ax.plot([x, x], [y, y_adj], color=color, linewidth=0.9, alpha=0.55,
                    zorder=4, clip_on=False)
        ax.plot([x], [y], marker="o", markersize=4.5, color=color, zorder=6,
                markeredgecolor="white", markeredgewidth=1.1, clip_on=False)
        ax.annotate(text, xy=(x, y_adj), xycoords="data", xytext=(dx, 0),
                    textcoords="offset points", annotation_clip=False,
                    color=color, fontsize=8.8, ha=ha, va="center",
                    fontweight="medium", zorder=7, bbox=LABEL_BBOX)


def main():
    global DATASETS, OUT_PATH
    args = parse_args()
    DATASETS = resolve_datasets(args.trust_table)
    OUT_PATH = args.out

    data = {name: load_metrics(path) for name, path, _ in DATASETS}

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    ax_support, ax_minority, ax_jaccard, ax_vif = axes.flat

    lines = {}
    for name, _, color in DATASETS:
        d = data[name]

        x, y = ecdf_xy(d["support"])
        (ln,) = ax_support.plot(x, y, color=color, linewidth=2, drawstyle="steps-post",
                                 zorder=3)
        lines[name] = ln

        x, y = ecdf_xy(d["minority"])
        ax_minority.plot(x, y, color=color, linewidth=2, drawstyle="steps-post", zorder=3)

        x, y = ecdf_xy(d["jaccard"])
        ax_jaccard.plot(x, y, color=color, linewidth=2, drawstyle="steps-post", zorder=3)

        x, y = ecdf_xy(d["vif"])
        ax_vif.plot(x, y, color=color, linewidth=2, drawstyle="steps-post", zorder=3)

    # --- support: symlog, zero is a real value (never-fired rules); no gate ---
    ax_support.set_xscale("symlog", linthresh=1)
    ax_support.set_xlim(0, 5e6)
    ax_support.set_title("Support (fired_total)", fontsize=12, fontweight="medium", loc="left")
    ax_support.set_xlabel("fired_total  (symlog, linear below 1)")
    style_axis(ax_support)

    # --- minority / EPV: log, always >= 1 among non-separated rules ---
    ax_minority.set_xscale("log")
    ax_minority.set_title("Minority count / EPV  (non-separated rules)", fontsize=12,
                           fontweight="medium", loc="left")
    ax_minority.set_xlabel("min(true_fired, support − true_fired)  (log)")
    gate_line(ax_minority, EPV_GATE)
    # Anchor y = the ECDF height at the gate, which here IS the labelled
    # share (rules below the gate are the ones that fail it).
    epv_fracs = [float((data[n]["minority"] < EPV_GATE).mean())
                 for n, _, _ in DATASETS]
    epv_entries = [(color, f, f"{s} < {EPV_GATE:.0f}")
                   for (_, _, color), f, s in
                   zip(DATASETS, epv_fracs, pcts(epv_fracs))]
    label_group_at_x(ax_minority, EPV_GATE, epv_entries, side="right")
    style_axis(ax_minority)

    # --- max_jaccard: linear, bounded [0, 1]; no gate, label the x=1 spike ---
    ax_jaccard.set_xscale("linear")
    # Headroom past 1.0 so the exact-duplicate reference line sits inside the
    # axes rather than merging with the right spine.
    ax_jaccard.set_xlim(0, 1.06)
    ax_jaccard.set_title("max_jaccard  (nearest-rule overlap)", fontsize=12,
                          fontweight="medium", loc="left")
    ax_jaccard.set_xlabel("max_jaccard")
    gate_line(ax_jaccard, 1.0, style=(0, (1, 2)))
    # Anchor y = curve height just below 1.0; the label reports the spike AT
    # 1.0 (exact duplicates), i.e. the jump from that height up to 1.
    jac_fracs = [float((data[n]["jaccard"] >= 1.0).mean()) for n, _, _ in DATASETS]
    jaccard_entries = [(color, 1.0 - f, f"{s} = 1.0")
                       for (_, _, color), f, s in
                       zip(DATASETS, jac_fracs, pcts(jac_fracs))]
    label_group_at_x(ax_jaccard, 1.0, jaccard_entries, side="left")
    style_axis(ax_jaccard)

    # --- VIF: log, right-censored at VIF_CAP; gate moved to 10 ---
    ax_vif.set_xscale("log")
    ax_vif.set_xlim(1, VIF_CAP * 1.3)
    ax_vif.set_title("VIF  (computed rules only; capped at "
                      f"{VIF_CAP:,.0f})", fontsize=12, fontweight="medium", loc="left")
    ax_vif.set_xlabel("VIF  (log, capped)")
    gate_line(ax_vif, VIF_GATE)
    gate_line(ax_vif, VIF_CAP, style=(0, (1, 2)))
    ax_vif.annotate("capped: ∞ is included here", xy=(VIF_CAP, 0.02),
                     xycoords=("data", "axes fraction"), xytext=(-13, 0),
                     textcoords="offset points", annotation_clip=False,
                     color="#8a8377", fontsize=7.8, ha="right", va="bottom",
                     rotation=90)
    # Both VIF reference lines use the same anchor rule as the other panels:
    # the dot sits at the curve height, the label reports the failing tail
    # beyond the line.
    gate_fracs = [float((data[n]["vif"] >= VIF_GATE).mean()) for n, _, _ in DATASETS]
    cap_fracs = [float((data[n]["vif"] >= VIF_CAP).mean()) for n, _, _ in DATASETS]
    vif_gate_entries = [(color, 1.0 - f, f"{s} ≥ {VIF_GATE:.0f}")
                        for (_, _, color), f, s in
                        zip(DATASETS, gate_fracs, pcts(gate_fracs))]
    vif_cap_entries = [(color, 1.0 - f, f"{s} ≥ 10³")
                       for (_, _, color), f, s in
                       zip(DATASETS, cap_fracs, pcts(cap_fracs))]
    label_group_at_x(ax_vif, VIF_GATE, vif_gate_entries, side="right")
    label_group_at_x(ax_vif, VIF_CAP, vif_cap_entries, side="left")
    style_axis(ax_vif)

    fig.legend(lines.values(), lines.keys(), loc="upper center",
               ncol=max(1, len(DATASETS)),
               frameon=False, bbox_to_anchor=(0.5, 1.015), fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {OUT_PATH.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
