#!/usr/bin/env python3
"""
Merge the per-rule interpretability signals into ONE trust table and route every
rule into a FOUR-TIER interpretability taxonomy.

Joins (on rule_id) the outputs of:
  rule_logreg_selected.csv  beta, body, support (fired_total), gold_fired, precision
    (falls back to rule_logreg_train.csv if the selected-L2 CSV is absent; they
     differ only when the selected L2 != the primary L2, e.g. WN18RR)
  rule_collinearity.csv     max_jaccard, vif

and assigns each rule a TIER describing how far its coefficient can be trusted:

  FULL      individually interpretable -- read the beta MAGNITUDE directly.
            Requires: enough data (fired_total >= --min_support AND minority-class
            events >= --min_epv), not separated, and not collinear (vif <
            --vif_max AND max_jaccard < --jaccard_max AND vif was computable).
  CLUSTER   enough data + not separated/low-EPV, but collinear (vif >= --vif_max
            OR max_jaccard >= --jaccard_max OR vif was never computed for this
            rule's head relation -- see vif_unknown below). The individual beta
            is not identifiable, but the effect is interpretable at the head-
            relation / near-duplicate CLUSTER level (only the SUM of the
            collinear group's betas is pinned).
  SIGN_ONLY enough total data (fired_total >= --min_support), NOT collinear, but
            the magnitude is not estimable: the rule is (quasi-)SEPARATED
            (gold_fired in {0, fired_total}: data wants |beta|=inf, an L2
            artifact) OR has too few minority-class events (minority <
            --min_epv). The SIGN is real; the magnitude is not. Separation and
            low-EPV are the SAME failure mode. Collinearity is excluded here on
            purpose: if the rule is ALSO entangled with a correlated partner,
            L2 can push weight (and flip sign) onto whichever partner looks
            better in-fold, so "the sign is real" is not a safe claim -- that
            case falls through to NOT instead.
  NOT       fails the hard floor (fired_total < --min_support), OR fails BOTH
            the separated/low-EPV gate AND the collinearity gate at once: too
            little data, or too entangled with something else, to trust either
            the magnitude or the sign.

The two support floors gate different quantities on purpose. fired_total is total
firings; minority = min(gold_fired, fired_total - gold_fired) is the events-per-
variable (EPV) count that actually pins a logistic coefficient's MAGNITUDE. Both
are ABSOLUTE floors (estimation adequacy), NOT per-dataset percentiles -- so a
tier means the same thing across datasets and stays comparable.

VIF ("vif_unknown"): analyze_rule_collinearity.py skips the (dense) VIF/condition-
number computation for head relations bigger than its --max_k_eig, and writes
"nan" for every rule in that relation. That "nan" is NOT evidence of low
collinearity -- it means collinearity was never checked. Such rules are folded
into the `collinear` gate (fail_reasons: "vif_unknown") rather than defaulting to
"not collinear", since Jaccard alone can miss multi-way collinearity.

A `fail_reasons` column lists every check a rule fails (non-exclusive), so "why
this tier" is explicit; the tier itself is decided by this PRECEDENCE:
  low_support                          -> NOT
  (separated or low_epv) AND collinear -> NOT   (unreliable AND entangled)
  separated or low_epv                 -> SIGN_ONLY
  collinear                            -> CLUSTER
  else                                 -> FULL

Output (in --out_dir, default = --logreg_dir):
  rule_trust_table.csv     per-rule merged table + tier + fail_reasons
  rule_trust_summary.json  tier breakdown + fail-reason breakdown + thresholds

Usage (from repo root):
    python scripts/build_rule_trust_table.py \\
        --logreg_dir outputs/additive_family/logreg-family
"""

import argparse
import csv
import json
import os
from collections import Counter


def read_csv_by_id(path, key="rule_id"):
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row[key])] = row
    return out


def fnum(row, k, default=""):
    if row is None or k not in row or row[k] in ("", "nan"):
        return default
    return row[k]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logreg_dir", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--min_support", type=int, default=20,
                   help="Min fired_total to trust ANYTHING (the NOT floor) -- a "
                        "SIGN-adequacy floor (magnitude is governed separately by "
                        "--min_epv). Default 20 is the family-wise binomial "
                        "threshold at which a PERFECTLY separated rule (all N "
                        "firings on one side) survives multiple-testing under a "
                        "worst-case balanced null: solve 0.5^(N-1) < 0.05/R, which "
                        "gives N=17 (R=2400, family), 19 (R=7382, wn18rr), 20 "
                        "(R=18400, umls) -- a tight band because N enters only "
                        "logarithmically in R. Real (lopsided) base rates only "
                        "lower it, so 20 is a conservative uniform floor; raise it "
                        "for more margin. An ABSOLUTE floor, not a percentile.")
    p.add_argument("--min_epv", type=int, default=10,
                   help="Min minority-class events -- min(gold_fired, "
                        "fired_total-gold_fired) -- to trust the MAGNITUDE. Below "
                        "this (or separated) the sign may hold but the magnitude "
                        "is unreliable -> SIGN_ONLY. Absolute EPV floor.")
    p.add_argument("--jaccard_max", type=float, default=0.8,
                   help="max_jaccard >= this -> collinear (near-duplicate column).")
    p.add_argument("--vif_max", type=float, default=4.0,
                   help="vif >= this -> collinear (multi-way; VIF 4 ~ R^2 0.75).")
    return p.parse_args()


# Tier order for stable reporting.
TIERS = ["FULL", "CLUSTER", "SIGN_ONLY", "NOT"]
GATES = ["low_support", "separated", "low_epv", "high_jaccard", "high_vif",
         "vif_unknown"]


def main():
    args = parse_args()
    d = args.logreg_dir
    out_dir = args.out_dir or d

    # Prefer the selected-L2 per-rule CSV (beta at the chosen L2); fall back to
    # the primary-L2 train CSV. They match unless selected L2 != primary L2.
    train_path = os.path.join(d, "rule_logreg_selected.csv")
    if not os.path.isfile(train_path):
        train_path = os.path.join(d, "rule_logreg_train.csv")
    train = read_csv_by_id(train_path)
    coll  = read_csv_by_id(os.path.join(d, "rule_collinearity.csv"))
    if not train:
        raise SystemExit(f"ERROR: no rule_logreg_*.csv found in {d}")
    print(f"beta/body source : {os.path.basename(train_path)}")

    fields = ["rule_id", "head", "length", "body", "beta",
              "support", "gold_fired", "minority_support", "precision",
              "max_jaccard", "vif", "tier", "fail_reasons"]

    rows_out = []
    for rid in sorted(train):
        t = train[rid]; c = coll.get(rid)
        support = int(fnum(t, "fired_total", "0"))
        max_j = float(fnum(c, "max_jaccard", "0") or 0)

        # Read the raw cell directly (NOT via fnum): fnum collapses "nan" to
        # its default, which would erase the distinction between "no
        # collinearity row at all" (vif_cell == "") and "row exists but VIF
        # was never computed for this relation" (vif_cell == "nan", from
        # analyze_rule_collinearity.py skipping relations > --max_k_eig).
        vif_cell = c.get("vif", "") if c is not None else ""
        vif_v = None
        vif_unknown = (vif_cell == "nan")
        if vif_cell not in ("", "nan"):
            try:
                vif_v = float(vif_cell)   # "inf" parses to float('inf')
            except ValueError:
                vif_v = None

        # gold_fired -> structural separation + minority-class ("event") count.
        gf = fnum(t, "gold_fired", "")
        gold_fired_v = int(gf) if gf not in ("", "nan") else None
        minority = (min(gold_fired_v, support - gold_fired_v)
                    if (gold_fired_v is not None and support > 0) else None)

        # --- Gate evaluation (all reported; tier decided by precedence below) ---
        low_support = support < args.min_support
        separated = (gold_fired_v is not None and support > 0 and
                     (gold_fired_v == 0 or gold_fired_v == support))
        low_epv = (not separated and minority is not None
                   and minority < args.min_epv)
        high_jaccard = max_j >= args.jaccard_max
        high_vif = vif_v is not None and vif_v >= args.vif_max

        reasons = []
        if low_support:  reasons.append("low_support")
        if separated:    reasons.append("separated")
        if low_epv:      reasons.append("low_epv")
        if high_jaccard: reasons.append("high_jaccard")
        if high_vif:     reasons.append("high_vif")
        if vif_unknown:  reasons.append("vif_unknown")

        # --- Tier by precedence -------------------------------------------------
        unreliable = separated or low_epv
        collinear = high_jaccard or high_vif or vif_unknown

        if low_support:
            tier = "NOT"
        elif unreliable and collinear:
            tier = "NOT"          # unreliable AND entangled -- can't vouch for sign or size
        elif unreliable:
            tier = "SIGN_ONLY"
        elif collinear:
            tier = "CLUSTER"
        else:
            tier = "FULL"

        rows_out.append(dict(
            rule_id=rid, head=fnum(t, "head"), length=fnum(t, "length"),
            body=fnum(t, "body"), beta=fnum(t, "beta"),
            support=support, gold_fired=fnum(t, "gold_fired"),
            minority_support=(minority if minority is not None else ""),
            precision=fnum(t, "precision"),
            max_jaccard=fnum(c, "max_jaccard", ""),
            vif=vif_cell,
            tier=tier,
            fail_reasons=";".join(reasons)))

    out_csv = os.path.join(out_dir, "rule_trust_table.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows_out)

    # ---- Summary statistics (printed AND persisted to JSON) -----------------
    n_supp = sum(1 for r in rows_out if r["support"] > 0)
    n_unsupported = sum(1 for r in rows_out if r["support"] <= 0)
    tiers = Counter({tname: 0 for tname in TIERS})
    for r in rows_out:
        tiers[r["tier"]] += 1

    # Fail-reason breakdown over ALL rules with >=1 failing gate. Gate names are
    # seeded to 0 so a zero-count gate still shows. cnt is NON-EXCLUSIVE (a rule
    # counts under every gate it fails, so values overlap); cnt_sole is EXCLUSIVE
    # (rules failing ONLY that gate = the non-overlapping part).
    cnt      = Counter({g: 0 for g in GATES})
    cnt_sole = Counter({g: 0 for g in GATES})
    n_failing = 0
    for r in rows_out:
        whys = [w for w in r["fail_reasons"].split(";") if w]
        if not whys:
            continue
        n_failing += 1
        for w in whys:
            cnt[w] += 1
        if len(whys) == 1:
            cnt_sole[whys[0]] += 1
    order = sorted(GATES, key=lambda g: -cnt[g])

    summary = {
        "logreg_dir": os.path.abspath(d),
        "run_label": os.path.basename(os.path.dirname(os.path.abspath(d))),
        "beta_source": os.path.basename(train_path),
        # Provenance: every gate is a property of the TRAIN fit.
        "computed_on": "train (leave-one-out design matrix)",
        "provenance_note": (
            "beta/support/gold_fired/precision and collinearity (max_jaccard, "
            "vif) are all on the TRAIN LOO design matrix."),
        "thresholds": {
            "min_support": args.min_support,
            "min_epv": args.min_epv,
            "jaccard_max": args.jaccard_max,
            "vif_max": args.vif_max,
        },
        "rules_total": len(rows_out),
        "rules_supported": n_supp,
        "rules_unsupported": n_unsupported,
        "tiers": {tname: tiers[tname] for tname in TIERS},
        "tiers_note": (
            "FOUR-TIER routing of every rule (sums to rules_total). "
            "FULL = read the beta magnitude directly (not separated/low-epv, "
            "not collinear). CLUSTER = beta not identifiable individually "
            "(collinear: high_jaccard, high_vif, or vif_unknown) but otherwise "
            "reliable; interpret at the head-relation / near-duplicate cluster "
            "level. SIGN_ONLY = enough total data, NOT collinear, but magnitude "
            "not estimable (separated or minority < min_epv); trust the sign, "
            "not the size. NOT = fired_total < min_support, OR the rule is BOTH "
            "separated/low-epv AND collinear at once (a correlated partner can "
            "absorb/flip weight under L2, so the sign can't be trusted either "
            "in that case). Tier precedence: low_support -> NOT; "
            "(separated or low_epv) AND collinear -> NOT; "
            "separated or low_epv -> SIGN_ONLY; collinear -> CLUSTER; else FULL."),
        "fail_reasons_note": (
            "Over ALL rules with >=1 failing gate. 'fail_reasons_all' is NON-"
            "EXCLUSIVE -- a rule is counted under EVERY gate it fails, so values "
            "overlap and sum to more than the failing-rule count. "
            "'fail_reasons_sole' counts rules failing ONLY that gate (the non-"
            "overlapping part). 'separated' and 'low_epv' are mutually exclusive "
            "by construction (separated = minority 0). 'vif_unknown' means "
            "analyze_rule_collinearity.py skipped the VIF computation for this "
            "rule's head relation (--max_k_eig) -- treated as collinear-unknown, "
            "not as evidence of low collinearity."),
        "fail_reasons_all": {g: cnt[g] for g in order},
        "fail_reasons_sole": {g: cnt_sole[g] for g in order},
    }
    out_json = os.path.join(out_dir, "rule_trust_summary.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"rules total          : {len(rows_out)}")
    print(f"rules with support>0 : {n_supp}")
    print(f"interpretability tiers (of {len(rows_out)} rules):")
    for tname in TIERS:
        print(f"  {tname:10s} {tiers[tname]:6d}  "
              f"({100.0*tiers[tname]/max(len(rows_out),1):.1f}%)")
    print(f"fail-reason breakdown ({n_failing} rules with >=1 failing gate; "
          f"reasons OVERLAP):")
    print(f"  {'gate':14s} {'fails':>6} {'sole':>6}")
    for g in order:
        print(f"  {g:14s} {cnt[g]:>6} {cnt_sole[g]:>6}")
    if n_unsupported:
        print(f"  ({n_unsupported} rules with support==0 -> all in NOT)")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
