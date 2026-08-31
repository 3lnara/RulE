#!/usr/bin/env python3
"""Merge the per-rule interpretability signals into one trust table.

Joins, on rule_id, rule_logreg_selected.csv (beta, support, true_fired,
precision; falls back to rule_logreg_train.csv) with rule_collinearity.csv
(max_jaccard, vif), and derives `minority_support` = min(true_fired, support -
true_fired) and a `separated` flag (true_fired in {0, support}).
Writes rule_trust_table.csv and rule_trust_summary.json.

Usage:
    python scripts/build_rule_trust_table.py --logreg_dir outputs/<ds>/rq
"""

import argparse
import csv
import json
import os


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
    return p.parse_args()


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
              "support", "true_fired", "minority_support", "separated",
              "precision", "max_jaccard", "vif"]

    rows_out = []
    n_sep = n_vif_unknown = 0
    for rid in sorted(train):
        t = train[rid]; c = coll.get(rid)
        support = int(fnum(t, "fired_total", "0"))

        # Raw cell, not via fnum: keep "" (no collinearity row) distinct from
        # "nan" (row exists, VIF not computed for this relation).
        vif_cell = c.get("vif", "") if c is not None else ""
        if vif_cell == "nan":
            n_vif_unknown += 1

        # true_fired -> structural separation + minority-class ("event") count.
        # (The fitting scripts' CSVs name this column gold_fired.)
        gf = fnum(t, "gold_fired", "")
        true_fired_v = int(gf) if gf not in ("", "nan") else None
        defined = true_fired_v is not None and support > 0
        minority = min(true_fired_v, support - true_fired_v) if defined else None
        separated = defined and (true_fired_v == 0 or true_fired_v == support)
        if separated:
            n_sep += 1

        rows_out.append(dict(
            rule_id=rid, head=fnum(t, "head"), length=fnum(t, "length"),
            body=fnum(t, "body"), beta=fnum(t, "beta"),
            support=support, true_fired=gf,
            minority_support=(minority if minority is not None else ""),
            separated=(int(separated) if defined else ""),
            precision=fnum(t, "precision"),
            max_jaccard=fnum(c, "max_jaccard", ""),
            vif=vif_cell))

    out_csv = os.path.join(out_dir, "rule_trust_table.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows_out)

    # ---- Summary statistics (printed AND persisted to JSON) -----------------
    n_supp = sum(1 for r in rows_out if r["support"] > 0)
    n_unsupported = len(rows_out) - n_supp

    summary = {
        "logreg_dir": os.path.abspath(d),
        "run_label": os.path.basename(os.path.dirname(os.path.abspath(d))),
        "beta_source": os.path.basename(train_path),
        # Provenance: every column is a property of the TRAIN fit.
        "computed_on": "train (leave-one-out design matrix)",
        "provenance_note": (
            "beta/support/true_fired/precision and collinearity (max_jaccard, "
            "vif) are all on the TRAIN LOO design matrix. true_fired is the "
            "column the fitting scripts call gold_fired."),
        "rules_total": len(rows_out),
        "rules_supported": n_supp,
        "rules_unsupported": n_unsupported,
        "rules_separated": n_sep,
        "rules_vif_unknown": n_vif_unknown,
    }
    out_json = os.path.join(out_dir, "rule_trust_summary.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"rules total          : {len(rows_out)}")
    print(f"rules with support>0 : {n_supp}")
    print(f"separated rules      : {n_sep}")
    print(f"vif unknown (skipped): {n_vif_unknown}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
