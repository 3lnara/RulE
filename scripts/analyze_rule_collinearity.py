#!/usr/bin/env python3
"""Per-head-relation collinearity report for the logreg aggregator.

Reads design_matrix.pt from --logreg_dir and writes rule_jaccard_pairs.csv and
rule_collinearity.csv (the latter feeds build_rule_trust_table.py).

Usage:
    python scripts/analyze_rule_collinearity.py --logreg_dir outputs/<ds>/rq
"""

import argparse
import csv
import os

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logreg_dir", required=True,
                   help="Dir with design_matrix.pt and rule_logreg{,_selected}.pt.")
    p.add_argument("--design_matrix", default=None,
                   help="Override path to design_matrix.pt.")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--jaccard_thr", type=float, default=0.8,
                   help="Report pairs with Jaccard >= this (default 0.8).")
    p.add_argument("--dup_thr", type=float, default=0.999,
                   help="Jaccard >= this counts as an exact duplicate (default 0.999).")
    p.add_argument("--max_k_eig", type=int, default=4000,
                   help="Skip the dense VIF for relations with more than this many support>0 rules.")
    return p.parse_args()


def load_beta(logreg_dir):
    """Prefer the selected beta; fall back to the primary beta."""
    for name in ("rule_logreg_selected.pt", "rule_logreg.pt"):
        path = os.path.join(logreg_dir, name)
        if os.path.isfile(path):
            d = torch.load(path, map_location="cpu")
            return d["beta"].float().numpy()
    return None


def main():
    args = parse_args()
    out_dir = args.out_dir or args.logreg_dir
    dm_path = args.design_matrix or os.path.join(args.logreg_dir, "design_matrix.pt")
    if not os.path.isfile(dm_path):
        raise SystemExit(f"ERROR: {dm_path} not found. Run build_design_matrix_cache.py first.")

    dm = torch.load(dm_path, map_location="cpu")
    rows, cols = dm["rows"].long(), dm["cols"].long()
    M, R = int(dm["M"]), int(dm["R"])
    support = dm["support"].numpy()
    head    = dm["head"].numpy()
    length  = dm["length"].numpy()
    beta    = load_beta(args.logreg_dir)
    if beta is None:
        beta = np.full(R, np.nan)

    # CSC-style index: rule -> its row ids (sorted by column).
    order = torch.argsort(cols)
    cols_s, rows_s = cols[order], rows[order]
    col_start = torch.searchsorted(cols_s, torch.arange(R + 1))
    def rule_rows(r):
        return rows_s[col_start[r]:col_start[r + 1]]

    # Group support>0 rules by head relation.
    rel2rules = {}
    for r in range(R):
        if support[r] > 0:
            rel2rules.setdefault(int(head[r]), []).append(r)

    pair_csv = os.path.join(out_dir, "rule_jaccard_pairs.csv")
    rule_csv = os.path.join(out_dir, "rule_collinearity.csv")

    # Per-rule accumulators (over support>0 rules; others default to 0/NaN).
    max_j      = np.zeros(R)
    argmax_j   = np.full(R, -1, dtype=np.int64)
    n_high     = np.zeros(R, dtype=np.int64)
    dup_group  = np.full(R, -1, dtype=np.int64)
    vif        = np.full(R, np.nan)   # diag of inverse phi-correlation (per relation)
    next_group = 0

    pair_rows = []
    n_rel_analyzed = 0

    for rel in sorted(rel2rules):
        rl = rel2rules[rel]
        k = len(rl)
        n_rel_analyzed += 1
        if k == 1:
            vif[rl[0]] = 1.0   # lone column in its relation -> no collinearity
            continue

        local = {g: i for i, g in enumerate(rl)}
        # Build local sparse [M, k] and its Gram G = X^T X (k x k, dense).
        rr, cc = [], []
        for i, g in enumerate(rl):
            rws = rule_rows(g)
            rr.append(rws)
            cc.append(torch.full((rws.numel(),), i, dtype=torch.long))
        rr = torch.cat(rr); cc = torch.cat(cc)
        Xs = torch.sparse_coo_tensor(torch.stack([rr, cc]),
                                     torch.ones(rr.numel()), (M, k)).coalesce()
        G = torch.sparse.mm(Xs.t(), Xs).to_dense().numpy()   # [k, k] co-fire counts
        a = np.diag(G).copy()                                # supports
        union = a[:, None] + a[None, :] - G
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.where(union > 0, G / union, 0.0)
        np.fill_diagonal(J, 0.0)

        # Flagged pairs.
        iu, ju = np.triu_indices(k, k=1)
        Jv = J[iu, ju]
        high = Jv >= args.jaccard_thr

        for idx in np.nonzero(high)[0]:
            gi, gj = rl[iu[idx]], rl[ju[idx]]
            pair_rows.append(dict(
                head=rel, rule_i=gi, rule_j=gj, jaccard=f"{Jv[idx]:.4f}",
                support_i=int(a[iu[idx]]), support_j=int(a[ju[idx]]),
                beta_i=f"{beta[gi]:.6f}", beta_j=f"{beta[gj]:.6f}",
                len_i=int(length[gi]), len_j=int(length[gj])))

        # Per-rule max-Jaccard partner + high-partner count.
        for i, g in enumerate(rl):
            jrow = J[i]
            mj_i = int(np.argmax(jrow))
            max_j[g]    = float(jrow[mj_i])
            argmax_j[g] = rl[mj_i] if jrow[mj_i] > 0 else -1
            n_high[g]   = int((jrow >= args.jaccard_thr).sum())

        # Exact-duplicate groups (connected components at J >= dup_thr).
        adj = J >= args.dup_thr
        seen = np.zeros(k, dtype=bool)
        for i in range(k):
            if seen[i] or not adj[i].any():
                continue
            stack, comp = [i], []
            seen[i] = True
            while stack:
                u = stack.pop(); comp.append(u)
                for v in np.nonzero(adj[u] & ~seen)[0]:
                    seen[v] = True; stack.append(v)
            if len(comp) > 1:
                for u in comp:
                    dup_group[rl[u]] = next_group
                next_group += 1

        # Per-rule VIF from the phi (binary correlation) matrix.
        if k <= args.max_k_eig:
            mean = a / M
            std = np.sqrt(np.clip(mean * (1 - mean), 1e-12, None))
            cov = G / M - mean[:, None] * mean[None, :]
            C = cov / (std[:, None] * std[None, :])
            C = np.clip(C, -1.0, 1.0)
            np.fill_diagonal(C, 1.0)

            # VIF_i = (C^{-1})_{ii}. Duplicate columns make C singular (true VIF
            # inf): pinv for the rest, dup members stamped inf.
            try:
                vdiag = np.diag(np.linalg.inv(C))
            except np.linalg.LinAlgError:
                vdiag = np.diag(np.linalg.pinv(C))
            for i, g in enumerate(rl):
                vif[g] = np.inf if dup_group[g] >= 0 else max(float(vdiag[i]), 1.0)

    with open(pair_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["head", "rule_i", "rule_j", "jaccard",
                                           "support_i", "support_j", "beta_i",
                                           "beta_j", "len_i", "len_j"])
        w.writeheader()
        for row in sorted(pair_rows, key=lambda d: -float(d["jaccard"])):
            w.writerow(row)

    with open(rule_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule_id", "head", "length", "support",
                                           "beta", "max_jaccard", "argmax_partner",
                                           "n_high_partners", "dup_group_id", "vif"])
        w.writeheader()
        for r in range(R):
            v = vif[r]
            vif_s = "inf" if np.isinf(v) else (f"{v:.4f}" if np.isfinite(v) else "nan")
            w.writerow(dict(
                rule_id=r, head=int(head[r]), length=int(length[r]),
                support=int(support[r]),
                beta=f"{beta[r]:.6f}" if np.isfinite(beta[r]) else "nan",
                max_jaccard=f"{max_j[r]:.4f}", argmax_partner=int(argmax_j[r]),
                n_high_partners=int(n_high[r]), dup_group_id=int(dup_group[r]),
                vif=vif_s))

    n_dup_rules = int((dup_group >= 0).sum())
    print(f"relations analyzed : {n_rel_analyzed}")
    print(f"flagged pairs (J>={args.jaccard_thr}) : {len(pair_rows)}")
    print(f"rules in exact-dup groups (J>={args.dup_thr}) : {n_dup_rules} "
          f"in {next_group} groups")
    print(f"Wrote:\n  {pair_csv}\n  {rule_csv}")


if __name__ == "__main__":
    main()
