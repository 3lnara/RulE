#!/usr/bin/env python3
"""
Compute per-rule empirical train PCA precision and its rank correlation
with frozen embedding confidence (add_ruleE = gamma_rule - dist).

PCA precision for rule R (head relation r, body path B):

    precision_R = #{(h,t): B grounds h->t  AND  t in TrueTrainTails(h, r)}
                / #{(h,t): B grounds h->t  AND  h in gold[r].keys()}

Denominator counts only firings from known subjects (heads with >=1 true
r-tail in train) — the subject-anchored Partial Completeness Assumption.
Grounding uses the full train graph (no leave-one-out).

Outputs (all written to --out_dir):
  rule_precision.pt           tensors: precision, support, gold_fired,
                              head, length, w_R_unclamped  (all indexed by rule_id)
  rule_precision_train.csv    per-rule table (one row per rule)
  precision_vs_confidence.png scatter plot (skipped with --no_plot)

Confidence source: reads w_R_unclamped from rule_meta.pt in --out_dir (primary).
Falls back to loading a checkpoint if rule_meta.pt is absent and --checkpoint
+ --config are provided. No checkpoint is needed for the grounding itself.

Usage (from repo root):
    python scripts/rule_precision_train.py \\
        --data_path data/umls \\
        --rule_file data/umls/mined_rules.txt \\
        --out_dir   outputs/additive_umls_aggregation_2x2/analysis

    python scripts/rule_precision_train.py \\
        --data_path data/family \\
        --rule_file data/family/mined_rules.txt \\
        --out_dir   outputs/additive_family/analysis
"""

import argparse
import csv
import math
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


# ---------------------------------------------------------------------------
# Statistics helpers (scipy optional for exact p-values).
# ---------------------------------------------------------------------------

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float((xc ** 2).sum()) * float((yc ** 2).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((xc * yc).sum() / denom)


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x, kind="stable")).astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="stable")).astype(np.float64)
    return pearson_r(rx, ry)


def fisher_ci_p(r: float, n: int):
    """Two-sided p-value and 95% CI via Fisher z-transform. Returns (p, lo, hi)."""
    if not math.isfinite(r) or n < 4:
        return float("nan"), float("nan"), float("nan")
    if abs(r) >= 1.0:
        return 0.0, r, r
    zr = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    p  = 2.0 * (1.0 - _norm_cdf(abs(zr / se)))
    lo = math.tanh(zr - 1.96 * se)
    hi = math.tanh(zr + 1.96 * se)
    return p, lo, hi


def correlation_report(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson + Spearman with Fisher-z CI/p. scipy used when available."""
    n = len(x)
    r = pearson_r(x, y)
    p_r, lo, hi = fisher_ci_p(r, n)
    rho = spearman_r(x, y)
    p_rho, _, _ = fisher_ci_p(rho, n)
    if _scipy_stats is not None and n >= 3:
        try:
            sr, sp = _scipy_stats.pearsonr(x, y)
            if math.isfinite(sr):
                r, p_r = float(sr), float(sp)
                _, lo, hi = fisher_ci_p(r, n)
            srho, sprho = _scipy_stats.spearmanr(x, y)
            if math.isfinite(srho):
                rho, p_rho = float(srho), float(sprho)
        except Exception:
            pass
    return dict(n=n,
                r=r,
                r2=(r * r if math.isfinite(r) else float("nan")),
                p=p_r,
                ci_lo=lo,
                ci_hi=hi,
                rho=rho,
                p_rho=p_rho)


def print_corr_row(label: str, rep: dict) -> None:
    ci = f"[{rep['ci_lo']:+.3f},{rep['ci_hi']:+.3f}]"
    print(f"  {label:<26}  n={rep['n']:>6}  "
          f"rho={rep['rho']:>+7.4f}  p_rho={rep['p_rho']:.2e}  "
          f"r={rep['r']:>+7.4f}  r2={rep['r2']:.4f}  "
          f"p={rep['p']:.2e}  95%CI={ci}")


# ---------------------------------------------------------------------------
# Build gold dict: gold[r][h] = set of true train tails for (h, r)
# ---------------------------------------------------------------------------

def build_gold(graph: MinimalGraph) -> dict:
    """gold[r][h] = frozenset of true train tails for directed relation r, head h.

    graph.train_facts already includes both forward (h, r, t) and inverse
    (t, r+NR, h) triples, so this covers all directed relations uniformly.
    """
    gold: dict = defaultdict(lambda: defaultdict(set))
    for h, r, t in graph.train_facts:
        gold[r][h].add(t)
    return gold


# ---------------------------------------------------------------------------
# PCA precision computation
# ---------------------------------------------------------------------------

def compute_precision(
    graph: MinimalGraph,
    rules: list,
    gold: dict,
    device: torch.device = None,
    chunk_size: int = None,
) -> tuple:
    """Return (fired_total, gold_fired) int64 numpy arrays indexed by rule_id.

    fired_total[gid] = #{(h,t): body fires from h to t,  h in gold[r].keys()}
    gold_fired[gid]  = #{(h,t): body fires from h to t,  h in gold[r].keys(),
                                 t in gold[r][h]}

    Rules for which no known subject exists get both counts = 0.

    device     : torch device for grounding (CPU or CUDA). Grounding tensors
                 (H_t, gold_mask) and MinimalGraph adjacency follow this device.
    chunk_size : process the known-subject heads of each relation in blocks of
                 this many at a time. gold_mask is [B, N] per relation, so for
                 large datasets (e.g. WN18RR, N ~= 40943, many known heads) the
                 full [B, N] mask can be large; chunking bounds peak memory.
                 None / <=0 means no chunking (single block per relation), which
                 reproduces the original behavior for UMLS / Family.
    """
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
    n_rels = len(relation2rules)
    for i_rel, (r, rule_list) in enumerate(sorted(relation2rules.items())):
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


# ---------------------------------------------------------------------------
# Scatter plot: precision (y) vs confidence (x), coloured by rule length
# ---------------------------------------------------------------------------

def make_scatter(
    precision: np.ndarray,
    confidence: np.ndarray,
    lengths: np.ndarray,
    support: np.ndarray,
    out_path: str,
    min_support: int = 1,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available; skipping scatter)")
        return

    mask = (support >= min_support) & np.isfinite(precision)
    p = precision[mask]
    c = confidence[mask]
    L = lengths[mask]

    unique_lengths = sorted(np.unique(L).tolist())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, ln in enumerate(unique_lengths):
        m = L == ln
        ax.scatter(c[m], p[m], s=10, alpha=0.4,
                   label=f"len={int(ln)}", color=cmap(i % 10), linewidths=0)

    ax.set_xlabel("Embedding confidence  (w_R = γ_rule − dist(body_emb, head_emb))")
    ax.set_ylabel("PCA precision  (train, known subjects only)")
    ax.set_title(
        f"Rule precision vs embedding confidence\n"
        f"(n={int(mask.sum())} rules with support≥{min_support})"
    )
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Scatter saved: {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

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
                   help="Where to read rule_meta.pt (confidence) and write outputs.")
    p.add_argument("--checkpoint",   default=None,
                   help="Pretrain checkpoint — used ONLY as fallback when rule_meta.pt "
                        "is absent in --out_dir.")
    p.add_argument("--config",       default=None,
                   help="Config JSON required together with --checkpoint.")
    p.add_argument("--min_support",  type=int, default=1,
                   help="Minimum fired_total to include a rule in the correlation "
                        "statistics. Rules below this threshold are kept in the CSV "
                        "but excluded from correlation/scatter (default 1 = any fire).")
    p.add_argument("--no_plot",      action="store_true",
                   help="Skip writing precision_vs_confidence.png.")
    p.add_argument("--per_relation", action="store_true",
                   help="Print an additional per-relation correlation table.")
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda"],
                   help="Device for grounding. 'cuda' scales the precision "
                        "computation to larger datasets (WN18RR/FB15k). "
                        "Falls back to cpu if CUDA is unavailable.")
    p.add_argument("--chunk_size",   type=int, default=0,
                   help="Process each relation's known-subject heads in blocks "
                        "of this size to bound the [B, N] gold_mask memory "
                        "(useful for WN18RR/FB15k). 0 = no chunking (default; "
                        "matches the original behavior for UMLS/Family).")
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
    print(f"scipy     : {'available (exact p-values)' if _scipy_stats else 'not found (Fisher-z approx)'}")

    # ---- Graph + rules -------------------------------------------------------
    print("\nLoading graph...")
    graph = MinimalGraph(args.data_path)
    NR = graph.relation_size
    N  = graph.entity_size
    print(f"  entities={N}  directed_relations={NR * 2}  train_facts={len(graph.train_facts)}")

    print("Loading rules...")
    rules = load_rules(rule_file, NR)
    R = len(rules)
    print(f"  num_rules={R}")

    # Build quick lookup: gid -> body (for CSV output)
    id_to_body = {rule[0]: rule[2:] for rule in rules}

    # ---- Rule metadata: head + body length are always derivable from the rule
    # file (gid == line index in load_rules), so the precision computation runs
    # fully from scratch -- no rule_meta.pt or checkpoint required.
    lengths_meta = np.array([len(r[2:]) for r in rules], dtype=np.int64)
    heads_meta   = np.array([r[1]       for r in rules], dtype=np.int64)

    # ---- Confidence (w_R) is OPTIONAL: it is only used for the
    # precision-vs-confidence correlation report + scatter and saved alongside
    # precision for reference. The additive precision*binary scorer reads only
    # `precision` from rule_precision.pt, so confidence is not needed to produce
    # usable weights. Resolution order: rule_meta.pt (if present) -> checkpoint
    # (+ config) -> none (skip correlation).
    w_R = None
    meta_path = os.path.join(args.out_dir, "rule_meta.pt")
    if os.path.exists(meta_path):
        print(f"\nLoading confidence from existing {meta_path} ...")
        meta = torch.load(meta_path, weights_only=False)
        w_R = meta["w_R_unclamped"].float().numpy()
        if len(w_R) != R:
            raise ValueError(
                f"rule_meta.pt has {len(w_R)} entries but rule_file has {R} rules. "
                "Ensure both come from the same training run.")
    elif args.checkpoint and args.config:
        print(f"\nrule_meta.pt absent; computing confidence from checkpoint "
              f"{args.checkpoint} ...")
        import json
        with open(args.config) as f:
            cfg = json.load(f)
        cdev = torch.device("cpu")
        model = _drc.build_model(cfg, graph, rules, cdev)
        ckpt  = torch.load(args.checkpoint, map_location=cdev)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()
        w_R = _drc.compute_unclamped_weights(model, cdev).numpy()
        if len(w_R) != R:
            raise ValueError(f"Checkpoint yielded {len(w_R)} weights but {R} rules loaded.")
    else:
        print("\nNo rule_meta.pt or --checkpoint provided; computing precision "
              "from scratch WITHOUT confidence (precision-vs-confidence "
              "correlation + scatter will be skipped).")

    has_confidence = w_R is not None
    if not has_confidence:
        # NaN placeholder so downstream save/CSV stay uniform; the additive
        # scorer ignores this field.
        w_R = np.full(R, np.nan, dtype=np.float64)
    else:
        n_neg = int((w_R < 0).sum())
        print(f"  confidence  range=[{w_R.min():.3f}, {w_R.max():.3f}]  "
              f"mean={w_R.mean():.3f}  negative={n_neg}/{R}")

    # ---- Gold dict: gold[r][h] = set of true train tails for (h, r) ----------
    print("\nBuilding gold dict from train facts...")
    gold = build_gold(graph)
    total_known = sum(len(v) for v in gold.values())
    print(f"  unique (r, h) pairs with >=1 gold tail: {total_known}")

    # ---- PCA precision computation -------------------------------------------
    print("\nComputing PCA precision (grounding from known-subject heads)...")
    fired_total, gold_fired = compute_precision(
        graph, rules, gold, device=device, chunk_size=args.chunk_size)

    support   = fired_total
    precision = np.where(support > 0,
                         gold_fired / support.astype(np.float64),
                         float("nan"))
    precision = precision.astype(np.float32)

    has_prec   = np.isfinite(precision)
    n_positive = int(has_prec.sum())
    n_minsup   = int((support >= args.min_support).sum())
    if n_positive > 0:
        prec_valid = precision[has_prec]
        print(f"  Rules with support>0        : {n_positive}/{R}")
        print(f"  Rules with support>={args.min_support:<3d}      : {n_minsup}/{R}")
        print(f"  Precision (support>0)  "
              f"range=[{prec_valid.min():.4f}, {prec_valid.max():.4f}]  "
              f"mean={prec_valid.mean():.4f}  median={np.median(prec_valid):.4f}")
    else:
        print("  WARNING: no rule fired from any known subject. Check data_path and rule_file.")

    # ---- Correlation analysis ------------------------------------------------
    # Restrict to rules with support >= min_support and finite precision.
    mask_all = has_prec & (support >= args.min_support)
    if not has_confidence:
        print("\nSkipping precision-vs-confidence correlation "
              "(no confidence available).")
    elif mask_all.sum() < 4:
        print("\nWARNING: fewer than 4 rules pass the support filter; "
              "correlation not computed.")
    else:
        p_all   = precision[mask_all].astype(np.float64)
        c_all   = w_R[mask_all].astype(np.float64)
        c_clamp = np.maximum(c_all, 0.0)
        L_all   = lengths_meta[mask_all]
        H_all   = heads_meta[mask_all]

        sep = "=" * 82
        print(f"\n{sep}")
        print(f"CORRELATION ANALYSIS  |  dataset={dataset}  "
              f"min_support={args.min_support}  n={int(mask_all.sum())}")
        print(sep)
        hdr = (f"  {'label':<26}  {'n':>6}  {'rho':>8}  {'p_rho':>10}  "
               f"{'r':>8}  {'r2':>7}  {'p':>10}  95%CI")
        div = "  " + "-" * 88
        print(hdr)
        print(div)

        rep_raw   = correlation_report(p_all, c_all)
        rep_clamp = correlation_report(p_all, c_clamp)
        print_corr_row("vs raw confidence",    rep_raw)
        print_corr_row("vs clamped conf (>=0)", rep_clamp)

        # Per-body-length breakdown
        print(f"\n  Per-body-length (precision vs raw confidence):")
        print(hdr)
        print(div)
        for ln in sorted(np.unique(L_all).tolist()):
            m = L_all == int(ln)
            if m.sum() < 4:
                print(f"  {'  length=' + str(int(ln)):<26}  n={m.sum():>6}  (too few; skipped)")
                continue
            rep_l = correlation_report(p_all[m], c_all[m])
            print_corr_row(f"  length={int(ln)}", rep_l)

        # Per-relation breakdown (optional)
        if args.per_relation:
            print(f"\n  Per-relation (precision vs raw confidence):")
            print(hdr)
            print(div)
            for rel in sorted(np.unique(H_all).tolist()):
                m = H_all == int(rel)
                if m.sum() < 4:
                    continue
                rep_r = correlation_report(p_all[m], c_all[m])
                print_corr_row(f"  rel={int(rel)}", rep_r)

        # Headline summary
        print(f"\n{sep}")
        print(f"HEADLINE")
        print(f"  Spearman(precision, raw_confidence)    = {rep_raw['rho']:>+7.4f}"
              f"  (p={rep_raw['p_rho']:.2e})")
        print(f"  Spearman(precision, clamped_conf >=0) = {rep_clamp['rho']:>+7.4f}"
              f"  (p={rep_clamp['p_rho']:.2e})")
        rho = rep_raw["rho"]
        if abs(rho) >= 0.5:
            verdict = ("High rank correlation: precision and confidence agree "
                       "substantially. Precision x binary scoring is a direct "
                       "comparison to confidence x binary clamp (Phase 2).")
        elif abs(rho) >= 0.25:
            verdict = ("Moderate rank correlation: partial agreement. Precision "
                       "and confidence diverge for a notable fraction of rules; "
                       "both are worth comparing as weighting strategies.")
        else:
            verdict = ("Low rank correlation: precision and confidence diverge "
                       "substantially. They measure different aspects of rule "
                       "quality; comparing their MRR impact is informative.")
        print(f"  -> {verdict}")
        print(sep)

    # ---- Save rule_precision.pt ----------------------------------------------
    out_pt = os.path.join(args.out_dir, "rule_precision.pt")
    torch.save({
        "precision":     torch.tensor(precision,    dtype=torch.float32),
        "support":       torch.tensor(support,      dtype=torch.long),
        "gold_fired":    torch.tensor(gold_fired,   dtype=torch.long),
        "head":          torch.tensor(heads_meta,   dtype=torch.long),
        "length":        torch.tensor(lengths_meta, dtype=torch.long),
        "w_R_unclamped": torch.tensor(w_R,          dtype=torch.float32),
    }, out_pt)
    print(f"\nSaved {out_pt}")

    # ---- Save per-rule CSV ---------------------------------------------------
    out_csv = os.path.join(args.out_dir, "rule_precision_train.csv")
    fieldnames = ["rule_id", "head", "length", "body",
                  "fired_total", "gold_fired", "precision", "w_R_unclamped"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for gid in range(R):
            body = id_to_body.get(gid, [])
            prec = float(precision[gid])
            writer.writerow({
                "rule_id":       gid,
                "head":          int(heads_meta[gid]),
                "length":        int(lengths_meta[gid]),
                "body":          " ".join(str(b) for b in body),
                "fired_total":   int(support[gid]),
                "gold_fired":    int(gold_fired[gid]),
                "precision":     f"{prec:.6f}" if math.isfinite(prec) else "nan",
                "w_R_unclamped": f"{float(w_R[gid]):.6f}",
            })
    print(f"Saved {out_csv}  ({R} rows)")

    # ---- Scatter plot --------------------------------------------------------
    if not args.no_plot and has_confidence:
        out_png = os.path.join(args.out_dir, "precision_vs_confidence.png")
        make_scatter(precision, w_R, lengths_meta, support, out_png, args.min_support)
    elif not args.no_plot:
        print("Skipping precision_vs_confidence.png (no confidence available).")

    print("\nDone.")


if __name__ == "__main__":
    main()
