"""
Feature-validity screen (training-free).

A query-conditioned mixing head (adaptive beta) can only help if some per-query
feature actually predicts *where the rule scorer beats the KGE scorer*. This
script measures that directly, WITHOUT training any beta:

  rule_advantage(q) = RR_rule(q) - RR_kge(q)

where RR_* is the reciprocal of the filtered rank of the true tail under the
rule-only / KGE-only scores. For each candidate feature we report how strongly
it correlates with rule_advantage. A feature with a sizeable, sign-consistent
correlation is a usable router; one with r ~ 0 cannot help no matter how the
mixing head is tuned.

Reported per feature (vs rule_advantage):
  - Pearson r          linear association (the headline number)
  - r^2                fraction of advantage variance the feature linearly explains
  - 95% CI on r        Fisher-z interval (is the effect distinguishable from 0?)
  - p                  two-sided significance of r
  - Spearman rho       monotone association (robust to non-linearity / outliers)
  - r(feat, RR_rule)   decomposition: does the feature track rule strength ...
  - r(feat, RR_kge)    ... or (inversely) KGE strength?
  - promising          |r| >= threshold AND per-bucket advantage is monotone

The rule-only / KGE-only ranks reuse the exact `_filtered_rank` used by the
beta trainer, so the `density` / `num_rules` rows reproduce the FEATURE-VALIDITY
block printed during normal training (correctness anchor).
"""
import os
import sys

# RulE_original/src holds the model with the feature plumbing (_last_ground_mask,
# _last_num_rules) the screen depends on. Put it on the path before importing it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RulE_original', 'src'))

import argparse
import csv
import math

import torch
from torch.utils.data import DataLoader

from data import KnowledgeGraph, TestDataset, ValidDataset, RuleDataset  # noqa: F401
from model import RulE
from utils import load_config

# Reuse the trainer's filtered-rank logic verbatim so density/num_rules match
# the FEATURE-VALIDITY diagnostic emitted during beta training.
from train_beta_grounding import _filtered_rank

try:
    from scipy import stats as _scipy_stats  # exact p-values when available
except Exception:  # pragma: no cover - scipy is optional
    _scipy_stats = None


# --------------------------------------------------------------------------- #
# Statistics helpers (dependency-free; scipy only used for exact p-values).
# --------------------------------------------------------------------------- #
def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pearson_r(x, y):
    """Pearson correlation of two 1-D float tensors. NaN if either is constant."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if denom.item() <= 1e-12:
        return float('nan')
    return (xc * yc).sum().item() / denom.item()


def _ranks(x):
    """Average ranks would be ideal, but argsort-of-argsort matches the trainer's
    Spearman convention (ties broken by index) and is fine for screening."""
    return x.argsort().argsort().float()


def spearman_r(x, y):
    return pearson_r(_ranks(x), _ranks(y))


def fisher_ci_p(r, n):
    """Two-sided p-value and 95% CI for a correlation via the Fisher z-transform.

    p uses the z = atanh(r) * sqrt(n-3) ~ N(0,1) test; CI back-transforms
    atanh(r) +/- 1.96 / sqrt(n-3). Returns (p, lo, hi).
    """
    if not math.isfinite(r) or n < 4:
        return float('nan'), float('nan'), float('nan')
    if abs(r) >= 1.0:
        return 0.0, r, r
    zr = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zstat = zr / se
    p = 2.0 * (1.0 - _norm_cdf(abs(zstat)))
    lo = math.tanh(zr - 1.96 * se)
    hi = math.tanh(zr + 1.96 * se)
    return p, lo, hi


def correlation_report(x, y):
    """Full report for corr(x, y): Pearson r (+ p, CI), Spearman rho (+ p)."""
    n = x.numel()
    r = pearson_r(x, y)
    p_r, lo, hi = fisher_ci_p(r, n)
    rho = spearman_r(x, y)
    p_rho, _, _ = fisher_ci_p(rho, n)
    # Prefer scipy's exact p-values when present (Student-t for Pearson, the
    # AS 89 / asymptotic for Spearman); keep the Fisher-z CI either way.
    if _scipy_stats is not None and n >= 3:
        xn = x.cpu().numpy()
        yn = y.cpu().numpy()
        try:
            sr, sp = _scipy_stats.pearsonr(xn, yn)
            if math.isfinite(sr):
                r, p_r = float(sr), float(sp)
                _, lo, hi = fisher_ci_p(r, n)
            srho, sprho = _scipy_stats.spearmanr(xn, yn)
            if math.isfinite(srho):
                rho, p_rho = float(srho), float(sprho)
        except Exception:
            pass
    return {
        'n': n,
        'r': r,
        'r2': (r * r) if math.isfinite(r) else float('nan'),
        'p': p_r,
        'ci_lo': lo,
        'ci_hi': hi,
        'rho': rho,
        'p_rho': p_rho,
    }


def is_monotone(seq, tol=1e-9):
    """True if the sequence is (weakly) non-decreasing or non-increasing."""
    vals = [v for v in seq if math.isfinite(v)]
    if len(vals) < 2:
        return False
    inc = all(vals[i + 1] >= vals[i] - tol for i in range(len(vals) - 1))
    dec = all(vals[i + 1] <= vals[i] + tol for i in range(len(vals) - 1))
    return inc or dec


def quantile_buckets(feat, num_buckets):
    """Return a list of (label, index_tensor) splitting feat into quantile bins.

    Edges are de-duplicated, so heavily tied features (e.g. integer num_rules)
    collapse into fewer, still-meaningful buckets instead of empty ones.
    """
    n = feat.numel()
    if n == 0:
        return []
    qs = torch.linspace(0, 1, num_buckets + 1, dtype=feat.dtype)[1:-1]
    if qs.numel() == 0:
        edges = torch.zeros(0, dtype=feat.dtype)
    else:
        edges = torch.quantile(feat, qs)
        edges = torch.unique(edges)
    if edges.numel() == 0:
        return [("all", torch.arange(n))]
    bucket_id = torch.bucketize(feat, edges, right=False)
    out = []
    nb = int(bucket_id.max().item()) + 1
    for b in range(nb):
        idx = (bucket_id == b).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        lo = feat[idx].min().item()
        hi = feat[idx].max().item()
        out.append((f"[{lo:.4g}, {hi:.4g}]", idx))
    return out


# --------------------------------------------------------------------------- #
# Per-query feature extraction.
# --------------------------------------------------------------------------- #
def _candidate_scores(score_row, flag_row):
    """Scores restricted to the filtered candidate set for one query."""
    return score_row[flag_row]


def _margin(scores):
    """top1 - top2 (confidence of the winner). 0 if < 2 candidates."""
    if scores.numel() < 2:
        return 0.0
    top2 = torch.topk(scores, 2).values
    return (top2[0] - top2[1]).item()


def _entropy(scores):
    """Shannon entropy of softmax(scores). Flat distribution -> high entropy."""
    if scores.numel() < 2:
        return 0.0
    p = torch.softmax(scores.float(), dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())


def collect(model, dataloader, device, feature_names):
    """Single pass over the split. Returns dict feature_name -> tensor[N] plus
    RR_rule, RR_kge, rule_advantage, and global anchor stats.
    """
    feats = {name: [] for name in feature_names}
    rr_rule, rr_kge = [], []
    rels = []

    with torch.no_grad():
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)

            rule_logits, mask = model(all_h, all_r, None)
            kge_score = model.compute_g_KGE(all_h, all_r)

            # Vectorized whole-row features (match get_query_feature semantics).
            ground_mask = getattr(model, '_last_ground_mask', None)
            num_rules_stash = getattr(model, '_last_num_rules', None)

            B = all_t.size(0)
            for k in range(B):
                t = all_t[k]
                r_rank = _filtered_rank(rule_logits[k], flag[k], mask[k], t)
                k_rank = _filtered_rank(kge_score[k], flag[k], mask[k], t)
                rr_rule.append(1.0 / r_rank)
                rr_kge.append(1.0 / k_rank)
                rels.append(int(all_r[k].item()))

                rule_c = _candidate_scores(rule_logits[k], flag[k])
                kge_c = _candidate_scores(kge_score[k], flag[k])

                for name in feature_names:
                    if name == 'density':
                        val = (ground_mask[k].float().mean().item()
                               if ground_mask is not None else float('nan'))
                    elif name == 'num_rules':
                        val = (float(num_rules_stash[k].item())
                               if num_rules_stash is not None else float('nan'))
                    elif name == 'kge_margin':
                        val = _margin(kge_c)
                    elif name == 'kge_entropy':
                        val = _entropy(kge_c)
                    elif name == 'kge_max':
                        val = kge_c.max().item() if kge_c.numel() else float('nan')
                    elif name == 'rule_margin':
                        val = _margin(rule_c)
                    elif name == 'rule_entropy':
                        val = _entropy(rule_c)
                    elif name == 'rule_kge_disagreement':
                        # 1 - Spearman(rule, kge) over candidates: how much the
                        # two scorers reorder the candidate set for this query.
                        if rule_c.numel() >= 2:
                            val = 1.0 - pearson_r(_ranks(rule_c), _ranks(kge_c))
                        else:
                            val = float('nan')
                    else:
                        raise ValueError(f"Unknown feature: {name}")
                    feats[name].append(val)

    rr_rule_t = torch.tensor(rr_rule, dtype=torch.float64)
    rr_kge_t = torch.tensor(rr_kge, dtype=torch.float64)
    advantage = rr_rule_t - rr_kge_t
    feat_t = {name: torch.tensor(vals, dtype=torch.float64)
              for name, vals in feats.items()}
    return {
        'features': feat_t,
        'rr_rule': rr_rule_t,
        'rr_kge': rr_kge_t,
        'advantage': advantage,
        'rels': torch.tensor(rels, dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def print_bucket_table(name, feat, advantage, rr_rule, rr_kge, num_buckets):
    print(f"\nPer-bucket advantage for '{name}' ({num_buckets} quantile bins):")
    header = (f"  {'bucket':>22} {'n':>6} {'mean_feat':>10} "
              f"{'RR_rule':>9} {'RR_kge':>9} {'advantage':>10} {'rule>kge':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    adv_seq = []
    for label, idx in quantile_buckets(feat, num_buckets):
        n = idx.numel()
        mf = feat[idx].mean().item()
        rru = rr_rule[idx].mean().item()
        rrk = rr_kge[idx].mean().item()
        adv = advantage[idx].mean().item()
        frac = (advantage[idx] > 0).float().mean().item()
        adv_seq.append(adv)
        print(f"  {label:>22} {n:>6d} {mf:>10.4f} "
              f"{rru:>9.4f} {rrk:>9.4f} {adv:>+10.4f} {frac:>9.2f}")
    return adv_seq


def print_oracle_ceiling(rr_rule, rr_kge):
    """Quantify the maximum MRR any mixing strategy could reach.

    - best_single  = max(MRR_rule, MRR_kge): what you get with the better
                     scorer alone (no mixing). A *global* beta / fixed alpha
                     can at best recover this.
    - oracle       = mean(max(RR_rule, RR_kge)): a perfect *per-query* router
                     that always picks the better scorer for each query. This
                     is the hard ceiling for ANY adaptive head.
    - floor        = mean(min(...)): the worst per-query pick (context only).

    `oracle - best_single` is the entire prize available to per-query routing.
    If it's tiny, no feature (found or not) can help much and fixed alpha is
    already near-optimal; if it's large, the complementarity is real and the
    search is for a better router/feature.
    """
    mean_rule = rr_rule.mean().item()
    mean_kge = rr_kge.mean().item()
    best_single = max(mean_rule, mean_kge)
    oracle = torch.maximum(rr_rule, rr_kge).mean().item()
    floor = torch.minimum(rr_rule, rr_kge).mean().item()
    gain = oracle - best_single
    rel = (gain / best_single * 100.0) if best_single > 0 else float('nan')
    winner = 'rule' if mean_rule >= mean_kge else 'kge'
    print(f"\n{'-'*70}")
    print("ORACLE CEILING (max gain available to per-query routing)")
    print(f"{'-'*70}")
    print(f"  MRR rule-only           : {mean_rule:.4f}")
    print(f"  MRR kge-only            : {mean_kge:.4f}")
    print(f"  Best single scorer      : {best_single:.4f}  ({winner}-only)")
    print(f"  Per-query oracle        : {oracle:.4f}  (always pick better scorer)")
    print(f"  Per-query floor         : {floor:.4f}  (always pick worse scorer)")
    print(f"  Headroom oracle-best    : {gain:+.4f}  ({rel:+.1f}% over best single)")


def relation_label(rid, id2relation, relation_size):
    """Human-readable name for a (possibly inverse) relation id."""
    if rid < relation_size:
        return id2relation.get(rid, str(rid))
    base = id2relation.get(rid - relation_size, str(rid - relation_size))
    return f"{base}_inv"


def per_relation_report(rels, rr_rule, rr_kge, advantage,
                        id2relation, relation_size, top_k, csv_path=None):
    """Per-relation advantage breakdown + the per-relation routing ceiling.

    Per-relation beta is a *coarse* router: it can pick a different rule/KGE
    balance per relation, but cannot distinguish queries within a relation.
    Its ceiling is therefore mean over queries of the better *per-relation*
    scorer:

        perrel_oracle = sum_r n_r * max(MRR_rule_r, MRR_kge_r) / N

    Comparing this to the global best-single tells you whether a per-relation
    head has anything to exploit at all (gap > 0 means some relations are
    rule-favored and others KGE-favored).
    """
    N = rels.numel()
    uniq = torch.unique(rels).tolist()
    stats = []
    weighted_oracle = 0.0
    for rid in uniq:
        idx = (rels == rid).nonzero(as_tuple=True)[0]
        nr = idx.numel()
        mrr_rule = rr_rule[idx].mean().item()
        mrr_kge = rr_kge[idx].mean().item()
        adv = advantage[idx].mean().item()
        frac = (advantage[idx] > 0).float().mean().item()
        weighted_oracle += nr * max(mrr_rule, mrr_kge)
        stats.append({
            'rel_id': rid,
            'relation': relation_label(rid, id2relation, relation_size),
            'n': nr,
            'mrr_rule': mrr_rule,
            'mrr_kge': mrr_kge,
            'advantage': adv,
            'frac_rule_gt_kge': frac,
        })
    perrel_oracle = weighted_oracle / N if N > 0 else float('nan')
    best_single = max(rr_rule.mean().item(), rr_kge.mean().item())
    gap = perrel_oracle - best_single

    n_rule_fav = sum(1 for s in stats if s['advantage'] > 0)
    n_kge_fav = sum(1 for s in stats if s['advantage'] < 0)

    print(f"\n{'-'*70}")
    print("PER-RELATION ROUTING (ceiling for per-relation beta)")
    print(f"{'-'*70}")
    print(f"  Relations (with queries): {len(stats)}")
    print(f"  Rule-favored / KGE-favored: {n_rule_fav} / {n_kge_fav}")
    print(f"  Best single scorer       : {best_single:.4f}")
    print(f"  Per-relation oracle      : {perrel_oracle:.4f}  "
          f"(pick better scorer per relation)")
    print(f"  Headroom perrel-best     : {gap:+.4f}  "
          f"({(gap / best_single * 100.0) if best_single > 0 else float('nan'):+.1f}%)")

    stats.sort(key=lambda s: s['advantage'], reverse=True)
    name_w = 48
    header = (f"  {'relation':<{name_w}} {'n':>6} {'MRR_rule':>9} {'MRR_kge':>9} "
              f"{'advantage':>10} {'rule>kge':>9}")

    def _fit(name, width):
        # Middle-truncate so the discriminative suffix (e.g. '_inv') survives.
        if len(name) <= width:
            return name
        keep = width - 1
        head = keep // 2
        return name[:head] + '\u2026' + name[-(keep - head):]

    def _print_rows(rows):
        for s in rows:
            print(f"  {_fit(s['relation'], name_w):<{name_w}} {s['n']:>6d} "
                  f"{s['mrr_rule']:>9.4f} {s['mrr_kge']:>9.4f} "
                  f"{s['advantage']:>+10.4f} {s['frac_rule_gt_kge']:>9.2f}")

    k = min(top_k, len(stats))
    print(f"\n  Top {k} RULE-favored relations:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    _print_rows(stats[:k])
    print(f"\n  Top {k} KGE-favored relations:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    _print_rows(stats[-k:][::-1])

    if csv_path:
        stats_by_adv = sorted(stats, key=lambda s: s['advantage'], reverse=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_by_adv[0].keys()))
            writer.writeheader()
            for s in stats_by_adv:
                writer.writerow(s)
        print(f"\nPer-relation stats written to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Training-free feature-validity screen for adaptive beta.")
    parser.add_argument('--config', required=True, help='Path to model config json.')
    parser.add_argument('--checkpoint', required=True,
                        help='Grounding checkpoint (same one beta training uses).')
    parser.add_argument('--split', choices=['test', 'valid'], default='test',
                        help='Which split to screen on (default: test).')
    parser.add_argument(
        '--features',
        default='density,num_rules,kge_margin,kge_entropy,kge_max,'
                'rule_margin,rule_entropy',
        help='Comma-separated features. Available: density, num_rules, '
             'kge_margin, kge_entropy, kge_max, rule_margin, rule_entropy, '
             'rule_kge_disagreement.')
    parser.add_argument('--num_buckets', type=int, default=5)
    parser.add_argument('--promising_r', type=float, default=0.1,
                        help='|Pearson r| threshold for the "promising" flag.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override dataset grouping batch size (use 1 for '
                             'very large graphs to cap memory).')
    parser.add_argument('--output', default=None,
                        help='CSV output path for per-feature stats.')
    parser.add_argument('--per_relation_topk', type=int, default=15,
                        help='How many rule/KGE-favored relations to print '
                             '(0 disables the per-relation report).')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_names = [f.strip() for f in args.features.split(',') if f.strip()]

    print(f"Using device: {device}")
    print(f"Split: {args.split}")
    print(f"Features: {feature_names}")
    print(f"scipy available for exact p-values: {_scipy_stats is not None}")

    # ---- Load config / data / model / rules (mirrors train_beta_grounding) ----
    config = load_config(args.config)
    if isinstance(config, (list, tuple)):
        config = config[0]

    data_path = config.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(args.config), data_path)
    data_path = os.path.normpath(data_path)
    dataset_name = os.path.basename(data_path)

    print("Loading data...")
    graph = KnowledgeGraph(data_path)

    print("Loading model...")
    model = RulE(
        graph=graph,
        p_norm=config.p_norm,
        mlp_rule_dim=config.mlp_rule_dim,
        gamma_fact=config.gamma_fact,
        gamma_rule=config.gamma_rule,
        hidden_dim=config.hidden_dim,
        device=device,
        dataset=config.data_path,
        feature_name='density',
        use_per_relation=True,
    )

    rule_file = config.rule_file
    if not os.path.isabs(rule_file):
        rule_file = os.path.join(os.path.dirname(args.config), rule_file)
    rule_file = os.path.normpath(rule_file)
    rule_negative_size = getattr(config, 'rule_negative_size', 32)
    ruleset = RuleDataset(graph.relation_size, rule_file, rule_negative_size)
    rules = [rule[0] for rule in ruleset.rules]
    model.set_rules(rules)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    result = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)
    model = model.to(device)
    model.eval()
    model.eval_compute_rule_weight(device)

    batch_size = args.batch_size or getattr(config, 'g_batch_size', 8)
    if args.split == 'test':
        screen_set = TestDataset(graph, batch_size)
    else:
        screen_set = ValidDataset(graph, batch_size)
    dataloader = DataLoader(screen_set, batch_size=1, num_workers=0)

    print(f"\n{'='*70}")
    print(f"FEATURE-VALIDITY SCREEN | dataset={dataset_name} | split={args.split}")
    print(f"{'='*70}")

    data = collect(model, dataloader, device, feature_names)
    advantage = data['advantage']
    rr_rule = data['rr_rule']
    rr_kge = data['rr_kge']
    n = advantage.numel()

    # ---- Global anchors (reproduce the training-time diagnostic) ----
    mean_adv = advantage.mean().item()
    frac_rules_better = (advantage > 0).float().mean().item()
    print(f"\nQueries (N)            : {n}")
    print(f"Mean RR_rule           : {rr_rule.mean().item():.4f}")
    print(f"Mean RR_kge            : {rr_kge.mean().item():.4f}")
    print(f"Mean rule_advantage    : {mean_adv:+.4f}")
    print(f"Frac queries rule>kge  : {frac_rules_better:.3f}")

    # ---- Oracle ceiling: max gain available to ANY per-query router ----
    print_oracle_ceiling(rr_rule, rr_kge)

    # ---- Per-feature correlation table ----
    rows = []
    for name in feature_names:
        feat = data['features'][name]
        rep = correlation_report(feat, advantage)
        rep_rule = correlation_report(feat, rr_rule)
        rep_kge = correlation_report(feat, rr_kge)
        adv_seq = quantile_bucket_adv(feat, advantage, args.num_buckets)
        promising = (math.isfinite(rep['r'])
                     and abs(rep['r']) >= args.promising_r
                     and is_monotone(adv_seq))
        rows.append({
            'feature': name,
            'n': rep['n'],
            'r_adv': rep['r'],
            'p_adv': rep['p'],
            'ci_lo': rep['ci_lo'],
            'ci_hi': rep['ci_hi'],
            'r2_adv': rep['r2'],
            'rho_adv': rep['rho'],
            'p_rho_adv': rep['p_rho'],
            'r_RRrule': rep_rule['r'],
            'r_RRkge': rep_kge['r'],
            'promising': promising,
        })

    rows.sort(key=lambda d: (abs(d['r_adv']) if math.isfinite(d['r_adv']) else -1),
              reverse=True)

    print(f"\n{'-'*112}")
    print(f"{'feature':>22} {'r_adv':>8} {'95% CI':>17} {'p':>9} "
          f"{'r2':>7} {'rho':>8} {'p_rho':>9} {'r:RRrule':>9} {'r:RRkge':>9} {'flag':>5}")
    print(f"{'-'*112}")
    for d in rows:
        ci = f"[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]"
        flag = 'YES' if d['promising'] else ''
        print(f"{d['feature']:>22} {d['r_adv']:>+8.3f} {ci:>17} {d['p_adv']:>9.2e} "
              f"{d['r2_adv']:>7.3f} {d['rho_adv']:>+8.3f} {d['p_rho_adv']:>9.2e} "
              f"{d['r_RRrule']:>+9.3f} {d['r_RRkge']:>+9.3f} {flag:>5}")
    print(f"{'-'*112}")
    print("Note: with large N even tiny correlations get small p-values; judge "
          "usefulness by |r_adv| / r2 and the 95% CI, not by p alone.")

    # ---- Per-bucket tables for the anchors + top feature ----
    bucket_targets = []
    for anchor in ('density', 'num_rules'):
        if anchor in feature_names:
            bucket_targets.append(anchor)
    if rows:
        top = rows[0]['feature']
        if top not in bucket_targets:
            bucket_targets.append(top)
    for name in bucket_targets:
        print_bucket_table(name, data['features'][name], advantage,
                           rr_rule, rr_kge, args.num_buckets)

    # ---- CSV ----
    out_path = args.output or f"feature_validity_{dataset_name}_{args.split}.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for d in rows:
            writer.writerow(d)
    print(f"\nPer-feature stats written to: {out_path}")

    # ---- Per-relation routing ceiling + breakdown ----
    if args.per_relation_topk > 0:
        if out_path.endswith('.csv'):
            perrel_csv = out_path[:-4] + '_per_relation.csv'
        else:
            perrel_csv = out_path + '_per_relation.csv'
        per_relation_report(
            data['rels'], rr_rule, rr_kge, advantage,
            graph.id2relation, graph.relation_size,
            top_k=args.per_relation_topk, csv_path=perrel_csv,
        )


def quantile_bucket_adv(feat, advantage, num_buckets):
    """Mean advantage per quantile bucket (for the monotonicity test)."""
    return [advantage[idx].mean().item()
            for _, idx in quantile_buckets(feat, num_buckets)]


if __name__ == '__main__':
    main()
