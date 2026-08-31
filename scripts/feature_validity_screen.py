"""Training-free feature-validity screen for the adaptive-beta mixing head.

Screens the five features the beta trainers accept
(density, num_rules, top_rule_confidence, kge_max, kge_entropy).
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


def _filtered_rank(scores_row, flag_row, mask_row, t):
    """Tie-averaged, filtered rank of true tail t (same rule as train_beta_grounding_chunked.py)."""
    if mask_row[t].item():
        val = scores_row[t]
        L = (scores_row[flag_row] > val).sum().item() + 1
        H = (scores_row[flag_row] >= val).sum().item() + 2
    else:
        L = 1
        H = flag_row.numel() + 1
    return (L + H - 1) / 2.0


# The five features the screen supports; identical to the beta trainers'
# --feature choices, so a feature that screens well can be trained directly.
FEATURE_CHOICES = ('density', 'num_rules', 'top_rule_confidence',
                   'kge_max', 'kge_entropy')


# --------------------------------------------------------------------------- #
# Statistics helpers.
# --------------------------------------------------------------------------- #
def pearson_r(x, y):
    """Pearson correlation of two 1-D float tensors. NaN if either is constant."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if denom.item() <= 1e-12:
        return float('nan')
    return (xc * yc).sum().item() / denom.item()


def fisher_ci(r, n):
    """95% CI for a correlation via the Fisher z-transform.

    Back-transforms atanh(r) +/- 1.96 / sqrt(n-3). Returns (lo, hi).
    """
    if not math.isfinite(r) or n < 4:
        return float('nan'), float('nan')
    if abs(r) >= 1.0:
        return r, r
    zr = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    return math.tanh(zr - 1.96 * se), math.tanh(zr + 1.96 * se)


def correlation_report(x, y):
    """Pearson r of (x, y) plus its Fisher-z 95% CI."""
    n = x.numel()
    r = pearson_r(x, y)
    lo, hi = fisher_ci(r, n)
    return {'n': n, 'r': r, 'ci_lo': lo, 'ci_hi': hi}


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


def _entropy(scores):
    """Shannon entropy of softmax(scores). Flat distribution -> high entropy."""
    if scores.numel() < 2:
        return 0.0
    p = torch.softmax(scores.float(), dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())


def _filtered_rr_grid(rule_row, kge_row, beta_grid, flag_row, mask_row, t):
    """Reciprocal filtered rank of the true tail under beta*rule + (1-beta)*kge for
    every beta in beta_grid (one [B, E] broadcast); returns [B] float64 on CPU."""
    B = beta_grid.numel()
    if bool(mask_row[t].item()):
        bg = beta_grid.to(rule_row.device).view(B, 1)
        mix = bg * rule_row.view(1, -1) + (1.0 - bg) * kge_row.view(1, -1)   # [B, E]
        true_val = mix[:, t].view(B, 1)                                      # [B, 1]
        cand = mix[:, flag_row]                                              # [B, C]
        # Tie-averaged rank: midpoint of strict-greater and greater-equal counts,
        # matching _filtered_rank's (L + H - 1) / 2 with L = #(>)+1, H = #(>=)+2.
        gt = (cand > true_val).sum(dim=1).double() + 1.0
        ge = (cand >= true_val).sum(dim=1).double() + 2.0
        rank = (gt + ge - 1.0) / 2.0
    else:
        rank = torch.full((B,), (flag_row.numel() + 1) / 2.0, dtype=torch.float64)
    return (1.0 / rank).cpu()


def collect(model, dataloader, device, feature_names, beta_grid):
    """One pass over the split: per-feature tensors, RR_rule/RR_kge, rule_advantage,
    relation ids, and rr_mix [N, B] (RR per beta_grid column)."""
    feats = {name: [] for name in feature_names}
    rr_rule, rr_kge = [], []
    rr_mix = []
    rels = []
    beta_grid_dev = beta_grid.to(device)

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
            top_conf_stash = getattr(model, '_last_top_rule_conf', None)

            B = all_t.size(0)
            for k in range(B):
                t = all_t[k]
                r_rank = _filtered_rank(rule_logits[k], flag[k], mask[k], t)
                k_rank = _filtered_rank(kge_score[k], flag[k], mask[k], t)
                rr_rule.append(1.0 / r_rank)
                rr_kge.append(1.0 / k_rank)
                rels.append(int(all_r[k].item()))

                # Convex beta grid; reuses the exact filtered-rank logic so its
                # endpoints are directly comparable to rr_rule / rr_kge above.
                rr_mix.append(_filtered_rr_grid(rule_logits[k], kge_score[k],
                                                beta_grid_dev, flag[k], mask[k], t))

                kge_c = _candidate_scores(kge_score[k], flag[k])

                for name in feature_names:
                    if name == 'density':
                        val = (ground_mask[k].float().mean().item()
                               if ground_mask is not None else float('nan'))
                    elif name == 'num_rules':
                        val = (float(num_rules_stash[k].item())
                               if num_rules_stash is not None else float('nan'))
                    elif name == 'top_rule_confidence':
                        # Max rule_confidence (RulE plausibility) among rules that
                        # fired for this query; 0 if no rule fired.
                        val = (float(top_conf_stash[k].item())
                               if top_conf_stash is not None else float('nan'))
                    elif name == 'kge_max':
                        val = kge_c.max().item() if kge_c.numel() else float('nan')
                    elif name == 'kge_entropy':
                        val = _entropy(kge_c)
                    else:
                        raise ValueError(f"Unknown feature: {name}")
                    feats[name].append(val)

    rr_rule_t = torch.tensor(rr_rule, dtype=torch.float64)
    rr_kge_t = torch.tensor(rr_kge, dtype=torch.float64)
    advantage = rr_rule_t - rr_kge_t
    if rr_mix:
        rr_mix_t = torch.stack(rr_mix, dim=0).to(torch.float64)   # [N, B]
    else:
        rr_mix_t = torch.zeros((0, beta_grid.numel()), dtype=torch.float64)
    feat_t = {name: torch.tensor(vals, dtype=torch.float64)
              for name, vals in feats.items()}
    return {
        'features': feat_t,
        'rr_rule': rr_rule_t,
        'rr_kge': rr_kge_t,
        'advantage': advantage,
        'rels': torch.tensor(rels, dtype=torch.long),
        'rr_mix': rr_mix_t,
        'beta_grid': beta_grid.to(torch.float64),
    }


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def print_bucket_table(name, feat, advantage, num_buckets):
    """Mean rule_advantage per quantile bucket of one feature."""
    print(f"\nPer-bucket advantage for '{name}' ({num_buckets} quantile bins):")
    header = f"  {'bucket':>22} {'n':>6} {'mean_feat':>10} {'advantage':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, idx in quantile_buckets(feat, num_buckets):
        print(f"  {label:>22} {idx.numel():>6d} {feat[idx].mean().item():>10.4f} "
              f"{advantage[idx].mean().item():>+10.4f}")


def mixing_ceilings(rr_mix, beta_grid, rels):
    """MRR ceilings from choosing the best convex beta per relation and per query."""
    N = rr_mix.size(0)
    weighted = 0.0
    for rid in torch.unique(rels).tolist():
        idx = (rels == rid).nonzero(as_tuple=True)[0]
        mrr_by_beta = rr_mix[idx].mean(dim=0)               # [B]
        weighted += idx.numel() * mrr_by_beta.max().item()
    return {
        'perrel_mix_mrr': (weighted / N) if N else float('nan'),
        'per_query_mix_mrr': rr_mix.max(dim=1).values.mean().item() if N else float('nan'),
    }


def print_oracle_ceiling(rr_rule, rr_kge, ceil):
    """Print the MRR ceiling at each mixing granularity (pure scorers, per-relation, per-query)."""
    mean_rule = rr_rule.mean().item()
    mean_kge = rr_kge.mean().item()
    print(f"\n{'-'*70}")
    print("ORACLE CEILING (max MRR available at each mixing granularity)")
    print(f"{'-'*70}")
    print(f"  MRR rule-only           : {mean_rule:.4f}")
    print(f"  MRR kge-only            : {mean_kge:.4f}")
    print(f"  Per-relation oracle     : {ceil['perrel_mix_mrr']:.4f}  "
          f"(best convex beta per relation)")
    print(f"  Per-query oracle        : {ceil['per_query_mix_mrr']:.4f}  "
          f"(best convex beta per query; ceiling for any adaptive head)")


def main():
    parser = argparse.ArgumentParser(
        description="Training-free feature-validity screen for adaptive beta.")
    parser.add_argument('--config', required=True, help='Path to model config json.')
    parser.add_argument('--checkpoint', required=True,
                        help='Grounding checkpoint (same one beta training uses).')
    parser.add_argument('--split', choices=['test', 'valid'], default='valid',
                        help='Split to screen on; use valid (test is for the final eval).')
    parser.add_argument(
        '--features',
        default=','.join(FEATURE_CHOICES),
        help='Comma-separated features. Available: ' + ', '.join(FEATURE_CHOICES) + '.')
    parser.add_argument('--num_buckets', type=int, default=5)
    parser.add_argument('--beta_grid_steps', type=int, default=21,
                        help='Number of betas in [0,1] for the convex mixing grid.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override dataset grouping batch size (1 for large graphs).')
    parser.add_argument('--output', default=None,
                        help='CSV output path for per-feature stats.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_names = [f.strip() for f in args.features.split(',') if f.strip()]
    unknown = [f for f in feature_names if f not in FEATURE_CHOICES]
    if unknown:
        parser.error(f"Unknown feature(s): {', '.join(unknown)}. "
                     f"Available: {', '.join(FEATURE_CHOICES)}.")

    print(f"Using device: {device}")
    print(f"Split: {args.split}")
    print(f"Features: {feature_names}")

    # ---- Load config / data / model / rules (mirrors train_beta_grounding) ----
    config = load_config(args.config)
    if isinstance(config, (list, tuple)):
        config = config[0]

    beta_grid = torch.linspace(0.0, 1.0, args.beta_grid_steps)
    print(f"Convex beta grid: {beta_grid.numel()} points in [0,1]")

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
    # Per-rule confidence (RulE plausibility) for 'top_rule_confidence'. Safe
    # no-op for model variants that lack it (the feature then reads as NaN).
    if hasattr(model, 'eval_compute_rule_confidence'):
        model.eval_compute_rule_confidence(device)

    batch_size = args.batch_size or getattr(config, 'g_batch_size', 8)
    if args.split == 'test':
        screen_set = TestDataset(graph, batch_size)
    else:
        screen_set = ValidDataset(graph, batch_size)
    dataloader = DataLoader(screen_set, batch_size=1, num_workers=0)

    print(f"\n{'='*70}")
    print(f"FEATURE-VALIDITY SCREEN | dataset={dataset_name} | split={args.split}")
    print(f"{'='*70}")

    data = collect(model, dataloader, device, feature_names, beta_grid)
    advantage = data['advantage']
    rr_rule = data['rr_rule']
    rr_kge = data['rr_kge']
    n = advantage.numel()

    # ---- Global anchors ----
    print(f"\nQueries (N)            : {n}")
    print(f"Mean rule_advantage    : {advantage.mean().item():+.4f}")
    print(f"Frac queries rule>kge  : {(advantage > 0).float().mean().item():.3f}")

    # ---- Oracle ceiling: pure scorers vs per-relation / per-query best mix ----
    ceil = mixing_ceilings(data['rr_mix'], data['beta_grid'], data['rels'])
    print_oracle_ceiling(rr_rule, rr_kge, ceil)

    # ---- Per-feature correlation table ----
    rows = []
    for name in feature_names:
        feat = data['features'][name]
        rep = correlation_report(feat, advantage)
        rows.append({
            'feature': name,
            'n': rep['n'],
            'r_adv': rep['r'],
            'ci_lo': rep['ci_lo'],
            'ci_hi': rep['ci_hi'],
            'r_RRrule': pearson_r(feat, rr_rule),
            'r_RRkge': pearson_r(feat, rr_kge),
        })

    rows.sort(key=lambda d: (abs(d['r_adv']) if math.isfinite(d['r_adv']) else -1),
              reverse=True)

    print(f"\n{'-'*70}")
    print(f"{'feature':>22} {'r_adv':>8} {'95% CI':>17} {'r:RRrule':>9} {'r:RRkge':>9}")
    print(f"{'-'*70}")
    for d in rows:
        ci = f"[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]"
        print(f"{d['feature']:>22} {d['r_adv']:>+8.3f} {ci:>17} "
              f"{d['r_RRrule']:>+9.3f} {d['r_RRkge']:>+9.3f}")
    print(f"{'-'*70}")
    print("Note: with large N even tiny correlations are 'significant'; judge "
          "usefulness by |r_adv| and whether the 95% CI excludes 0.")

    # ---- Per-bucket table for the single strongest feature ----
    if rows:
        top = rows[0]['feature']
        print_bucket_table(top, data['features'][top], advantage, args.num_buckets)

    # ---- CSV ----
    out_path = args.output or f"feature_validity_{dataset_name}_{args.split}.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for d in rows:
            writer.writerow(d)
    print(f"\nPer-feature stats written to: {out_path}")


if __name__ == '__main__':
    main()
