"""
Train beta parameter during grounding phase (after main model is trained).
This allows learning optimal KGE/Rule score balance per relation without retraining the full model.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RulE_original', 'src'))
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import KnowledgeGraph, TestDataset, ValidDataset, RuleDataset
from model import RulE
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--density_lr', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=3.0,
                       help='Fixed alpha for comparison baseline')
    # Adaptive-beta feature + score scaling -----------------------------------
    parser.add_argument('--feature', type=str, default='density',
                       choices=['density', 'num_rules'],
                       help="Per-query feature the adaptive head conditions on. "
                            "'density': fraction of candidate entities with >=1 "
                            "rule fired. 'num_rules': fraction of the relation's "
                            "rules that fired for the (h, r) query.")
    parser.add_argument('--normalize_scores', action='store_true',
                       help='Per-query (over entities) divide rule_logits and '
                            'kge_score by their std before mixing, so the convex '
                            'combination beta*rule+(1-beta)*kge is on a common '
                            'scale (beta=0.5 ~ balanced). Applied to ALL methods '
                            'uniformly (fixed-alpha, per-relation, adaptive).')
    parser.add_argument('--standardize_feature', action='store_true',
                       help='Z-score the adaptive feature using mean/std computed '
                            'once on the beta-training split, so beta[r] is the '
                            'log-odds at the average query and beta_density is '
                            'well-conditioned.')
    parser.add_argument('--no_per_relation', action='store_true',
                       help='Ablation: replace per-relation intercept beta[r] with a '
                            'single global scalar (global_beta). Stage 3 then trains '
                            'global_beta; stage 6 trains beta_density on top. Isolates '
                            'whether per-relation calibration is needed.')
    # Negative sampling for the margin loss -----------------------------------
    parser.add_argument('--neg_sampling', type=str, default='mixed',
                       choices=['uniform', 'hard', 'mixed'],
                       help="Which negatives enter the margin loss per query. "
                            "'uniform': K random negatives (legacy behaviour). "
                            "'hard': the K highest-scoring negatives under the "
                            "current mixed logits. 'mixed' (default): the hardest "
                            "round(K*mixed_hard_frac) negatives plus the remainder "
                            "drawn at random from the complement.")
    parser.add_argument('--num_negatives', type=int, default=100,
                       help='K: number of negatives sampled per query for the '
                            'margin loss. If a query has <=K negatives, all are '
                            'used. Default 100 matches prior runs.')
    parser.add_argument('--mixed_hard_frac', type=float, default=0.5,
                       help="Fraction of --num_negatives taken as hardest when "
                            "--neg_sampling=mixed; the rest are random from the "
                            "complement (no duplicates). Ignored otherwise.")
    # Reproducibility / persistence ------------------------------------------
    parser.add_argument('--seed', type=int, default=42,
                       help='Global RNG seed (torch / random / cuda) and seed '
                            'for the per-relation valid split. Default 42 '
                            'matches the previously-hardcoded behaviour.')
    parser.add_argument('--beta_checkpoint', type=str, default=None,
                       help='Path to load/save the trained beta + beta_density '
                            'tensors. If unset, defaults to <basecheckpoint_dir>/beta.pt. '
                            'After full training the tensors are saved here; '
                            'with --skip_beta_training, they are loaded from here.')
    parser.add_argument('--skip_beta_training', action='store_true',
                       help='Skip stages [2], [3], [5], [6]: load already-trained '
                            'beta and beta_density from --beta_checkpoint and run '
                            'only the test evaluations + post-eval analyses. '
                            'Useful for cheap iteration on diagnostics.')
    parser.add_argument('--beta_l2', type=float, default=0.0,
                       help='L2 regularization coefficient on model.beta (or global_beta). '
                            'Adds beta_l2 * mean(beta**2) to the per-batch loss, shrinking '
                            'per-relation betas toward 0 in logit space (sigmoid=0.5, '
                            'neutral routing). Counteracts overfitting on small-n relations. '
                            'Default 0.0 (disabled). Try 0.01, 0.1, 1.0.')
    return parser.parse_args()


def evaluate(model, dataloader, device, use_beta=False, adaptive_beta=False, alpha=3.0,
             return_per_query=False, normalize_scores=False, extra_ranks=False):
    """Evaluate the model.

    Ranks are computed per-batch and moved to CPU immediately so the GPU never
    holds more than one batch of [batch, num_entities] logits at a time. This
    is critical for large datasets such as WN18RR (40 943 entities).

    If `normalize_scores` is True, rule_logits and kge_score are each divided by
    their per-query std (over the entity axis) before mixing, so the convex
    combination is on a common scale. Rank-based quantities (corr, extra ranks)
    are computed BEFORE normalization since per-query scaling is order-preserving.

    If `return_per_query` is True, returns (metrics, per_query_dict). The dict
    keys mirror what the post-eval analyses need:
      - 'ranks'      : Tensor[N]  filtered, tie-averaged rank per query.
      - 'feature'    : Tensor[N]  raw adaptive feature per query (the signal
                                  compute_adaptive_beta conditions on).
      - 'relations'  : Tensor[N]  relation id per query (long).
      - 'top1'       : Tensor[N]  predicted entity id (argmax over flag-True
                                  candidates) under THIS method's mixed score.
      - 'true_tails' : Tensor[N]  ground-truth tail entity id per query.
      - 'corr'       : Tensor[N]  per-query Spearman correlation between the
                                  pre-mix rule score and the kge_score over all
                                  candidate entities. Method-invariant.
    If `extra_ranks` is True it additionally returns:
      - 'rank_rule'  : Tensor[N]  true-tail rank under rule-only scores (beta=1).
      - 'rank_kge'   : Tensor[N]  true-tail rank under kge-only scores (beta=0).
    All tensors are on CPU and aligned by index. Memory: per-batch transients
    are freed by the existing `empty_cache()` call; only O(num_queries) floats
    accumulate across the eval loop.
    """
    model.eval()
    all_ranks = []
    all_feature   = [] if return_per_query else None
    all_corr      = [] if return_per_query else None
    all_relations = [] if return_per_query else None
    all_top1      = [] if return_per_query else None
    all_true      = [] if return_per_query else None
    all_rank_rule = [] if extra_ranks else None
    all_rank_kge  = [] if extra_ranks else None

    with torch.no_grad():
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)

            # Forward pass (populates model._last_ground_mask / _last_num_rules).
            # Keep `rule_logits` distinct from the mixed `logits` -- we need it
            # below for the rule/KGE rank correlation.
            rule_logits, mask = model(all_h, all_r, None)

            # Read the per-query adaptive feature (density or num_rules) the
            # same way compute_adaptive_beta does, so bucket edges line up.
            if return_per_query:
                all_feature.append(model.get_query_feature().detach().cpu())

            # Combine with KGE
            kge_score = model.compute_g_KGE(all_h, all_r)

            # Per-query Spearman corr between rule and KGE rankings (Pearson on
            # rank vectors via argsort().argsort()). Computed once per batch on
            # GPU, reduced to one float per query, then moved to CPU. Frees its
            # temporaries before mixing so peak GPU memory does not grow.
            if return_per_query:
                ra = rule_logits.argsort(dim=-1).argsort(dim=-1).float()
                rb = kge_score.argsort(dim=-1).argsort(dim=-1).float()
                a_c = ra - ra.mean(dim=-1, keepdim=True)
                b_c = rb - rb.mean(dim=-1, keepdim=True)
                num = (a_c * b_c).sum(dim=-1)
                denom = torch.sqrt(
                    (a_c ** 2).sum(dim=-1) * (b_c ** 2).sum(dim=-1)
                ).clamp_min(1e-12)
                all_corr.append((num / denom).detach().cpu())
                del ra, rb, a_c, b_c, num, denom

            # Rule-only / KGE-only true-tail ranks (for the feature-validity
            # diagnostic). Captured BEFORE normalization since per-query scaling
            # does not change the ranking.
            rule_cpu = rule_logits.cpu() if extra_ranks else None
            kge_cpu  = kge_score.cpu()   if extra_ranks else None

            # Optional per-query score normalization (common scale before mix).
            if normalize_scores:
                rule_logits = rule_logits / (rule_logits.std(dim=-1, keepdim=True) + 1e-6)
                kge_score   = kge_score   / (kge_score.std(dim=-1, keepdim=True) + 1e-6)

            if adaptive_beta:
                beta, _ = model.compute_adaptive_beta(rule_logits, all_r)
                logits = beta * rule_logits + (1 - beta) * kge_score
            elif use_beta:
                if model.use_per_relation:
                    beta = torch.sigmoid(model.beta[all_r[0]]).unsqueeze(-1)
                else:
                    beta = torch.sigmoid(model.global_beta).unsqueeze(-1)
                logits = beta * rule_logits + (1 - beta) * kge_score
            else:
                logits = rule_logits + alpha * kge_score

            # Move to CPU immediately — avoids accumulating GB of GPU tensors
            logits_cpu = logits.cpu()
            flag_cpu = flag.cpu()
            mask_cpu = mask.cpu()
            all_t_cpu = all_t.cpu()
            all_r_cpu = all_r.cpu() if return_per_query else None
            del logits, kge_score, rule_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for k in range(all_t_cpu.size(0)):
                t = all_t_cpu[k]
                if mask_cpu[k, t].item():
                    val = logits_cpu[k, t]
                    L = (logits_cpu[k][flag_cpu[k]] > val).sum().item() + 1
                    H = (logits_cpu[k][flag_cpu[k]] >= val).sum().item() + 2
                else:
                    L = 1
                    H = flag_cpu.size(1) + 1
                all_ranks.append((L + H - 1) / 2.0)

                if return_per_query:
                    # argmax over flag-True candidates, mapped back to global
                    # entity id. Done per query inside the existing CPU loop.
                    cand_idx = flag_cpu[k].nonzero(as_tuple=True)[0]
                    cand_scores = logits_cpu[k][cand_idx]
                    top1_pos = cand_scores.argmax().item()
                    all_top1.append(int(cand_idx[top1_pos].item()))

                if extra_ranks:
                    all_rank_rule.append(_filtered_rank(rule_cpu[k], flag_cpu[k], mask_cpu[k], t))
                    all_rank_kge.append(_filtered_rank(kge_cpu[k], flag_cpu[k], mask_cpu[k], t))

            if return_per_query:
                all_relations.append(all_r_cpu)
                all_true.append(all_t_cpu)

    ranks = torch.tensor(all_ranks, dtype=torch.float)
    metrics = {
        'Hit@1': (ranks <= 1).float().mean().item(),
        'Hit@3': (ranks <= 3).float().mean().item(),
        'Hit@10': (ranks <= 10).float().mean().item(),
        'MR': ranks.mean().item(),
        'MRR': (1.0 / ranks).mean().item()
    }

    if return_per_query:
        pq = {
            'ranks':      ranks,
            'feature':    torch.cat(all_feature, dim=0),
            'relations':  torch.cat(all_relations, dim=0).long(),
            'top1':       torch.tensor(all_top1, dtype=torch.long),
            'true_tails': torch.cat(all_true, dim=0).long(),
            'corr':       torch.cat(all_corr, dim=0),
        }
        if extra_ranks:
            pq['rank_rule'] = torch.tensor(all_rank_rule, dtype=torch.float)
            pq['rank_kge'] = torch.tensor(all_rank_kge, dtype=torch.float)
        return metrics, pq
    return metrics


def select_negative_indices(neg_scores_detached, scheme, num_neg, hard_frac):
    """Pick which negatives enter the margin loss for one query.

    Ranks by `neg_scores_detached` (the current mixed logits, detached, so the
    selection is standard hard-negative mining w.r.t. the current model). The
    caller gathers the *non-detached* scores at the returned indices so the
    margin loss keeps grad through beta.

    Returns a 1-D LongTensor of indices into the negative pool.
    """
    n = neg_scores_detached.size(0)
    device = neg_scores_detached.device
    if n <= num_neg:
        return torch.arange(n, device=device)
    if scheme == 'uniform':
        return torch.randperm(n, device=device)[:num_neg]
    if scheme == 'hard':
        return torch.topk(neg_scores_detached, num_neg).indices
    # mixed: hardest n_hard + random from the rest (no duplicates).
    n_hard = min(int(round(num_neg * hard_frac)), n)
    hard_idx = torch.topk(neg_scores_detached, n_hard).indices
    keep = torch.ones(n, dtype=torch.bool, device=device)
    keep[hard_idx] = False
    rest = keep.nonzero(as_tuple=True)[0]
    n_rand = num_neg - n_hard
    rand_idx = rest[torch.randperm(rest.size(0), device=device)[:n_rand]]
    return torch.cat([hard_idx, rand_idx])


def train_beta_epoch(model, dataloader, optimizer, device, adaptive=False,
                     normalize_scores=False, neg_sampling='mixed',
                     num_negatives=100, mixed_hard_frac=0.5, beta_l2=0.0):
    """Train beta (and beta_density if adaptive) for one epoch.

    Memory strategy for large datasets (e.g. WN18RR, 40 943 entities):
    - rule_logits and kge_score are computed once under no_grad and kept on GPU
      only for the duration of one dataloader batch.
    - beta is applied per-sample (slice [i:i+1]) so each backward graph covers
      only one query rather than the whole batch, keeping peak memory low.
    - Each per-sample loss is divided by N_valid before .backward(), so the
      accumulated grad equals the batch *mean* (not sum). This matches the
      non-chunked version and keeps the effective lr batch-size-invariant.
    - optimizer.step() is called once per dataloader batch.
    """
    model.train()

    # Explicit freezing tied to the training stage. Avoids the previous
    # 'beta in name' substring match, which silently re-enabled grad on
    # `model.beta` during stage 6 even though only `beta_density` was meant
    # to train.
    for name, param in model.named_parameters():
        if name == 'beta':
            param.requires_grad = (not adaptive) and model.use_per_relation
        elif name == 'global_beta':
            param.requires_grad = (not adaptive) and (not model.use_per_relation)
        elif name == 'beta_density':
            param.requires_grad = adaptive
        else:
            param.requires_grad = False

    total_loss = 0.0
    num_updates = 0

    for batch in tqdm(dataloader, desc="Training beta"):
        all_h, all_r, all_t, flag = batch
        all_h = all_h.squeeze(0).to(device)
        all_r = all_r.squeeze(0).to(device)
        all_t = all_t.squeeze(0).to(device)
        flag = flag.squeeze(0).to(device)

        # Compute frozen activations once for the whole batch
        with torch.no_grad():
            rule_logits, mask = model(all_h, all_r, None)
            kge_score = model.compute_g_KGE(all_h, all_r)
            # Per-query score normalization (same transform as evaluate) so the
            # convex combination trains on a common scale. The adaptive feature
            # is read from _last_ground_mask / _last_num_rules, not from these
            # tensors, so normalizing here does not touch the feature.
            if normalize_scores:
                rule_logits = rule_logits / (rule_logits.std(dim=-1, keepdim=True) + 1e-6)
                kge_score   = kge_score   / (kge_score.std(dim=-1, keepdim=True) + 1e-6)

        # Pre-pass: find indices of samples that contribute to the loss.
        # A sample contributes iff (a) its true tail is unmasked and
        # (b) there is at least one usable negative. We need N_valid up-front
        # to scale each per-sample loss by 1/N_valid → mean gradient.
        valid_indices = []
        for i in range(all_h.size(0)):
            if not mask[i, all_t[i]].item():
                continue
            negative_mask_i = flag[i] & (torch.arange(rule_logits.size(1), device=device) != all_t[i])
            if negative_mask_i.sum().item() == 0:
                continue
            valid_indices.append(i)

        if not valid_indices:
            del rule_logits, kge_score
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        N = len(valid_indices)
        optimizer.zero_grad()

        batch_loss_sum = 0.0  # sum of per-sample mean losses, divided by N -> batch mean
        logit_i = beta_i = neg_scores = true_score = loss = None  # for the final del
        # Stash the batch-wide feature tensors so we can temporarily expose only
        # row i to compute_adaptive_beta (which reads them internally). Without
        # this slicing, compute_adaptive_beta would use the WHOLE batch's
        # feature and the [0] indexing below would reuse the first query's
        # feature for every sample i -> wrong per-query density/num_rules.
        saved_ground_mask = model._last_ground_mask
        saved_num_rules = model._last_num_rules

        for i in valid_indices:
            r_i = all_r[i:i+1]
            rl_i = rule_logits[i:i+1]   # [1, E] — detached from frozen no_grad graph
            kg_i = kge_score[i:i+1]     # [1, E]

            if adaptive:
                # Expose only this sample's feature, compute beta, then restore.
                model._last_ground_mask = (
                    saved_ground_mask[i:i+1] if saved_ground_mask is not None else None
                )
                model._last_num_rules = (
                    saved_num_rules[i:i+1] if saved_num_rules is not None else None
                )
                beta_i, _ = model.compute_adaptive_beta(rl_i, r_i)
                model._last_ground_mask = saved_ground_mask
                model._last_num_rules = saved_num_rules
            elif model.use_per_relation:
                beta_i = torch.sigmoid(model.beta[r_i[0]]).unsqueeze(-1)
            else:
                beta_i = torch.sigmoid(model.global_beta).unsqueeze(-1)

            logit_i = beta_i * rl_i + (1 - beta_i) * kg_i   # [1, E], grad through beta only

            negative_mask = flag[i] & (torch.arange(logit_i.size(1), device=device) != all_t[i])
            neg_all = logit_i[0][negative_mask]
            # Select which negatives enter the margin loss. Rank by the detached
            # scores (hard-negative mining), gather the live scores so grad still
            # flows through beta.
            idx = select_negative_indices(
                neg_all.detach(), neg_sampling, num_negatives, mixed_hard_frac,
            )
            neg_scores = neg_all[idx]

            true_score = logit_i[0, all_t[i]]
            # Per-sample mean over negatives, then divide by N so the per-sample
            # gradients add up to the batch mean (equivalent to stacking and
            # calling .mean().backward() in the non-chunked version, but with
            # per-sample memory).
            loss = torch.clamp(1.0 - true_score + neg_scores, min=0).mean() / N
            loss.backward()
            batch_loss_sum += loss.item()

        # L2 regularization on the intercept parameter (stage 3 only; during
        # stage 6 beta/global_beta has requires_grad=False so this is a no-op).
        # Penalty shrinks logits toward 0 => sigmoid toward 0.5 (neutral routing).
        # Applied once per batch, after the per-sample accumulation, so its
        # gradient is not divided by N (it is a batch-level term, not per-sample).
        if beta_l2 > 0.0:
            if model.use_per_relation and model.beta.requires_grad:
                (beta_l2 * (model.beta ** 2).mean()).backward()
            elif not model.use_per_relation and model.global_beta.requires_grad:
                (beta_l2 * (model.global_beta ** 2)).backward()

        optimizer.step()

        # batch_loss_sum = (1/N) * sum_i loss_i  -> the actual batch mean loss
        total_loss += batch_loss_sum
        num_updates += 1

        del logit_i, beta_i, neg_scores, true_score, loss
        del rule_logits, kge_score
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_loss / max(num_updates, 1)


def print_results(name, metrics):
    print(f"\n{'='*60}")
    print(f"Results for: {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def _filtered_rank(scores_row, flag_row, mask_row, t):
    """Tie-averaged, filtered rank of true tail t under a 1-D score vector.

    Mirrors the inline rank logic used for the mixed logits, factored out so the
    rule-only / KGE-only ranks (feature-validity diagnostic) use the same rule.
    Works on CPU or GPU tensors.
    """
    if mask_row[t].item():
        val = scores_row[t]
        L = (scores_row[flag_row] > val).sum().item() + 1
        H = (scores_row[flag_row] >= val).sum().item() + 2
    else:
        L = 1
        H = flag_row.numel() + 1
    return (L + H - 1) / 2.0


def compute_feature_stats(model, dataloader, device):
    """Compute mean/std of the model's adaptive feature on the given split.

    One forward pass over `dataloader` (no grad), collecting model.get_query_feature()
    per query, then store mean/std into model.feature_mean / model.feature_std so
    compute_adaptive_beta can z-score the feature. Run on the beta-TRAIN split
    only -- never on test -- so standardization does not peek at evaluation data.
    """
    model.eval()
    feats = []
    with torch.no_grad():
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            model(all_h, all_r, None)
            feats.append(model.get_query_feature().detach().cpu())
    feats = torch.cat(feats, dim=0).float()
    mean = feats.mean()
    std = feats.std().clamp_min(1e-6)
    model.feature_mean.data = mean.view(1).to(model.feature_mean.device)
    model.feature_std.data = std.view(1).to(model.feature_std.device)
    print(f"Feature '{model.feature_name}' stats (beta-train split): "
          f"mean={mean.item():.4f}  std={std.item():.4f}  "
          f"(n={feats.numel()})")


def print_bucket_analysis(method_to_ranks, densities, num_buckets=5,
                          feature_name='density', title=None):
    """Print per-bucket MRR / Hit@1 / Hit@10 / MR across combination methods.

    Layout is one row per (bucket, method): for each density bucket we emit
    one block of method rows, with MRR, Hit@1, Hit@10, MR as columns. This
    lets the reader see *which* aspect of the ranking moved (e.g. MR can change
    while MRR is stationary).

    Memory: pure CPU bookkeeping over O(num_queries) floats. Safe for WN18RR.
    """
    if title is None:
        title = f"PER-BUCKET ANALYSIS (test set, bucketed by feature='{feature_name}')"
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

    if isinstance(densities, torch.Tensor):
        d = densities.detach().cpu().float()
    else:
        d = torch.tensor(densities, dtype=torch.float)
    n_total = d.numel()

    if n_total == 0:
        print("  (no queries to bucket)")
        return

    method_names = list(method_to_ranks.keys())
    name_w = max(len(m) for m in method_names) + 2

    header = (
        f"{'Bucket':<26} {'n':>6}  "
        f"{'Method':<{name_w}}"
        f"{'MRR':>10}  {'Hit@1':>10}  {'Hit@10':>10}  {'MR':>10}"
    )
    print(header)
    print("-" * len(header))

    def _row_block(bucket_label, mask):
        n = int(mask.sum().item())
        if n == 0:
            return
        for i, name in enumerate(method_names):
            r = method_to_ranks[name][mask].float()
            mrr = (1.0 / r).mean().item()
            h1 = (r <= 1).float().mean().item()
            h10 = (r <= 10).float().mean().item()
            mr = r.mean().item()
            if i == 0:
                prefix = f"{bucket_label:<26} {n:>6}  "
            else:
                prefix = f"{'':<26} {'':>6}  "
            print(
                f"{prefix}{name:<{name_w}}"
                f"{mrr:>10.4f}  {h1:>10.4f}  {h10:>10.4f}  {mr:>10.2f}"
            )

    # Isolate zero-density queries as their own bucket: rule signal collapses
    # to the bias term, distinct regime from low-but-positive density.
    zero_mask = d <= 1e-9
    nonzero_mask = ~zero_mask

    if zero_mask.any():
        _row_block("zero (no rule signal)", zero_mask)

    if nonzero_mask.any():
        d_nz = d[nonzero_mask]
        probs = torch.linspace(0.0, 1.0, num_buckets + 1)
        cuts = torch.quantile(d_nz, probs).tolist()
        for b in range(num_buckets):
            lo, hi = cuts[b], cuts[b + 1]
            if b < num_buckets - 1:
                bucket_mask = nonzero_mask & (d >= lo) & (d < hi)
            else:
                bucket_mask = nonzero_mask & (d >= lo) & (d <= hi)
            label = f"Q{b+1} [{lo:.4f}, {hi:.4f}]"
            _row_block(label, bucket_mask)

    print(f"\n  Total queries: {n_total}   {feature_name} range: [{d.min().item():.4f}, {d.max().item():.4f}]")


def print_per_relation_table(method_to_ranks, relations, model, densities,
                             fixed_method_name, adaptive_method_name, top_n=None):
    """Per-relation breakdown linking learned beta[r] to per-relation MRR.

    For each relation that appears in the test set, prints:
      n            number of test queries with this relation,
      beta[r]      sigmoid(model.beta[r]) -- the learned per-relation rule weight,
      dens_mean    mean grounding density of test queries for this relation,
      one column per method: per-relation MRR,
      dMRR(A-F)    MRR(adaptive_method_name) - MRR(fixed_method_name).

    Rows are sorted by |beta[r] - 0.5| descending so the most extreme routings
    are at the top -- that's where any per-relation effect should show up first.
    """
    print(f"\n{'='*80}")
    print("PER-RELATION TABLE (sorted by |beta[r] - 0.5| descending)")
    print(f"{'='*80}")

    relations = relations.cpu().long()
    densities = densities.cpu().float()

    with torch.no_grad():
        beta_sigmoid = torch.sigmoid(model.beta).detach().cpu()

    method_names = list(method_to_ranks.keys())

    rows = []
    for r in sorted(set(relations.tolist())):
        rel_mask = (relations == r)
        n = int(rel_mask.sum().item())
        if n == 0:
            continue
        beta_r = beta_sigmoid[r].item() if r < beta_sigmoid.numel() else float('nan')
        dens_mean = densities[rel_mask].mean().item()
        per_method = {}
        for name in method_names:
            ranks_r = method_to_ranks[name][rel_mask].float()
            per_method[name] = (1.0 / ranks_r).mean().item()
        d_mrr = per_method[adaptive_method_name] - per_method[fixed_method_name]
        rows.append({
            'r': r, 'n': n, 'beta': beta_r, 'dens_mean': dens_mean,
            'per_method': per_method, 'd_mrr': d_mrr,
        })

    rows.sort(key=lambda x: abs(x['beta'] - 0.5), reverse=True)
    if top_n is not None:
        rows = rows[:top_n]

    method_header = "  ".join(f"{m:>11}" for m in method_names)
    header = (
        f"{'r':>4}  {'n':>6}  {'beta[r]':>9}  {'feat_mean':>10}  "
        f"{method_header}  {'dMRR(A-F)':>11}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        cells = "  ".join(f"{row['per_method'][m]:>11.4f}" for m in method_names)
        print(
            f"{row['r']:>4}  {row['n']:>6}  {row['beta']:>9.4f}  {row['dens_mean']:>10.4f}  "
            f"{cells}  {row['d_mrr']:>+11.4f}"
        )


def print_feature_validity(feature, rank_rule, rank_kge, feature_name='density',
                           num_buckets=5):
    """Does the per-query feature predict when rules beat KGE?

    For each query we compare the true-tail reciprocal rank under rule-only
    (beta=1) vs KGE-only (beta=0):
        rule_advantage = RR_rule - RR_kge   (> 0  => rules rank truth higher).
    A positive correlation between the feature and rule_advantage means the
    feature is a *valid router*: high feature => rules are the better branch =>
    "trust rules more when the feature is high" is justified. This replaces the
    decision-flip table, which only said whether predictions changed, not
    whether the feature explains where rules should be trusted.

    Args:
        feature:   Tensor[N] raw per-query feature (density or num_rules).
        rank_rule: Tensor[N] true-tail rank under rule-only scores.
        rank_kge:  Tensor[N] true-tail rank under kge-only scores.
    """
    print(f"\n{'='*80}")
    print(f"FEATURE-VALIDITY DIAGNOSTIC (feature='{feature_name}')")
    print(f"{'='*80}")

    f = feature.detach().cpu().float()
    rr = 1.0 / rank_rule.detach().cpu().float()
    rk = 1.0 / rank_kge.detach().cpu().float()
    adv = rr - rk

    fc = f - f.mean()
    ac = adv - adv.mean()
    denom = torch.sqrt((fc ** 2).sum() * (ac ** 2).sum()).clamp_min(1e-12)
    pearson = (fc * ac).sum() / denom

    print(f"  Pearson corr(feature, rule_advantage): {pearson.item():+.4f}")
    print(f"  Mean rule_advantage (RR_rule - RR_kge): {adv.mean().item():+.4f}")
    print(f"  Frac queries rules strictly better:     {(adv > 0).float().mean().item():.4f}")
    print(f"  RR_rule mean: {rr.mean().item():.4f}   RR_kge mean: {rk.mean().item():.4f}\n")

    print(f"  Per-{feature_name}-bucket rule advantage:")
    bh = (f"  {'Bucket':<26} {'n':>6}  {'mean feat':>10}  "
          f"{'RR_rule':>9}  {'RR_kge':>9}  {'adv':>9}  {'rules>kge':>9}")
    print(bh)
    print("  " + "-" * (len(bh) - 2))

    def _bucket_row(label, mask):
        nn_ = int(mask.sum().item())
        if nn_ == 0:
            return
        print(f"  {label:<26} {nn_:>6}  {f[mask].mean().item():>10.4f}  "
              f"{rr[mask].mean().item():>9.4f}  {rk[mask].mean().item():>9.4f}  "
              f"{adv[mask].mean().item():>+9.4f}  {(adv[mask] > 0).float().mean().item():>9.4f}")

    zero_mask = f <= 1e-9
    nonzero_mask = ~zero_mask
    if zero_mask.any():
        _bucket_row("zero (no rule signal)", zero_mask)
    if nonzero_mask.any():
        f_nz = f[nonzero_mask]
        probs = torch.linspace(0.0, 1.0, num_buckets + 1)
        cuts = torch.quantile(f_nz, probs).tolist()
        for b in range(num_buckets):
            lo, hi = cuts[b], cuts[b + 1]
            if b < num_buckets - 1:
                bm = nonzero_mask & (f >= lo) & (f < hi)
            else:
                bm = nonzero_mask & (f >= lo) & (f <= hi)
            _bucket_row(f"Q{b+1} [{lo:.4f}, {hi:.4f}]", bm)

    print()
    print("  Reading: corr > 0 => higher feature picks out queries where rules")
    print("           outrank KGE, so 'trust rules more when feature is high' is justified.")
    print("           corr ~ 0 => feature does not identify where rules help (weak router).")


def print_headroom_diagnostic(corrs, densities, num_buckets=5, feature_name='density'):
    """Mean Spearman correlation between rule and KGE rankings, plus per-bucket means.

    Interpretation: if rule and KGE rank candidates near-identically (corr ~ 1),
    *any* convex combination produces the same ranking, and no mixing strategy
    -- including adaptive beta -- can move the metrics. This number upper-bounds
    the headroom for routing.
    """
    print(f"\n{'='*80}")
    print("HEADROOM DIAGNOSTIC (Spearman corr between rule and KGE rankings)")
    print(f"{'='*80}")

    c = corrs.detach().cpu().float()
    d = densities.detach().cpu().float()

    valid = ~torch.isnan(c)
    if valid.sum() == 0:
        print("  (no valid correlation values)")
        return

    cv = c[valid]
    print(f"  Mean correlation:   {cv.mean().item():+.4f}")
    print(f"  Median correlation: {cv.median().item():+.4f}")
    print(f"  Std:                {cv.std().item():.4f}")
    print(f"  Min/Max:            [{cv.min().item():+.4f}, {cv.max().item():+.4f}]\n")

    print(f"  Per-{feature_name}-bucket mean correlation:")
    bh = f"  {'Bucket':<26} {'n':>6}  {'mean corr':>11}"
    print(bh)
    print("  " + "-" * (len(bh) - 2))

    def _bucket_row(label, mask):
        m = mask & valid
        n = int(m.sum().item())
        if n == 0:
            return
        print(f"  {label:<26} {n:>6}  {c[m].mean().item():>+11.4f}")

    zero_mask = d <= 1e-9
    nonzero_mask = ~zero_mask

    if zero_mask.any():
        _bucket_row("zero (no rule signal)", zero_mask)

    if nonzero_mask.any():
        d_nz = d[nonzero_mask]
        probs = torch.linspace(0.0, 1.0, num_buckets + 1)
        cuts = torch.quantile(d_nz, probs).tolist()
        for b in range(num_buckets):
            lo, hi = cuts[b], cuts[b + 1]
            if b < num_buckets - 1:
                bm = nonzero_mask & (d >= lo) & (d < hi)
            else:
                bm = nonzero_mask & (d >= lo) & (d <= hi)
            _bucket_row(f"Q{b+1} [{lo:.4f}, {hi:.4f}]", bm)

    print()
    print("  Reading: corr ~ +1.0  =>  rule and KGE rank candidates near-identically;")
    print("                            mixing has little headroom (any beta produces ~same ranking).")
    print("           corr ~  0.0  =>  rankings independent; mixing has maximum potential.")
    print("           corr ~ -1.0  =>  rankings opposite; mixing can swing strongly either way.")


def print_beta_stats(model):
    """Print statistics about learned beta values."""
    with torch.no_grad():
        if model.use_per_relation:
            beta_values = torch.sigmoid(model.beta).cpu().numpy()
            print(f"\nBeta Statistics (per-relation intercept):")
            print(f"  Mean:   {beta_values.mean():.4f}")
            print(f"  Std:    {beta_values.std():.4f}")
            print(f"  Min:    {beta_values.min():.4f}")
            print(f"  Max:    {beta_values.max():.4f}")
            print(f"  Median: {float(torch.median(torch.tensor(beta_values))):.4f}")
        else:
            print(f"\nBeta Statistics (global intercept, no per-relation):")
            print(f"  global_beta (logit): {model.global_beta.item():.4f}")
            print(f"  global_beta (prob):  {torch.sigmoid(model.global_beta).item():.4f}")
        bd = model.beta_density.item()
        fn = model.feature_name
        print(f"  Feature slope (beta_density): {bd:.4f}  (feature='{fn}')")
        if bd > 0:
            print(f"    -> Positive: higher {fn} pushes beta UP (trust rules more)")
        elif bd < 0:
            print(f"    -> Negative: higher {fn} pushes beta DOWN (trust KGE more)")
        else:
            print(f"    -> Zero: {fn} has no effect on mixing")


def save_beta_checkpoint(path, model, **meta):
    """Persist the trained beta tensors + metadata to a small .pt file.

    The file contains:
      - 'beta':         CPU clone of model.beta (per-relation, shape [2*R]).
      - 'beta_density': CPU clone of model.beta_density (scalar).
      - any keyword args passed in (e.g. seed, stage, base_checkpoint,
        best_mrr_*, epoch_best_*).

    Tiny -- a few hundred floats. Safe to call multiple times during a run
    (e.g. after stage 3 and again after stage 6) so partial progress is
    preserved if a job is interrupted.
    """
    state = {
        'beta':             model.beta.data.detach().cpu().clone(),
        'beta_density':     model.beta_density.data.detach().cpu().clone(),
        'global_beta':      model.global_beta.data.detach().cpu().clone(),
        'use_per_relation': model.use_per_relation,
        **meta,
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(state, path)
    print(f"  Saved beta checkpoint -> {path}  (stage={meta.get('stage', '?')})")


def split_facts_per_relation(facts, train_ratio=0.5, seed=42):
    """Per-relation stratified split of a (h, r, t) fact list.

    Guarantees every relation that appears in `facts` is present in both halves
    (except for relations with a single fact, which go to the train half).
    Returns (train_facts, select_facts).
    """
    rng = random.Random(seed)
    by_rel = {}
    for fact in facts:
        by_rel.setdefault(fact[1], []).append(fact)
    train_facts, select_facts = [], []
    for _, items in by_rel.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            train_facts.append(items[0])
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_train = min(n_train, n - 1)
        train_facts.extend(items[:n_train])
        select_facts.extend(items[n_train:])
    return train_facts, select_facts


class FactListValidDataset(Dataset):
    """Variant of ValidDataset that takes an explicit fact list.

    Used to feed the per-relation train/select halves of valid_facts to the
    beta-training loop without modifying the original ValidDataset class.
    Behavior (batching, filter mask via hr2ooo) mirrors ValidDataset/TestDataset.
    """

    def __init__(self, graph, batch_size, facts):
        self.graph = graph
        self.batch_size = batch_size
        r2instances = [[] for _ in range(self.graph.relation_size * 2)]
        for h, r, t in facts:
            r2instances[r].append((h, r, t))
        self.batches = []
        for _, instances in enumerate(r2instances):
            random.shuffle(instances)
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]
        all_h = torch.LongTensor([t[0] for t in data])
        all_r = torch.LongTensor([t[1] for t in data])
        all_t = torch.LongTensor([t[2] for t in data])
        mask = torch.ones(len(data), self.graph.entity_size).bool()
        for k, (h, r, _t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2ooo[hr_index])
            mask[k][t_index] = 0
        return all_h, all_r, all_t, mask


def main():
    args = parse_args()

    # --- Reproducibility -----------------------------------------------------
    # Seed the three RNG sources that affect this script:
    #   - random            : used by FactListValidDataset and split_facts_per_relation.
    #   - torch             : used by torch.randperm in train_beta_epoch's
    #                         negative subsampling.
    #   - torch.cuda        : kernels that use the device RNG.
    # Without this, every run produced slightly different `beta` / `beta_density`
    # tensors and the trained values were not reconstructible from logs.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Adaptive feature: {args.feature}")
    print(f"Normalize rule/KGE scores: {args.normalize_scores}")
    print(f"Standardize feature: {args.standardize_feature}")
    print(f"Use per-relation intercept: {not args.no_per_relation}")
    print(f"Negative sampling: {args.neg_sampling} (K={args.num_negatives}, hard_frac={args.mixed_hard_frac})")
    
    # Load config
    config = load_config(args.config)
    if isinstance(config, (list, tuple)):
        config = config[0]
    
    # Load data
    print("Loading data...")
    data_path = config.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(args.config), data_path)
    data_path = os.path.normpath(data_path)
    
    graph = KnowledgeGraph(data_path)
    print("Data loading | DONE!")
    
    # Load model
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
        feature_name=args.feature,
        use_per_relation=not args.no_per_relation,
    )
    
    # Load rules
    rule_file = config.rule_file
    if not os.path.isabs(rule_file):
        rule_file = os.path.join(os.path.dirname(args.config), rule_file)
    rule_file = os.path.normpath(rule_file)
    
    rule_negative_size = getattr(config, 'rule_negative_size', 32)
    ruleset = RuleDataset(graph.relation_size, rule_file, rule_negative_size)
    rules = [rule[0] for rule in ruleset.rules]
    model.set_rules(rules)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    result = model.load_state_dict(state_dict, strict=False)
    print("Missing keys (in model, not in checkpoint):", result.missing_keys)
    print("Unexpected keys (in checkpoint, not in model):", result.unexpected_keys)
    model = model.to(device)
    model.eval_compute_rule_weight(device)
    
    # Create dataloaders.
    # Use batch_size=1 regardless of config's g_batch_size: compute_g_KGE
    # materializes a [batch, num_entities, hidden_dim*2] tail tensor on the GPU.
    # For WN18RR (40 943 entities, hidden_dim=500) that is
    #   g_batch_size=32 → [32, 40 943, 1000] ≈ 5.2 GB  (OOM)
    #   g_batch_size=1  → [1,  40 943, 1000] ≈ 164 MB  (safe)
    #
    # The valid set is split 50/50 per-relation: one half is used to TRAIN beta
    # (margin loss), the other half is used to SELECT the best epoch (early
    # stopping). This removes the bias of training and selecting on the same
    # data. The test set is untouched.
    beta_batch_size = 4
    train_facts, select_facts = split_facts_per_relation(
        graph.valid_facts, train_ratio=0.5, seed=args.seed
    )
    print(f"Valid split for beta training: "
          f"{len(train_facts)} train / {len(select_facts)} select "
          f"(from {len(graph.valid_facts)} total valid facts)")

    # Resolve the beta-checkpoint path. Default: alongside the base checkpoint,
    # named 'beta.pt'. This co-locates trained-mixing tensors with the base
    # model checkpoint they were trained on top of.
    if args.beta_checkpoint:
        beta_ckpt_path = args.beta_checkpoint
    else:
        beta_ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(args.checkpoint)) or '.',
            'beta.pt',
        )
    print(f"Beta checkpoint path: {beta_ckpt_path}")
    if args.skip_beta_training and not os.path.exists(beta_ckpt_path):
        raise FileNotFoundError(
            f"--skip_beta_training requires {beta_ckpt_path} to exist. "
            f"Run without --skip_beta_training first to train and save the betas."
        )

    valid_train_set = FactListValidDataset(graph, beta_batch_size, train_facts)
    valid_train_loader = DataLoader(valid_train_set, batch_size=1, num_workers=0)

    valid_select_set = FactListValidDataset(graph, beta_batch_size, select_facts)
    valid_select_loader = DataLoader(valid_select_set, batch_size=1, num_workers=0)

    test_set = TestDataset(graph, beta_batch_size)
    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0)

    # Feature standardization: compute mean/std of the adaptive feature on the
    # beta-TRAIN split only (never test), so compute_adaptive_beta can z-score it.
    # When skipping training the stats are restored from the beta checkpoint below.
    model.standardize_feature = args.standardize_feature
    if args.standardize_feature and not args.skip_beta_training:
        compute_feature_stats(model, valid_train_loader, device)

    print(f"\n{'='*60}")
    print("BETA TRAINING DURING GROUNDING PHASE")
    print(f"{'='*60}")
    
    # === Baseline: Fixed Alpha ===
    # Capture per-query data here (and at the two later test evals) so we can
    # run the post-eval analyses at the end. The dict contains:
    #   ranks, feature, relations, top1, true_tails, corr (+ rank_rule/rank_kge).
    # Of these, feature/relations/true_tails/corr (+ extra ranks) are method-
    # invariant -- we read them off the Fixed-alpha call only. top1 and ranks
    # are method-specific.
    print(f"\n[1] Testing with FIXED ALPHA = {args.alpha} (baseline)...")
    metrics_fixed, pq_fixed = evaluate(
        model, test_dataloader, device, use_beta=False, alpha=args.alpha,
        return_per_query=True, normalize_scores=args.normalize_scores,
        extra_ranks=True,
    )
    print_results(f"Fixed Alpha (alpha={args.alpha})", metrics_fixed)
    
    # Default values used by the metadata save when training is skipped.
    best_mrr = float('nan')
    best_epoch_3 = -1

    if args.skip_beta_training:
        # Load already-trained beta + beta_density and skip stages [2], [3].
        beta_state = torch.load(beta_ckpt_path, map_location=device)
        model.beta.data.copy_(beta_state['beta'].to(device))
        model.beta_density.data.copy_(beta_state['beta_density'].to(device))
        if 'global_beta' in beta_state:
            model.global_beta.data.copy_(beta_state['global_beta'].to(device))
        # Restore feature-standardization stats so adaptive beta matches training.
        if 'feature_mean' in beta_state:
            model.feature_mean.data = beta_state['feature_mean'].to(device)
            model.feature_std.data = beta_state['feature_std'].to(device)
            model.standardize_feature = beta_state.get('standardize_feature', False)
        print(f"\n[2/3] SKIPPED -- loaded trained beta from {beta_ckpt_path}")
        print(f"      seed:           {beta_state.get('seed', 'unknown')}")
        print(f"      stage:          {beta_state.get('stage', 'unknown')}")
        print(f"      best_mrr (3):   {beta_state.get('best_mrr_stage3', 'unknown')}")
        print(f"      epoch_best (3): {beta_state.get('epoch_best_stage3', 'unknown')}")
        print(f"      base_ckpt:      {beta_state.get('base_checkpoint', 'unknown')}")
        best_mrr = beta_state.get('best_mrr_stage3', best_mrr)
        best_epoch_3 = beta_state.get('epoch_best_stage3', best_epoch_3)
        print_beta_stats(model)
    else:
        # === Per-relation / Global Beta before training ===
        _stage3_label = "Per-relation Beta" if model.use_per_relation else "Global Beta"
        print(f"\n[2] Testing with {_stage3_label.upper()} (before training)...")
        metrics_beta_before = evaluate(model, test_dataloader, device, use_beta=True,
                                       normalize_scores=args.normalize_scores)
        print_results(f"{_stage3_label} (untrained)", metrics_beta_before)
        print_beta_stats(model)

        # === Train intercept (per-relation or global) ===
        print(f"\n[3] Training {_stage3_label.lower()} for {args.epochs} epochs...")

        if model.use_per_relation:
            model.beta.requires_grad = True
            model.global_beta.requires_grad = False
            _intercept_param = model.beta
        else:
            model.beta.requires_grad = False
            model.global_beta.requires_grad = True
            _intercept_param = model.global_beta
        model.beta_density.requires_grad = False
        optimizer = torch.optim.Adam([_intercept_param], lr=args.lr)

        best_mrr = 0
        best_beta = model.beta.data.clone()
        best_global = model.global_beta.data.clone()
        best_epoch_3 = 0

        for epoch in range(args.epochs):
            loss = train_beta_epoch(model, valid_train_loader, optimizer, device, adaptive=False,
                                    normalize_scores=args.normalize_scores,
                                    neg_sampling=args.neg_sampling,
                                    num_negatives=args.num_negatives,
                                    mixed_hard_frac=args.mixed_hard_frac,
                                    beta_l2=args.beta_l2)
            model.eval()
            val_metrics = evaluate(model, valid_select_loader, device, use_beta=True,
                                   normalize_scores=args.normalize_scores)

            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Valid-select MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")

            if val_metrics['MRR'] > best_mrr:
                best_mrr = val_metrics['MRR']
                best_beta = model.beta.data.clone()
                best_global = model.global_beta.data.clone()
                best_epoch_3 = epoch + 1
                print(f"  New best MRR!")

        model.beta.data = best_beta
        model.global_beta.data = best_global

        # Save progress after stage 3. beta_density is still 0 here; it will
        # be overwritten by the post-stage-6 save below.
        save_beta_checkpoint(
            beta_ckpt_path, model,
            stage='stage3',
            seed=args.seed,
            best_mrr_stage3=best_mrr,
            epoch_best_stage3=best_epoch_3,
            base_checkpoint=os.path.abspath(args.checkpoint),
            feature_name=model.feature_name,
            standardize_feature=model.standardize_feature,
            feature_mean=model.feature_mean.detach().cpu().clone(),
            feature_std=model.feature_std.detach().cpu().clone(),
        )

    _stage3_label = "Per-relation Beta" if model.use_per_relation else "Global Beta"
    print(f"\n[4] Testing with TRAINED {_stage3_label.lower()}...")
    metrics_beta_after, pq_beta = evaluate(
        model, test_dataloader, device, use_beta=True, return_per_query=True,
        normalize_scores=args.normalize_scores,
    )
    print_results(f"{_stage3_label} (trained)", metrics_beta_after)
    print_beta_stats(model)
    
    # === Adaptive Beta ===
    best_mrr_adaptive = float('nan')
    best_epoch_6 = -1

    if args.skip_beta_training:
        # beta and beta_density were already loaded above; just skip [5] and [6].
        print(f"\n[5/6] SKIPPED -- using loaded beta_density")
        print(f"      best_mrr (6):   {beta_state.get('best_mrr_stage6', 'unknown')}")
        print(f"      epoch_best (6): {beta_state.get('epoch_best_stage6', 'unknown')}")
        best_mrr_adaptive = beta_state.get('best_mrr_stage6', best_mrr_adaptive)
        best_epoch_6 = beta_state.get('epoch_best_stage6', best_epoch_6)
    else:
        print(f"\n[5] Testing with ADAPTIVE BETA (before training)...")
        metrics_adaptive_before = evaluate(model, test_dataloader, device, adaptive_beta=True,
                                           normalize_scores=args.normalize_scores)
        print_results("Adaptive Beta (untrained)", metrics_adaptive_before)

        # Train adaptive beta (beta_rel + beta_density)
        print(f"\n[6] Training adaptive beta for {args.epochs} epochs...")

        model.beta.requires_grad = False
        model.global_beta.requires_grad = False   # frozen from stage 3
        model.beta_density.requires_grad = True
        optimizer_adaptive = torch.optim.Adam(
            [model.beta_density], lr=args.density_lr
        )

        best_mrr_adaptive = 0
        best_beta_adaptive = model.beta.data.clone()
        best_global_adaptive = model.global_beta.data.clone()
        best_density = model.beta_density.data.clone()
        best_epoch_6 = 0

        for epoch in range(args.epochs):
            loss = train_beta_epoch(model, valid_train_loader, optimizer_adaptive, device, adaptive=True,
                                    normalize_scores=args.normalize_scores,
                                    neg_sampling=args.neg_sampling,
                                    num_negatives=args.num_negatives,
                                    mixed_hard_frac=args.mixed_hard_frac,
                                    beta_l2=args.beta_l2)
            model.eval()
            val_metrics = evaluate(model, valid_select_loader, device, adaptive_beta=True,
                                   normalize_scores=args.normalize_scores)

            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Valid-select MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")
            print(f"  beta_density = {model.beta_density.item():.4f}")
            if not model.use_per_relation:
                print(f"  global_beta  = {model.global_beta.item():.4f}  "
                      f"(prob: {torch.sigmoid(model.global_beta).item():.4f})")

            if val_metrics['MRR'] > best_mrr_adaptive:
                best_mrr_adaptive = val_metrics['MRR']
                best_beta_adaptive = model.beta.data.clone()
                best_global_adaptive = model.global_beta.data.clone()
                best_density = model.beta_density.data.clone()
                best_epoch_6 = epoch + 1
                print(f"  New best MRR!")

        model.beta.data = best_beta_adaptive
        model.global_beta.data = best_global_adaptive
        model.beta_density.data = best_density

        # Final save: full trained state from both stages.
        save_beta_checkpoint(
            beta_ckpt_path, model,
            stage='stage6',
            seed=args.seed,
            best_mrr_stage3=best_mrr,
            epoch_best_stage3=best_epoch_3,
            best_mrr_stage6=best_mrr_adaptive,
            epoch_best_stage6=best_epoch_6,
            base_checkpoint=os.path.abspath(args.checkpoint),
            feature_name=model.feature_name,
            standardize_feature=model.standardize_feature,
            feature_mean=model.feature_mean.detach().cpu().clone(),
            feature_std=model.feature_std.detach().cpu().clone(),
        )

    print(f"\n[7] Testing with TRAINED adaptive beta...")
    metrics_adaptive_after, pq_adaptive = evaluate(
        model, test_dataloader, device, adaptive_beta=True, return_per_query=True,
        normalize_scores=args.normalize_scores,
    )
    print_results("Adaptive Beta (trained)", metrics_adaptive_after)
    print_beta_stats(model)
    
    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    _rl = "Rel-Beta" if model.use_per_relation else "Global-Beta"
    print(f"{'Metric':<10} {'Fixed alpha':<14} {_rl:<14} {'Adaptive':<12}")
    print("-" * 54)
    
    for metric in ['Hit@1', 'Hit@3', 'Hit@10', 'MR', 'MRR']:
        fixed = metrics_fixed[metric]
        rel = metrics_beta_after[metric]
        adaptive = metrics_adaptive_after[metric]
        print(f"{metric:<10} {fixed:<14.4f} {rel:<14.4f} {adaptive:<12.4f}")
    
    print(f"\nbeta_density learned = {model.beta_density.item():.4f}")
    fn = model.feature_name
    if model.beta_density.item() > 0:
        print(f"Interpretation: higher {fn} -> trust rules more (as expected)")
    elif model.beta_density.item() < 0:
        print(f"Interpretation: higher {fn} -> trust KGE more (counterintuitive)")
    else:
        print(f"Interpretation: {fn} has no effect on mixing")

    # === Post-eval analyses ===
    # All four reports run on the per-query data captured at steps [1], [4], [7].
    # They are pure CPU bookkeeping; no extra GPU memory beyond what evaluate()
    # already consumed. Method-invariant fields (feature, relations,
    # true_tails, corr, rank_rule, rank_kge) are read from pq_fixed.
    fixed_label  = f"Fixed α={args.alpha}"
    rel_label    = "Rel-Beta" if model.use_per_relation else "Global-Beta"
    method_to_ranks = {
        fixed_label: pq_fixed['ranks'],
        rel_label:   pq_beta['ranks'],
        "Adaptive":  pq_adaptive['ranks'],
    }

    # 1. Per-bucket MRR / Hit@1 / Hit@10 / MR by feature quintile.
    print_bucket_analysis(
        method_to_ranks=method_to_ranks,
        densities=pq_fixed['feature'],
        num_buckets=5,
        feature_name=model.feature_name,
    )

    # 2. Per-relation breakdown linking learned beta[r] to per-relation MRR.
    print_per_relation_table(
        method_to_ranks=method_to_ranks,
        relations=pq_fixed['relations'],
        model=model,
        densities=pq_fixed['feature'],
        fixed_method_name=fixed_label,
        adaptive_method_name="Adaptive",
    )

    # 3. Feature-validity: does the per-query feature predict where rules beat
    #    KGE? Positive corr => 'trust rules more when feature is high' is
    #    justified. Replaces the decision-flip table.
    print_feature_validity(
        feature=pq_fixed['feature'],
        rank_rule=pq_fixed['rank_rule'],
        rank_kge=pq_fixed['rank_kge'],
        feature_name=model.feature_name,
        num_buckets=5,
    )

    # 4. Headroom diagnostic: how correlated are rule and KGE rankings?
    #    Upper-bounds what *any* mixing strategy could possibly achieve.
    print_headroom_diagnostic(
        corrs=pq_fixed['corr'],
        densities=pq_fixed['feature'],
        num_buckets=5,
        feature_name=model.feature_name,
    )


if __name__ == '__main__':
    main()
