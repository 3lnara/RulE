"""
train_beta_grounding_chunked.py — memory-efficient variant of train_beta_grounding.py.

Same pipeline and CLI flags as train_beta_grounding.py; the difference is that
evaluate() and train_beta_epoch() cap peak GPU memory by CPU-offloading per batch
and processing one query at a time during training backprop. Designed for large KGs
such as WN18RR (40 943 entities, hidden_dim=500) where batch-level tensors would OOM.

MEMORY STRATEGY
---------------
evaluate(): logits are moved to CPU and GPU tensors deleted + cache cleared after
each batch. Only O(num_queries) CPU floats accumulate across the loop.

train_beta_epoch(): activations are computed once per batch under no_grad.
Beta is applied per-query (slice [i:i+1]) so each backward graph covers a single
scalar only — peak GPU memory stays at O(num_entities) not O(batch × num_entities).
A single optimizer.step() fires per batch after gradients accumulate from all queries.

Density feature note: model._last_ground_mask is shape [batch, E] after forward().
In the per-query loop it is temporarily replaced with ground_mask_batch[i:i+1] so
compute_adaptive_beta reads the correct single-query density.

PIPELINE
--------
[1]  Fixed-alpha baseline evaluation (test set).
[2]  Per-relation beta evaluation before training (skipped with --joint).
[3]  Train per-relation beta (or joint beta+density with --joint).
[4]  Evaluate trained per-relation beta on test set.
[5]  Adaptive beta before stage-6 training (two-stage path only).
[6]  Train density slope beta_density (two-stage path only).
[7]  Evaluate fully-trained adaptive beta on test set.
Post-eval: per-bucket, per-relation, decision-flip, and headroom analyses.

AVAILABLE FLAGS
---------------
Training control:
  --epochs N               Epochs per stage (default 10).
  --lr LR                  LR for model.beta (default 0.01).
  --density_lr LR          LR for model.beta_density (default 0.01).
  --seed N                 Global RNG seed (default 42).
  --alpha A                Fixed-alpha baseline value (default 1.0).

Negative sampling:
  --negatives {random,hard,mixed}
    random  Uniform sample (original behavior).
    hard    Top-K by current mixed score (maximum gradient).
    mixed   Top-K/2 hard + K/2 random from remainder (default).
  --num_negatives K        Negatives per positive (default 100).

Mixing model:
  --feature {density,max_score,entropy,none}
    density    Fraction of entities where at least one rule fired (default).
    max_score  Max-centered rule logit; high = peaked rule distribution.
    entropy    Negative softmax entropy; high = peaked rule distribution.
    none       No query feature; collapses to per-relation (or global) beta.
  --no_per_relation        Single global beta instead of per-relation vector.
  --joint                  Train beta + beta_density jointly (single stage).
  --normalize_scores       Divide rule and KGE scores by std before mixing.
  --beta_l2 λ              L2 regularization on model.beta (default 0.0).

Persistence:
  --checkpoint PATH        Pre-trained KGE+rule checkpoint (required).
  --beta_checkpoint PATH   Save/load path for beta.pt.
  --skip_beta_training     Load beta.pt and run only test evaluations.

RECOMMENDED ABLATION CONFIGURATIONS
------------------------------------
  Baseline (fixed alpha):        --alpha 1.0
  Density adaptive (default):    --feature density
  Max-score adaptive:            --feature max_score
  Entropy adaptive:              --feature entropy
  Per-relation only (no density): --feature none
  Global beta (no per-rel):      --no_per_relation
  Joint two-param training:      --joint
  Normalized mixing:             --normalize_scores
  Hard negatives:                --negatives hard
  L2-regularized beta:           --beta_l2 0.01
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
    # Negative sampling strategy -----------------------------------------------
    parser.add_argument('--negatives', type=str, default='mixed',
                       choices=['random', 'hard', 'mixed'],
                       help='Negative sampling strategy during beta training. '
                            '"random": uniform sample of --num_negatives candidates '
                            '(original behavior). '
                            '"hard": top-num_negatives negatives by current mixed '
                            'score (maximum gradient contribution). '
                            '"mixed": top-half hardest + half random from the rest '
                            '(default, balances gradient signal and diversity).')
    parser.add_argument('--num_negatives', type=int, default=100,
                       help='Total negatives per positive during beta training. '
                            'Default 100 matches the original hardcoded value. '
                            'For "mixed", split evenly between hard and random halves.')
    # Mixing-model configuration (recorded in checkpoint; verified on load) ------
    parser.add_argument('--feature', type=str, default='density',
                       choices=['density', 'max_score', 'entropy', 'none'],
                       help='Query-level feature for the adaptive mixing head. '
                            '"density": fraction of entities with >=1 rule fired (default). '
                            '"max_score": max-centered rule logit (high = peaked). '
                            '"entropy": negative softmax entropy (high = peaked). '
                            '"none": no query feature; pure per-relation (or global) beta.')
    parser.add_argument('--no_per_relation', action='store_true',
                       help='Replace the 2R-dimensional per-relation beta with a single '
                            'global scalar beta. Useful for ablating the per-relation '
                            'hypothesis; model.beta will have shape [1].')
    parser.add_argument('--joint', action='store_true',
                       help='Train beta + beta_density jointly in one stage '
                            '(both optimized simultaneously with adaptive=True) '
                            'instead of the default two-stage sequential training.')
    parser.add_argument('--normalize_scores', action='store_true',
                       help='Divide rule_logits and kge_score each by their '
                            'batch-level std (+1e-6) before mixing. Applied in both '
                            'train_beta_epoch and evaluate() so all methods are '
                            'evaluated on the same scale.')
    parser.add_argument('--beta_l2', type=float, default=0.0,
                       help='L2 regularization coefficient on model.beta. '
                            'Adds beta_l2 * (beta**2).mean() to the batch loss. '
                            'Shrinks beta toward 0 (neutral 0.5 routing). '
                            'Useful when stage-3 training overshoots on small KGs. '
                            'Default 0.0 (disabled).')
    parser.add_argument('--baseline_only', action='store_true',
                       help='Run only step [1] (fixed-alpha evaluation) and exit. '
                            'No beta training or additional test evaluations are '
                            'performed. Designed for the ablation-matrix fixed_alpha '
                            'variant so the baseline is measured under identical '
                            'conditions to the other runs.')
    # Efficiency flags -----------------------------------------------------------
    parser.add_argument('--pq_cache', type=str, default=None,
                       help='Path for the fixed-alpha per-query cache (.pt). '
                            'If the file does not exist, step [1] runs normally and '
                            'saves its result here. If the file exists, step [1] is '
                            'skipped and the cached metrics_fixed / pq_fixed are '
                            'loaded instead. Pass the same path to all variants in a '
                            'multi-variant job so only the first variant pays the '
                            'cost of the fixed-alpha eval. '
                            'Cache is invalidated (re-run) if alpha or '
                            'normalize_scores do not match the cached values.')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                       help='Batch size for test and validation-select evaluation. '
                            'The training dataloader always uses batch_size=4 for '
                            'memory safety (chunked backward). Larger values here '
                            'are safe for smaller KGs: FB15k237 (14 541 entities) '
                            'can use 32; WN18RR (40 943 entities) should stay at 4. '
                            'Default 4 matches the original hardcoded value.')
    parser.add_argument('--skip_intermediate_eval', action='store_true',
                       help='Skip step [4] (per-relation beta test eval). '
                            'Saves one full test-set pass per run. The Rel-Beta '
                            'column is omitted from the summary and post-eval '
                            'analyses. Useful for hyperparameter sweeps where only '
                            'the final adaptive-beta result (step [7]) matters.')
    return parser.parse_args()


def evaluate(model, dataloader, device, use_beta=False, adaptive_beta=False, alpha=3.0,
             return_per_query=False, normalize_scores=False):
    """Evaluate the model.

    Ranks are computed per-batch and moved to CPU immediately so the GPU never
    holds more than one batch of [batch, num_entities] logits at a time. This
    is critical for large datasets such as WN18RR (40 943 entities).

    If `return_per_query` is True, returns (metrics, per_query_dict). The dict
    keys mirror what the post-eval analyses need:
      - 'ranks'      : Tensor[N]  filtered, tie-averaged rank per query.
      - 'densities'  : Tensor[N]  rule grounding density per query (same signal
                                  that compute_adaptive_beta conditions on).
      - 'relations'  : Tensor[N]  relation id per query (long).
      - 'top1'       : Tensor[N]  predicted entity id (argmax over flag-True
                                  candidates) under THIS method's mixed score.
      - 'true_tails' : Tensor[N]  ground-truth tail entity id per query.
      - 'corr'       : Tensor[N]  per-query Spearman correlation between the
                                  pre-mix rule score and the kge_score over all
                                  candidate entities. Method-invariant.
    All tensors are on CPU and aligned by index. Memory: per-batch transients
    are freed by the existing `empty_cache()` call; only O(num_queries) floats
    accumulate across the eval loop.
    """
    model.eval()
    all_ranks = []
    all_densities = [] if return_per_query else None
    all_corr      = [] if return_per_query else None
    all_relations = [] if return_per_query else None
    all_top1      = [] if return_per_query else None
    all_true      = [] if return_per_query else None

    with torch.no_grad():
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)

            # Forward pass (populates model._last_ground_mask). Keep
            # `rule_logits` distinct from the mixed `logits` -- we need it
            # below for the rule/KGE rank correlation.
            rule_logits, mask = model(all_h, all_r, None)

            # Read per-query density. mean(dim=-1) reduces [batch, num_entities]
            # bool -> [batch] float; .cpu() drops it off the GPU immediately.
            if return_per_query:
                density_batch = model._last_ground_mask.float().mean(dim=-1).cpu()
                all_densities.append(density_batch)

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

            # Score normalization: applied after Spearman (rank-invariant) but
            # before mixing so it affects every method consistently.
            if normalize_scores:
                rule_logits = rule_logits / (rule_logits.std() + 1e-6)
                kge_score   = kge_score   / (kge_score.std()   + 1e-6)

            if adaptive_beta:
                beta, _ = model.compute_adaptive_beta(rule_logits, all_r)
                logits = beta * rule_logits + (1 - beta) * kge_score
            elif use_beta:
                beta_idx = 0 if model.no_per_relation else all_r[0]
                beta = torch.sigmoid(model.beta[beta_idx]).unsqueeze(-1)
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
        return metrics, {
            'ranks':      ranks,
            'densities':  torch.cat(all_densities, dim=0),
            'relations':  torch.cat(all_relations, dim=0).long(),
            'top1':       torch.tensor(all_top1, dtype=torch.long),
            'true_tails': torch.cat(all_true, dim=0).long(),
            'corr':       torch.cat(all_corr, dim=0),
        }
    return metrics


def train_beta_epoch(model, dataloader, optimizer, device, adaptive=False,
                     negatives='mixed', num_negatives=100,
                     normalize_scores=False, joint=False, beta_l2=0.0):
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

    Args:
        negatives:        'random' | 'hard' | 'mixed'. See --negatives flag.
        num_negatives:    Total negatives per positive. See --num_negatives.
        normalize_scores: Divide rule_logits and kge_score by their std before
                          mixing. See --normalize_scores flag.
        joint:            If True, both model.beta and model.beta_density get
                          requires_grad=True simultaneously (--joint training).
        beta_l2:          L2 regularization coefficient on model.beta.
                          See --beta_l2 flag.

    Returns:
        (avg_loss, active_frac) — active_frac is the fraction of per-example
        losses that were > 0 (i.e., contributed a gradient).
    """
    model.train()

    # Freeze parameters not being trained this stage.
    # `joint` enables both beta and beta_density; but beta_density is skipped
    # when feature='none' because the density term is identically zero there
    # (gradient-free weight would just drift or sit at 0 pointlessly).
    density_trainable = model.feature_name != 'none'
    for name, param in model.named_parameters():
        if name == 'beta':
            param.requires_grad = joint or (not adaptive)
        elif name == 'beta_density':
            param.requires_grad = density_trainable and (joint or adaptive)
        else:
            param.requires_grad = False

    total_loss = 0.0
    num_updates = 0
    active_count = 0
    total_count = 0

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

        # Save the ground mask for the whole batch before the per-sample loop.
        # Inside the loop we temporarily replace it with the single-sample slice
        # so compute_adaptive_beta reads the correct density.
        ground_mask_batch = model._last_ground_mask

        # Score normalization applied after Spearman (rank-invariant) but before
        # mixing so it affects every method consistently.
        if normalize_scores:
            rule_logits = rule_logits / (rule_logits.std() + 1e-6)
            kge_score   = kge_score   / (kge_score.std()   + 1e-6)

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
        logit_i = beta_i = neg_scores = true_score = raw_loss = None  # for the final del
        for i in valid_indices:
            r_i = all_r[i:i+1]
            rl_i = rule_logits[i:i+1]   # [1, E] — detached from frozen no_grad graph
            kg_i = kge_score[i:i+1]     # [1, E]

            if adaptive:
                model._last_ground_mask = ground_mask_batch[i:i+1]
                beta_i, _ = model.compute_adaptive_beta(rl_i, r_i)
                model._last_ground_mask = ground_mask_batch
            else:
                beta_idx = 0 if model.no_per_relation else r_i[0]
                beta_i = torch.sigmoid(model.beta[beta_idx]).unsqueeze(-1)

            logit_i = beta_i * rl_i + (1 - beta_i) * kg_i   # [1, E], grad through beta only

            negative_mask = flag[i] & (torch.arange(logit_i.size(1), device=device) != all_t[i])
            neg_scores = logit_i[0][negative_mask]
            n_neg = neg_scores.size(0)

            if negatives == 'random':
                if n_neg > num_negatives:
                    idx = torch.randperm(n_neg, device=device)[:num_negatives]
                    neg_scores = neg_scores[idx]
            elif negatives == 'hard':
                k = min(num_negatives, n_neg)
                _, top_idx = torch.topk(neg_scores, k)
                neg_scores = neg_scores[top_idx]
            else:  # 'mixed'
                if n_neg > num_negatives:
                    half = num_negatives // 2
                    k_hard = min(half, n_neg)
                    _, hard_idx = torch.topk(neg_scores, k_hard)
                    rest_mask = torch.ones(n_neg, dtype=torch.bool, device=device)
                    rest_mask[hard_idx] = False
                    rest_scores = neg_scores[rest_mask]
                    k_rand = min(num_negatives - k_hard, rest_scores.size(0))
                    if k_rand > 0:
                        rand_idx = torch.randperm(rest_scores.size(0), device=device)[:k_rand]
                        neg_scores = torch.cat(
                            [neg_scores[hard_idx], rest_scores[rand_idx]]
                        )
                    else:
                        neg_scores = neg_scores[hard_idx]

            true_score = logit_i[0, all_t[i]]
            # Per-sample mean over negatives; clamp at zero = margin-ranking loss.
            # Divide by N so the per-sample gradients add up to the batch mean
            # (equivalent to stacking and calling .mean().backward() in the
            # non-chunked version, but with per-sample memory).
            raw_loss = torch.clamp(1.0 - true_score + neg_scores, min=0).mean()
            total_count += 1
            if raw_loss.item() > 0:
                active_count += 1
            loss = raw_loss / N
            loss.backward()
            batch_loss_sum += raw_loss.item() / N

        # Optional L2 regularization on beta (applied once per batch, after the
        # per-sample loop so its gradient is not divided by N).
        if beta_l2 > 0.0:
            l2_loss = beta_l2 * (model.beta ** 2).mean()
            l2_loss.backward()

        optimizer.step()

        # batch_loss_sum = (1/N) * sum_i loss_i  -> the actual batch mean loss
        total_loss += batch_loss_sum
        num_updates += 1

        del logit_i, beta_i, neg_scores, true_score, raw_loss
        del rule_logits, kge_score
        if device.type == "cuda":
            torch.cuda.empty_cache()

    active_frac = active_count / max(total_count, 1)
    return total_loss / max(num_updates, 1), active_frac


def print_results(name, metrics):
    print(f"\n{'='*60}")
    print(f"Results for: {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def print_bucket_analysis(method_to_ranks, densities, num_buckets=5,
                          title="PER-BUCKET ANALYSIS (test set, bucketed by rule grounding density)"):
    """Print per-bucket MRR / Hit@1 / Hit@10 / MR across combination methods.

    Layout is one row per (bucket, method): for each density bucket we emit
    one block of method rows, with MRR, Hit@1, Hit@10, MR as columns. This
    lets the reader see *which* aspect of the ranking moved (e.g. MR can change
    while MRR is stationary).

    Memory: pure CPU bookkeeping over O(num_queries) floats. Safe for WN18RR.
    """
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

    print(f"\n  Total queries: {n_total}   Density range: [{d.min().item():.4f}, {d.max().item():.4f}]")


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

    if model.no_per_relation:
        with torch.no_grad():
            g_beta = torch.sigmoid(model.beta[0]).item()
        print(f"  (--no_per_relation: single global beta = {g_beta:.4f}; "
              f"per-relation breakdown not available)")
        return

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
        f"{'r':>4}  {'n':>6}  {'beta[r]':>9}  {'dens_mean':>10}  "
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


def print_decision_flip_analysis(method_to_top1, true_tails, pairs,
                                 relations=None, model=None):
    """Decision-flip stats for each (method_A, method_B) pair.

    Global block, for every pair: flips, A_correct_only, B_correct_only,
    neither_correct, and the Hit@1 delta from those flips ((B - A) / N).

    Per-relation block (printed only when both `relations` and `model` are
    given): for each relation that has at least one flip, prints n, beta[r],
    flips, A_only, B_only, neither, and the *local* Hit@1 delta within that
    relation. Sorted by |local Hit@1 delta| descending.

    Args:
        method_to_top1: dict[str, Tensor[N]] of per-query top-1 entity ids.
        true_tails:     Tensor[N] of ground-truth tail ids.
        pairs:          list of (name_A, name_B). Report framed as "B over A".
        relations:      optional Tensor[N] of per-query relation ids.
        model:          optional RulE model whose .beta provides beta[r].
    """
    print(f"\n{'='*80}")
    print("DECISION-FLIP ANALYSIS")
    print(f"{'='*80}")
    n = true_tails.numel()
    print(f"Total test queries: {n}\n")

    rels_cpu = relations.cpu().long() if relations is not None else None
    beta_sigmoid = None
    if model is not None:
        with torch.no_grad():
            beta_sigmoid = torch.sigmoid(model.beta).detach().cpu()

    for (a_name, b_name) in pairs:
        a_top1 = method_to_top1[a_name]
        b_top1 = method_to_top1[b_name]

        flip_mask = (a_top1 != b_top1)
        n_flips = int(flip_mask.sum().item())
        if n_flips == 0:
            print(f"  {b_name} vs {a_name}: 0 flips (identical top-1 predictions)\n")
            continue

        a_correct = (a_top1 == true_tails)
        b_correct = (b_top1 == true_tails)

        a_only = int((flip_mask & a_correct & ~b_correct).sum().item())
        b_only = int((flip_mask & b_correct & ~a_correct).sum().item())
        neither = int((flip_mask & ~a_correct & ~b_correct).sum().item())
        # both_correct is impossible because the top-1s differ.

        flip_rate = n_flips / n
        hit1_delta = (b_only - a_only) / n

        print(f"  {b_name} vs {a_name}:")
        print(f"    flips:                   {n_flips:>6} ({flip_rate*100:5.2f}% of queries)")
        print(f"    {a_name} correct only:   {a_only:>6}")
        print(f"    {b_name} correct only:   {b_only:>6}")
        print(f"    neither correct:         {neither:>6}")
        print(f"    Hit@1 delta from flips: {hit1_delta:+.4f}")

        if rels_cpu is not None:
            rows = []
            for r in sorted(set(rels_cpu.tolist())):
                rel_mask = (rels_cpu == r)
                rel_n = int(rel_mask.sum().item())
                if rel_n == 0:
                    continue
                rel_flip = flip_mask & rel_mask
                rel_n_flips = int(rel_flip.sum().item())
                if rel_n_flips == 0:
                    continue
                rel_a_only = int((rel_flip & a_correct & ~b_correct).sum().item())
                rel_b_only = int((rel_flip & b_correct & ~a_correct).sum().item())
                rel_neither = int((rel_flip & ~a_correct & ~b_correct).sum().item())
                # Normalize by n_in_relation so the column reads as the
                # per-relation accuracy contribution from method B over A.
                rel_dh1 = (rel_b_only - rel_a_only) / rel_n
                if beta_sigmoid is None:
                    beta_r = float('nan')
                elif model is not None and model.no_per_relation:
                    beta_r = beta_sigmoid[0].item()  # same global scalar for every row
                else:
                    beta_r = beta_sigmoid[r].item() if r < beta_sigmoid.numel() else float('nan')
                rows.append({
                    'r': r, 'n': rel_n, 'beta': beta_r,
                    'flips': rel_n_flips,
                    'a_only': rel_a_only, 'b_only': rel_b_only, 'neither': rel_neither,
                    'dh1': rel_dh1,
                })
            rows.sort(key=lambda x: abs(x['dh1']), reverse=True)
            if rows:
                print()
                print(f"    Per-relation (relations with >=1 flip; sorted by |local dHit@1| desc):")
                hdr = (
                    f"      {'r':>4}  {'n':>6}  {'beta[r]':>9}  "
                    f"{'flips':>6}  {'A_only':>6}  {'B_only':>6}  {'neither':>7}  "
                    f"{'dHit@1':>9}"
                )
                print(hdr)
                print("      " + "-" * (len(hdr) - 6))
                for row in rows:
                    print(
                        f"      {row['r']:>4}  {row['n']:>6}  {row['beta']:>9.4f}  "
                        f"{row['flips']:>6}  {row['a_only']:>6}  {row['b_only']:>6}  "
                        f"{row['neither']:>7}  {row['dh1']:>+9.4f}"
                    )
        print()


def print_headroom_diagnostic(corrs, densities, num_buckets=5):
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

    print(f"  Per-density-bucket mean correlation:")
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
        beta_values = torch.sigmoid(model.beta).cpu().numpy()
        print(f"\nBeta Statistics (per-relation weight for rule score):")
        print(f"  Mean:   {beta_values.mean():.4f}")
        print(f"  Std:    {beta_values.std():.4f}")
        print(f"  Min:    {beta_values.min():.4f}")
        print(f"  Max:    {beta_values.max():.4f}")
        print(f"  Median: {float(torch.median(torch.tensor(beta_values))):.4f}")
        print(f"  Density weight (beta_density): {model.beta_density.item():.4f}")
        if model.beta_density.item() > 0:
            print(f"    -> Positive: denser groundings push beta UP (trust rules more)")
        elif model.beta_density.item() < 0:
            print(f"    -> Negative: denser groundings push beta DOWN (trust KGE more)")
        else:
            print(f"    -> Zero: grounding density has no effect (pure per-relation beta)")


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
        'beta':         model.beta.data.detach().cpu().clone(),
        'beta_density': model.beta_density.data.detach().cpu().clone(),
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
    print("\nFull configuration:")
    for key, val in sorted(vars(args).items()):
        print(f"  {key}: {val}")

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
        no_per_relation=args.no_per_relation,
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
    # Training always uses train_batch_size=4 (chunked per-sample backward).
    # Evaluation uses args.eval_batch_size (default 4, safe for WN18RR).
    # For FB15k237 (14 541 entities, hidden_dim=500):
    #   eval_batch_size=32 → [32, 14541, 1000] ≈ 0.94 GB  (safe on V100)
    # For WN18RR (40 943 entities):
    #   eval_batch_size=4  → [4,  40943, 1000] ≈ 655 MB   (safe)
    #   eval_batch_size=32 → [32, 40943, 1000] ≈ 5.2 GB   (OOM on 16 GB GPU)
    train_batch_size = 4
    eval_batch_size  = args.eval_batch_size
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

    valid_train_set = FactListValidDataset(graph, train_batch_size, train_facts)
    valid_train_loader = DataLoader(valid_train_set, batch_size=1, num_workers=0)

    valid_select_set = FactListValidDataset(graph, eval_batch_size, select_facts)
    valid_select_loader = DataLoader(valid_select_set, batch_size=1, num_workers=0)

    test_set = TestDataset(graph, eval_batch_size)
    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0)

    print(f"\n{'='*60}")
    print("BETA TRAINING DURING GROUNDING PHASE")
    print(f"{'='*60}")

    # === Baseline: Fixed Alpha ===
    # Capture per-query data here (and at the two later test evals) so we can
    # run the post-eval analyses at the end. The dict contains:
    #   ranks, densities, relations, top1, true_tails, corr.
    # Of these, densities/relations/true_tails/corr are method-invariant -- we
    # read them off the Fixed-alpha call only. top1 and ranks are method-specific.
    #
    # --pq_cache: if the cache file exists and matches alpha/normalize_scores,
    # skip the expensive test-set forward pass and load the saved result.
    metrics_fixed = pq_fixed = None
    pq_cache_hit = False
    if args.pq_cache and os.path.exists(args.pq_cache):
        cached = torch.load(args.pq_cache, map_location='cpu')
        if cached.get('alpha') == args.alpha and cached.get('normalize_scores') == args.normalize_scores:
            metrics_fixed = cached['metrics_fixed']
            pq_fixed      = cached['pq_fixed']
            pq_cache_hit  = True
            print(f"\n[1] SKIPPED -- loaded fixed-alpha results from cache: {args.pq_cache}")
            print_results(f"Fixed Alpha (alpha={args.alpha})", metrics_fixed)
        else:
            print(f"\n[1] pq_cache found but alpha/normalize_scores mismatch "
                  f"(cached alpha={cached.get('alpha')}, normalize_scores={cached.get('normalize_scores')}); "
                  f"re-running step [1].")

    if not pq_cache_hit:
        print(f"\n[1] Testing with FIXED ALPHA = {args.alpha} (baseline)...")
        metrics_fixed, pq_fixed = evaluate(
            model, test_dataloader, device, use_beta=False, alpha=args.alpha,
            return_per_query=True, normalize_scores=args.normalize_scores,
        )
        print_results(f"Fixed Alpha (alpha={args.alpha})", metrics_fixed)
        if args.pq_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.pq_cache)), exist_ok=True)
            torch.save({
                'metrics_fixed':    metrics_fixed,
                'pq_fixed':         pq_fixed,
                'alpha':            args.alpha,
                'normalize_scores': args.normalize_scores,
            }, args.pq_cache)
            print(f"  Saved fixed-alpha cache -> {args.pq_cache}")

    if args.baseline_only:
        print("\n(--baseline_only: stopping after step [1])")
        return

    # Default values used by the metadata save when training is skipped.
    best_mrr = float('nan')
    best_epoch_3 = -1

    if args.skip_beta_training:
        # Load already-trained beta + beta_density; skip stages [2], [3].
        beta_state = torch.load(beta_ckpt_path, map_location=device)

        # Shape guard: catches --no_per_relation mismatch before copy_.
        saved_beta = beta_state['beta'].to(device)
        if saved_beta.shape != model.beta.data.shape:
            raise ValueError(
                f"model.beta shape mismatch: checkpoint has {tuple(saved_beta.shape)} "
                f"but model expects {tuple(model.beta.data.shape)}. "
                f"Check --no_per_relation matches the training config."
            )
        model.beta.data.copy_(saved_beta)
        model.beta_density.data.copy_(beta_state['beta_density'].to(device))
        print(f"\n[2/3] SKIPPED -- loaded trained beta from {beta_ckpt_path}")
        print(f"      seed:           {beta_state.get('seed', 'unknown')}")
        print(f"      stage:          {beta_state.get('stage', 'unknown')}")
        print(f"      best_mrr (3):   {beta_state.get('best_mrr_stage3', 'unknown')}")
        print(f"      epoch_best (3): {beta_state.get('epoch_best_stage3', 'unknown')}")
        print(f"      base_ckpt:      {beta_state.get('base_checkpoint', 'unknown')}")
        best_mrr = beta_state.get('best_mrr_stage3', best_mrr)
        best_epoch_3 = beta_state.get('epoch_best_stage3', best_epoch_3)

        # Verify checkpoint was trained under a matching configuration.
        # Keys absent from old checkpoints (value=None) are silently skipped.
        compat_keys = ('feature', 'no_per_relation', 'joint', 'normalize_scores')
        cur_cfg = {k: getattr(args, k) for k in compat_keys}
        mismatches = [
            f"  {k}: checkpoint={beta_state.get(k)!r}, current={cur_cfg[k]!r}"
            for k in compat_keys
            if beta_state.get(k) is not None and beta_state.get(k) != cur_cfg[k]
        ]
        if mismatches:
            raise ValueError(
                "Beta checkpoint was trained under a different configuration:\n"
                + "\n".join(mismatches)
                + "\nEither retrain without --skip_beta_training, or pass the "
                "same flags that were used during training."
            )
        print_beta_stats(model)

    elif args.joint:
        # === Joint training: beta (+ optionally beta_density) simultaneously ===
        # When feature='none', beta_density * 0 = 0 always, so exclude it
        # from the optimizer and from requires_grad to avoid misleading output.
        has_density = (args.feature != 'none')
        stage_params = [model.beta, model.beta_density] if has_density else [model.beta]
        param_label = "beta + beta_density" if has_density else "beta"
        print(f"\n[3-joint] Jointly training {param_label} for {args.epochs} epochs...")
        optimizer_joint = torch.optim.Adam(stage_params, lr=args.lr)

        best_mrr = 0
        best_beta = model.beta.data.clone()
        best_density = model.beta_density.data.clone()
        best_epoch_3 = 0

        for epoch in range(args.epochs):
            loss, active_frac = train_beta_epoch(
                model, valid_train_loader, optimizer_joint, device,
                adaptive=True, joint=True,
                negatives=args.negatives, num_negatives=args.num_negatives,
                normalize_scores=args.normalize_scores, beta_l2=args.beta_l2,
            )
            model.eval()
            val_metrics = evaluate(model, valid_select_loader, device, adaptive_beta=True,
                                   normalize_scores=args.normalize_scores)

            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Active fraction: {active_frac:.4f}")
            print(f"  Valid-select MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")
            if has_density:
                print(f"  beta_density = {model.beta_density.item():.4f}")

            if val_metrics['MRR'] > best_mrr:
                best_mrr = val_metrics['MRR']
                best_beta = model.beta.data.clone()
                best_density = model.beta_density.data.clone()
                best_epoch_3 = epoch + 1
                print(f"  New best MRR!")

        model.beta.data = best_beta
        model.beta_density.data = best_density

        save_beta_checkpoint(
            beta_ckpt_path, model,
            stage='joint',
            seed=args.seed,
            best_mrr_stage3=best_mrr,
            epoch_best_stage3=best_epoch_3,
            best_mrr_stage6=best_mrr,
            epoch_best_stage6=best_epoch_3,
            base_checkpoint=os.path.abspath(args.checkpoint),
            feature=args.feature,
            no_per_relation=args.no_per_relation,
            joint=args.joint,
            normalize_scores=args.normalize_scores,
        )
        print_beta_stats(model)

    else:
        # === Two-stage path ===
        # [2] Untrained per-relation beta
        print(f"\n[2] Testing with PER-RELATION BETA (before training)...")
        metrics_beta_before = evaluate(model, test_dataloader, device, use_beta=True,
                                       normalize_scores=args.normalize_scores)
        print_results("Per-relation Beta (untrained)", metrics_beta_before)
        print_beta_stats(model)

        # [3] Train per-relation beta
        print(f"\n[3] Training per-relation beta for {args.epochs} epochs...")
        optimizer = torch.optim.Adam([model.beta], lr=args.lr)

        best_mrr = 0
        best_beta = model.beta.data.clone()
        best_epoch_3 = 0

        for epoch in range(args.epochs):
            loss, active_frac = train_beta_epoch(
                model, valid_train_loader, optimizer, device, adaptive=False,
                negatives=args.negatives, num_negatives=args.num_negatives,
                normalize_scores=args.normalize_scores, beta_l2=args.beta_l2,
            )
            model.eval()
            val_metrics = evaluate(model, valid_select_loader, device, use_beta=True,
                                   normalize_scores=args.normalize_scores)

            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Active fraction: {active_frac:.4f}")
            print(f"  Valid-select MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")

            if val_metrics['MRR'] > best_mrr:
                best_mrr = val_metrics['MRR']
                best_beta = model.beta.data.clone()
                best_epoch_3 = epoch + 1
                print(f"  New best MRR!")

        model.beta.data = best_beta

        # Save stage-3 checkpoint (beta_density still 0; overwritten by stage-6 save).
        save_beta_checkpoint(
            beta_ckpt_path, model,
            stage='stage3',
            seed=args.seed,
            best_mrr_stage3=best_mrr,
            epoch_best_stage3=best_epoch_3,
            base_checkpoint=os.path.abspath(args.checkpoint),
            feature=args.feature,
            no_per_relation=args.no_per_relation,
            joint=args.joint,
            normalize_scores=args.normalize_scores,
        )

    metrics_beta_after = pq_beta = None
    if args.skip_intermediate_eval:
        print(f"\n[4] SKIPPED (--skip_intermediate_eval)")
    else:
        print(f"\n[4] Testing with TRAINED per-relation beta...")
        metrics_beta_after, pq_beta = evaluate(
            model, test_dataloader, device, use_beta=True, return_per_query=True,
            normalize_scores=args.normalize_scores,
        )
        print_results("Per-relation Beta (trained)", metrics_beta_after)
        print_beta_stats(model)

    # === Adaptive Beta — stages 5 and 6 ===
    best_mrr_adaptive = float('nan')
    best_epoch_6 = -1

    if args.skip_beta_training:
        # beta and beta_density were already loaded above; skip [5] and [6].
        print(f"\n[5/6] SKIPPED -- using loaded beta_density")
        print(f"      best_mrr (6):   {beta_state.get('best_mrr_stage6', 'unknown')}")
        print(f"      epoch_best (6): {beta_state.get('epoch_best_stage6', 'unknown')}")
        best_mrr_adaptive = beta_state.get('best_mrr_stage6', best_mrr_adaptive)
        best_epoch_6 = beta_state.get('epoch_best_stage6', best_epoch_6)

    elif args.joint:
        # Already trained jointly in [3-joint]; nothing more to do.
        print(f"\n[5/6] SKIPPED -- joint training already optimised beta + beta_density")
        best_mrr_adaptive = best_mrr
        best_epoch_6 = best_epoch_3

    else:
        # [5] Untrained adaptive eval
        print(f"\n[5] Testing with ADAPTIVE BETA (before training)...")
        metrics_adaptive_before = evaluate(model, test_dataloader, device, adaptive_beta=True,
                                           normalize_scores=args.normalize_scores)
        print_results("Adaptive Beta (untrained)", metrics_adaptive_before)

        # [6] Train density slope beta_density
        print(f"\n[6] Training adaptive beta for {args.epochs} epochs...")
        optimizer_adaptive = torch.optim.Adam([model.beta_density], lr=args.density_lr)

        best_mrr_adaptive = 0
        best_beta_adaptive = model.beta.data.clone()
        best_density = model.beta_density.data.clone()
        best_epoch_6 = 0

        for epoch in range(args.epochs):
            loss, active_frac = train_beta_epoch(
                model, valid_train_loader, optimizer_adaptive, device, adaptive=True,
                negatives=args.negatives, num_negatives=args.num_negatives,
                normalize_scores=args.normalize_scores, beta_l2=args.beta_l2,
            )
            model.eval()
            val_metrics = evaluate(model, valid_select_loader, device, adaptive_beta=True,
                                   normalize_scores=args.normalize_scores)

            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Active fraction: {active_frac:.4f}")
            print(f"  Valid-select MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")
            print(f"  beta_density = {model.beta_density.item():.4f}")

            if val_metrics['MRR'] > best_mrr_adaptive:
                best_mrr_adaptive = val_metrics['MRR']
                best_beta_adaptive = model.beta.data.clone()
                best_density = model.beta_density.data.clone()
                best_epoch_6 = epoch + 1
                print(f"  New best MRR!")

        model.beta.data = best_beta_adaptive
        model.beta_density.data = best_density

        # Final checkpoint: full trained state from both stages.
        save_beta_checkpoint(
            beta_ckpt_path, model,
            stage='stage6',
            seed=args.seed,
            best_mrr_stage3=best_mrr,
            epoch_best_stage3=best_epoch_3,
            best_mrr_stage6=best_mrr_adaptive,
            epoch_best_stage6=best_epoch_6,
            base_checkpoint=os.path.abspath(args.checkpoint),
            feature=args.feature,
            no_per_relation=args.no_per_relation,
            joint=args.joint,
            normalize_scores=args.normalize_scores,
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
    if pq_beta is not None:
        print(f"{'Metric':<10} {'Fixed alpha':<12} {'Rel-Beta':<12} {'Adaptive':<12}")
        print("-" * 50)
        for metric in ['Hit@1', 'Hit@3', 'Hit@10', 'MR', 'MRR']:
            print(f"{metric:<10} {metrics_fixed[metric]:<12.4f} "
                  f"{metrics_beta_after[metric]:<12.4f} "
                  f"{metrics_adaptive_after[metric]:<12.4f}")
    else:
        print(f"{'Metric':<10} {'Fixed alpha':<12} {'Adaptive':<12}")
        print("-" * 38)
        for metric in ['Hit@1', 'Hit@3', 'Hit@10', 'MR', 'MRR']:
            print(f"{metric:<10} {metrics_fixed[metric]:<12.4f} "
                  f"{metrics_adaptive_after[metric]:<12.4f}")

    print(f"\nbeta_density learned = {model.beta_density.item():.4f}")
    if model.beta_density.item() > 0:
        print("Interpretation: Dense groundings -> trust rules more (as expected)")
    elif model.beta_density.item() < 0:
        print("Interpretation: Dense groundings -> trust KGE more (counterintuitive)")
    else:
        print("Interpretation: Grounding density has no effect")

    # === Post-eval analyses ===
    # All four reports run on the per-query data captured at steps [1], [4], [7].
    # They are pure CPU bookkeeping; no extra GPU memory beyond what evaluate()
    # already consumed. Method-invariant fields (densities, relations,
    # true_tails, corr) are read from pq_fixed.
    fixed_label = f"Fixed α={args.alpha}"
    method_to_ranks = {fixed_label: pq_fixed['ranks']}
    method_to_top1  = {fixed_label: pq_fixed['top1']}
    if pq_beta is not None:
        method_to_ranks["Rel-Beta"] = pq_beta['ranks']
        method_to_top1["Rel-Beta"]  = pq_beta['top1']
    method_to_ranks["Adaptive"] = pq_adaptive['ranks']
    method_to_top1["Adaptive"]  = pq_adaptive['top1']

    flip_pairs = [(fixed_label, "Adaptive")]
    if pq_beta is not None:
        flip_pairs.append(("Rel-Beta", "Adaptive"))

    # 1. Per-bucket MRR / Hit@1 / Hit@10 / MR by density quintile.
    print_bucket_analysis(
        method_to_ranks=method_to_ranks,
        densities=pq_fixed['densities'],
        num_buckets=5,
    )

    # 2. Per-relation breakdown linking learned beta[r] to per-relation MRR.
    print_per_relation_table(
        method_to_ranks=method_to_ranks,
        relations=pq_fixed['relations'],
        model=model,
        densities=pq_fixed['densities'],
        fixed_method_name=fixed_label,
        adaptive_method_name="Adaptive",
    )

    # 3. Decision-flip analysis: do Adaptive / Rel-Beta predict different
    #    top-1 entities than Fixed-alpha, and are the flips productive?
    #    relations=... and model=... enable a per-relation breakdown that
    #    can be cross-referenced with the PER-RELATION TABLE above.
    print_decision_flip_analysis(
        method_to_top1=method_to_top1,
        true_tails=pq_fixed['true_tails'],
        pairs=flip_pairs,
        relations=pq_fixed['relations'],
        model=model,
    )

    # 4. Headroom diagnostic: how correlated are rule and KGE rankings?
    #    Upper-bounds what *any* mixing strategy could possibly achieve.
    print_headroom_diagnostic(
        corrs=pq_fixed['corr'],
        densities=pq_fixed['densities'],
        num_buckets=5,
    )


if __name__ == '__main__':
    main()
