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
    return parser.parse_args()


def evaluate(model, dataloader, device, use_beta=False, adaptive_beta=False, alpha=3.0,
             return_per_query=False):
    """Evaluate the model.

    If `return_per_query` is True, returns (metrics, per_query_dict). The dict
    keys mirror what the post-eval analyses need:
      - 'ranks'      : Tensor[N]  filtered, tie-averaged rank per query.
      - 'densities'  : Tensor[N]  rule grounding density per query (same signal
                                  that compute_adaptive_beta conditions on).
      - 'relations'  : Tensor[N]  relation id per query (long).
      - 'top1'       : Tensor[N]  predicted entity id (argmax over flag-True
                                  candidates) under THIS method's mixed score.
      - 'true_tails' : Tensor[N]  ground-truth tail entity id per query.
                                  Method-invariant; same value at all three
                                  test calls -- carried per-call for symmetry.
      - 'corr'       : Tensor[N]  per-query Spearman correlation between the
                                  pre-mix rule_score and the kge_score over all
                                  candidate entities. Method-invariant for the
                                  same reason as 'true_tails'.
    All tensors are on CPU and aligned by index.
    """
    model.eval()

    concat_logits = []
    concat_all_h = []
    concat_all_r = []
    concat_all_t = []
    concat_flag = []
    concat_mask = []
    concat_density = [] if return_per_query else None
    concat_corr = [] if return_per_query else None

    with torch.no_grad():
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)

            # Forward pass: raw rule scores (with bias added, see model.forward)
            rule_logits, mask = model(all_h, all_r, None)

            # Per-query rule-grounding density: fraction of candidate entities
            # for which at least one rule fired. Read from the model's stash
            # populated by forward() -- this is the *same* signal that
            # compute_adaptive_beta uses, so the bucket edges line up with the
            # feature the adaptive head conditions on. NOTE: computing density
            # from rule_logits.min() / eps was wrong -- self.bias variance
            # collapsed it to ~1 for every query (see compute_adaptive_beta).
            if return_per_query:
                density = model._last_ground_mask.float().mean(dim=-1)  # [batch]
                concat_density.append(density.detach())

            kge_score = model.compute_g_KGE(all_h, all_r)

            # Per-query Spearman corr between rule and KGE rankings, over all
            # candidate entities (no flag filter for speed -- flag drops a
            # tiny fraction of entities per query, doesn't move correlation).
            # Spearman = Pearson on rank vectors. Computed on GPU once per
            # batch, reduced to one float per query, then moved to CPU.
            #
            # This number is a property of (rule_logits, kge_score) and does
            # NOT depend on the mixing strategy below, so it's identical
            # across the three test calls. We compute it anyway for symmetry
            # (cheap), and the caller can read it once from the Fixed-alpha
            # call.
            if return_per_query:
                ra = rule_logits.argsort(dim=-1).argsort(dim=-1).float()
                rb = kge_score.argsort(dim=-1).argsort(dim=-1).float()
                a_c = ra - ra.mean(dim=-1, keepdim=True)
                b_c = rb - rb.mean(dim=-1, keepdim=True)
                num = (a_c * b_c).sum(dim=-1)
                denom = torch.sqrt(
                    (a_c ** 2).sum(dim=-1) * (b_c ** 2).sum(dim=-1)
                ).clamp_min(1e-12)
                concat_corr.append((num / denom).detach().cpu())
                del ra, rb, a_c, b_c, num, denom

            if adaptive_beta:
                beta, _ = model.compute_adaptive_beta(rule_logits, all_r)
                logits = beta * rule_logits + (1 - beta) * kge_score
            elif use_beta:
                beta = torch.sigmoid(model.beta[all_r[0]]).unsqueeze(-1)
                logits = beta * rule_logits + (1 - beta) * kge_score
            else:
                # Use fixed alpha
                logits = rule_logits + alpha * kge_score

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_mask.append(mask)

    # Concatenate all batches
    concat_logits = torch.cat(concat_logits, dim=0)
    concat_all_h = torch.cat(concat_all_h, dim=0)
    concat_all_r = torch.cat(concat_all_r, dim=0)
    concat_all_t = torch.cat(concat_all_t, dim=0)
    concat_flag = torch.cat(concat_flag, dim=0)
    concat_mask = torch.cat(concat_mask, dim=0)

    # Compute ranks (and per-query top-1 prediction when requested)
    ranks = []
    top1 = [] if return_per_query else None
    for k in range(concat_all_t.size(0)):
        t = concat_all_t[k]
        if concat_mask[k, t].item() == True:
            val = concat_logits[k, t]
            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
        else:
            L = 1
            H = concat_flag.size(1) + 1
        ranks.append((L + H - 1) / 2.0)

        if return_per_query:
            # argmax over flag-True candidates (the same set used for ranking).
            # Map the position-within-candidates back to a global entity id.
            cand_idx = concat_flag[k].nonzero(as_tuple=True)[0]
            cand_scores = concat_logits[k][cand_idx]
            top1_pos = cand_scores.argmax().item()
            top1.append(cand_idx[top1_pos].item())

    ranks = torch.tensor(ranks, dtype=torch.float)

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
            'densities':  torch.cat(concat_density, dim=0).cpu(),
            'relations':  concat_all_r.detach().cpu().long(),
            'top1':       torch.tensor(top1, dtype=torch.long),
            'true_tails': concat_all_t.detach().cpu().long(),
            'corr':       torch.cat(concat_corr, dim=0).cpu(),
        }
    return metrics


def train_beta_epoch(model, dataloader, optimizer, device, adaptive=False):
    """Train beta (and beta_density if adaptive) for one epoch."""
    model.train()

    # Explicit freezing tied to the training stage. This avoids the previous
    # 'beta in name' substring match, which was True for BOTH `beta` and
    # `beta_density` and so silently re-enabled grad on `model.beta` during
    # stage 6 (the adaptive stage). The values still didn't drift because the
    # optimizer only owned `beta_density`, but the grad was wastefully computed.
    for name, param in model.named_parameters():
        if name == 'beta':
            param.requires_grad = (not adaptive)
        elif name == 'beta_density':
            param.requires_grad = adaptive
        else:
            param.requires_grad = False
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training beta"):
        all_h, all_r, all_t, flag = batch
        all_h = all_h.squeeze(0).to(device)
        all_r = all_r.squeeze(0).to(device)
        all_t = all_t.squeeze(0).to(device)
        flag = flag.squeeze(0).to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            rule_logits, mask = model(all_h, all_r, None)
            kge_score = model.compute_g_KGE(all_h, all_r)
        
        if adaptive:
            beta, _ = model.compute_adaptive_beta(rule_logits, all_r)
        else:
            beta = torch.sigmoid(model.beta[all_r[0]]).unsqueeze(-1)
        logits = beta * rule_logits + (1 - beta) * kge_score
        
        # Compute loss: we want to maximize score for true tails
        # Use margin ranking loss: score(true) should be higher than score(false)
        true_scores = logits.gather(1, all_t.unsqueeze(1)).squeeze(1)  # [batch]
        
        # Compute loss for each example
        losses = []
        for i in range(all_h.size(0)):
            if mask[i, all_t[i]].item():
                # Get all valid negative samples
                negative_mask = flag[i] & (torch.arange(logits.size(1), device=device) != all_t[i])
                if negative_mask.sum() > 0:
                    negative_scores = logits[i][negative_mask]
                    # Sample some negatives to keep computation manageable
                    if negative_scores.size(0) > 100:
                        idx = torch.randperm(negative_scores.size(0))[:100]
                        negative_scores = negative_scores[idx]
                    
                    # Margin ranking loss: true_score should be > negative_score + margin
                    true_score = true_scores[i]
                    margin = 1.0
                    loss = torch.clamp(margin - true_score + negative_scores, min=0).mean()
                    losses.append(loss)
        
        # Aggregate all losses and backpropagate once
        if len(losses) > 0:
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


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
    one block of method rows, with MRR, Hit@1, Hit@10, MR as columns. This is
    wider than the previous MRR-only layout but lets the reader see *which*
    aspect of the ranking moved (e.g. MR can change while MRR is stationary).

    Args:
        method_to_ranks: dict[str, Tensor[N]] - per-query (filtered, tie-averaged)
                         ranks for each method. All tensors must share the same
                         length and per-index correspondence with `densities`.
        densities:       Tensor[N]            - per-query rule grounding density.
        num_buckets:     int                  - quantile buckets for nonzero densities.
                         A separate "zero" bucket is reported if any density ~= 0.
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
        # First row prints the bucket label + n; subsequent rows blank them
        # out so the visual block stays grouped.
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
        # Quantile boundaries can repeat when many densities are identical
        # (e.g. dense KGs); the resulting buckets just print as empty rows.
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

    Args:
        method_to_ranks:       dict[str, Tensor[N]] of per-query ranks.
        relations:             Tensor[N] of per-query relation ids.
        model:                 the RulE model (read-only; uses model.beta).
        densities:             Tensor[N] of per-query rule grounding density.
        fixed_method_name:     key in method_to_ranks for the Fixed-alpha baseline.
        adaptive_method_name:  key in method_to_ranks for the Adaptive method.
        top_n:                 optional limit on number of rows printed.
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

    Global block, for every pair, reports:
      flips                queries where top-1(A) != top-1(B).
      A_correct_only       flips where A predicted the true tail and B did not.
      B_correct_only       flips where B predicted the true tail and A did not.
      neither_correct      flips where both top-1 predictions are wrong.
      Hit@1 delta          (B_correct_only - A_correct_only) / N -- exactly the
                           Hit@1 delta from B over A contributed by these flips.

    Per-relation block (printed only when both `relations` and `model` are
    given): for each relation that has at least one flip, prints n, beta[r],
    flips, A_only, B_only, neither, and the *local* Hit@1 delta within that
    relation. Sorted by |local Hit@1 delta| descending so the relations where
    the choice of method matters most for accuracy come first.

    Args:
        method_to_top1: dict[str, Tensor[N]] of per-query top-1 entity ids.
        true_tails:     Tensor[N] of ground-truth tail ids.
        pairs:          list of (name_A, name_B). Report framed as "B over A".
        relations:      optional Tensor[N] of per-query relation ids; enables
                        the per-relation breakdown.
        model:          optional RulE model; if given alongside relations, the
                        per-relation breakdown will include sigmoid(model.beta[r])
                        so the table can be cross-referenced with the
                        per-relation MRR table.
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
                    # Skip relations with no method disagreement -- they
                    # contribute no information to this section.
                    continue
                rel_a_only = int((rel_flip & a_correct & ~b_correct).sum().item())
                rel_b_only = int((rel_flip & b_correct & ~a_correct).sum().item())
                rel_neither = int((rel_flip & ~a_correct & ~b_correct).sum().item())
                # Local Hit@1 delta is normalized by n_in_relation (NOT total N),
                # so it expresses "how much did B move accuracy on this relation".
                rel_dh1 = (rel_b_only - rel_a_only) / rel_n
                beta_r = (
                    beta_sigmoid[r].item()
                    if beta_sigmoid is not None and r < beta_sigmoid.numel()
                    else float('nan')
                )
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

    Interpretation: if the rule and KGE rank candidates near-identically (corr ~ 1),
    then *any* convex combination beta * rule + (1 - beta) * kge produces the
    same ranking, and no mixing strategy -- including adaptive beta -- can move
    the metrics. This number upper-bounds the headroom for routing.

    Args:
        corrs:       Tensor[N] of per-query Spearman correlations from evaluate().
        densities:   Tensor[N] of per-query densities (for bucket alignment).
        num_buckets: number of density quantile buckets.
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
        dataset=config.data_path
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
    # The valid set is split 50/50 per-relation: one half is used to TRAIN
    # beta (margin loss), the other half is used to SELECT the best epoch
    # (early stopping). This removes the bias of training and selecting on
    # the same data. The test set is untouched.
    batch_size = getattr(config, 'g_batch_size', 8)
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

    valid_train_set = FactListValidDataset(graph, batch_size, train_facts)
    valid_train_loader = DataLoader(valid_train_set, batch_size=1, num_workers=0)

    valid_select_set = FactListValidDataset(graph, batch_size, select_facts)
    valid_select_loader = DataLoader(valid_select_set, batch_size=1, num_workers=0)

    test_set = TestDataset(graph, batch_size)
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
    print(f"\n[1] Testing with FIXED ALPHA = {args.alpha} (baseline)...")
    metrics_fixed, pq_fixed = evaluate(
        model, test_dataloader, device, use_beta=False, alpha=args.alpha,
        return_per_query=True,
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
        # === Per-relation Beta before training ===
        print(f"\n[2] Testing with PER-RELATION BETA (before training)...")
        metrics_beta_before = evaluate(model, test_dataloader, device, use_beta=True)
        print_results("Per-relation Beta (untrained)", metrics_beta_before)
        print_beta_stats(model)

        # === Train per-relation Beta ===
        print(f"\n[3] Training per-relation beta for {args.epochs} epochs...")

        model.beta.requires_grad = True
        model.beta_density.requires_grad = False
        optimizer = torch.optim.Adam([model.beta], lr=args.lr)

        best_mrr = 0
        best_beta = model.beta.data.clone()
        best_epoch_3 = 0

        for epoch in range(args.epochs):
            loss = train_beta_epoch(model, valid_train_loader, optimizer, device, adaptive=False)
            model.eval()
            val_metrics = evaluate(model, valid_select_loader, device, use_beta=True)

            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Valid-select MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")

            if val_metrics['MRR'] > best_mrr:
                best_mrr = val_metrics['MRR']
                best_beta = model.beta.data.clone()
                best_epoch_3 = epoch + 1
                print(f"  New best MRR!")

        model.beta.data = best_beta

        # Save progress after stage 3. beta_density is still 0 here; it will
        # be overwritten by the post-stage-6 save below.
        save_beta_checkpoint(
            beta_ckpt_path, model,
            stage='stage3',
            seed=args.seed,
            best_mrr_stage3=best_mrr,
            epoch_best_stage3=best_epoch_3,
            base_checkpoint=os.path.abspath(args.checkpoint),
        )

    print(f"\n[4] Testing with TRAINED per-relation beta...")
    metrics_beta_after, pq_beta = evaluate(
        model, test_dataloader, device, use_beta=True, return_per_query=True,
    )
    print_results("Per-relation Beta (trained)", metrics_beta_after)
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
        metrics_adaptive_before = evaluate(model, test_dataloader, device, adaptive_beta=True)
        print_results("Adaptive Beta (untrained)", metrics_adaptive_before)

        # Train adaptive beta (beta_rel + beta_density)
        print(f"\n[6] Training adaptive beta for {args.epochs} epochs...")

        model.beta.requires_grad = False
        model.beta_density.requires_grad = True
        optimizer_adaptive = torch.optim.Adam([model.beta_density], lr=args.density_lr)

        best_mrr_adaptive = 0
        best_beta_adaptive = model.beta.data.clone()
        best_density = model.beta_density.data.clone()
        best_epoch_6 = 0

        for epoch in range(args.epochs):
            loss = train_beta_epoch(model, valid_train_loader, optimizer_adaptive, device, adaptive=True)
            model.eval()
            val_metrics = evaluate(model, valid_select_loader, device, adaptive_beta=True)

            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {loss:.4f}")
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
        )

    print(f"\n[7] Testing with TRAINED adaptive beta...")
    metrics_adaptive_after, pq_adaptive = evaluate(
        model, test_dataloader, device, adaptive_beta=True, return_per_query=True,
    )
    print_results("Adaptive Beta (trained)", metrics_adaptive_after)
    print_beta_stats(model)
    
    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<10} {'Fixed alpha':<12} {'Rel-Beta':<12} {'Adaptive':<12}")
    print("-" * 50)
    
    for metric in ['Hit@1', 'Hit@3', 'Hit@10', 'MR', 'MRR']:
        fixed = metrics_fixed[metric]
        rel = metrics_beta_after[metric]
        adaptive = metrics_adaptive_after[metric]
        print(f"{metric:<10} {fixed:<12.4f} {rel:<12.4f} {adaptive:<12.4f}")
    
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
    method_to_ranks = {
        fixed_label: pq_fixed['ranks'],
        "Rel-Beta":  pq_beta['ranks'],
        "Adaptive":  pq_adaptive['ranks'],
    }
    method_to_top1 = {
        fixed_label: pq_fixed['top1'],
        "Rel-Beta":  pq_beta['top1'],
        "Adaptive":  pq_adaptive['top1'],
    }

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
        pairs=[(fixed_label, "Adaptive"), ("Rel-Beta", "Adaptive")],
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
