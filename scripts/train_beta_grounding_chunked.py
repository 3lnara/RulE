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
    parser.add_argument('--feature_lr', '--density_lr', type=float, default=0.01,
                       dest='feature_lr',
                       help='LR for the beta_feature slope (stage 6); --density_lr is a deprecated alias.')
    parser.add_argument('--alpha', type=float, default=None,
                       help="Fixed-alpha baseline (rule + alpha*kge); default = config alpha.")
    parser.add_argument('--feature', type=str, default='density',
                       choices=['density', 'num_rules', 'kge_max',
                                'kge_entropy', 'top_rule_confidence'],
                       help="Per-query feature the adaptive head conditions on.")
    parser.add_argument('--normalize_scores', action='store_true',
                       help='Per-query divide rule_logits and kge_score by their std before mixing.')
    parser.add_argument('--standardize_feature', action='store_true',
                       help='Z-score the adaptive feature (stats from the beta-train split).')
    parser.add_argument('--no_per_relation', action='store_true',
                       help='Ablation: use one global intercept instead of per-relation beta[r].')
    parser.add_argument('--neg_sampling', type=str, default='mixed',
                       choices=['uniform', 'hard', 'mixed'],
                       help="Which negatives enter the margin loss (uniform / hard / mixed).")
    parser.add_argument('--num_negatives', type=int, default=100,
                       help='K negatives sampled per query for the margin loss.')
    parser.add_argument('--mixed_hard_frac', type=float, default=0.5,
                       help="Fraction of K taken as hardest under --neg_sampling mixed.")
    parser.add_argument('--seed', type=int, default=42,
                       help='RNG seed (torch / random / cuda) and per-relation valid split.')
    parser.add_argument('--beta_checkpoint', type=str, default=None,
                       help='Load/save path for the trained beta + beta_feature; default <ckpt_dir>/beta.pt.')
    parser.add_argument('--skip_beta_training', action='store_true',
                       help='Load beta from --beta_checkpoint and run only the eval / summary.')
    parser.add_argument('--beta_l2', type=float, default=0.0,
                       help='L2 on model.beta (adds beta_l2 * mean(beta**2) per batch).')
    parser.add_argument('--loss', type=str, default='margin',
                       choices=['margin', 'ce'],
                       help="Mixing-weight objective: margin (pairwise hinge) or ce (filtered softmax CE).")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help="Label smoothing for --loss ce (ignored under margin).")
    parser.add_argument('--skip_adaptive', action='store_true',
                       help='Train/eval the per-relation intercept only; skip the adaptive stage.')
    parser.add_argument('--freeze_intercept', action='store_true',
                       help='Leave the intercept at 0.5 and train only beta_feature (needs the adaptive stage).')
    return parser.parse_args()


def evaluate(model, dataloader, device, use_beta=False, adaptive_beta=False, alpha=3.0,
             normalize_scores=False):
    """Evaluate the model -> Hit@k / MR / MRR dict. Ranks are computed per batch and
    moved to CPU immediately to bound GPU memory on large graphs."""
    model.eval()
    all_ranks = []

    with torch.no_grad():
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)

            # Forward pass (populates model._last_ground_mask / _last_num_rules).
            rule_logits, mask = model(all_h, all_r, None)

            # Refresh the KGE stash before compute_adaptive_beta below, so the
            # kge_max / kge_entropy features are populated in time.
            kge_score = model.compute_g_KGE(all_h, all_r)
            model.update_kge_stash(kge_score, flag)

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

    ranks = torch.tensor(all_ranks, dtype=torch.float)
    metrics = {
        'Hit@1': (ranks <= 1).float().mean().item(),
        'Hit@3': (ranks <= 3).float().mean().item(),
        'Hit@10': (ranks <= 10).float().mean().item(),
        'MR': ranks.mean().item(),
        'MRR': (1.0 / ranks).mean().item()
    }

    return metrics


def select_negative_indices(neg_scores_detached, scheme, num_neg, hard_frac):
    """Pick the negatives entering the margin loss for one query (hard mining on
    the detached mixed logits); returns indices into the negative pool."""
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
                     num_negatives=100, mixed_hard_frac=0.5, beta_l2=0.0,
                     loss_type='margin', label_smoothing=0.0):
    """Train beta (and beta_feature if adaptive) for one epoch. loss_type='margin'
    is pairwise hinge over sampled negatives; 'ce' is filtered softmax
    cross-entropy over the full candidate set (neg_sampling args ignored).
    Scores/betas are applied per sample and each loss divided by N_valid so the
    accumulated grad is the batch mean (bounds memory on large graphs)."""
    model.train()

    # Explicit per-name freezing tied to the training stage.
    for name, param in model.named_parameters():
        if name == 'beta':
            param.requires_grad = (not adaptive) and model.use_per_relation
        elif name == 'global_beta':
            param.requires_grad = (not adaptive) and (not model.use_per_relation)
        elif name == 'beta_feature':
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
            # Fill the kge stash before compute_adaptive_beta (no-op for density/num_rules).
            model.update_kge_stash(kge_score, flag)
            # Per-query score normalization (same transform as evaluate).
            if normalize_scores:
                rule_logits = rule_logits / (rule_logits.std(dim=-1, keepdim=True) + 1e-6)
                kge_score   = kge_score   / (kge_score.std(dim=-1, keepdim=True) + 1e-6)

        # Pre-pass: samples contributing to the loss (true tail unmasked and
        # >=1 usable negative); N_valid scales each per-sample loss to a mean grad.
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
        # Stash the batch-wide feature tensors; each iteration exposes only row i
        # to compute_adaptive_beta, then restores them.
        saved_ground_mask = model._last_ground_mask
        saved_num_rules = model._last_num_rules
        saved_kge_max = model._last_kge_max
        saved_kge_entropy = model._last_kge_entropy
        saved_top_rule_conf = model._last_top_rule_conf

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
                model._last_kge_max = (
                    saved_kge_max[i:i+1] if saved_kge_max is not None else None
                )
                model._last_kge_entropy = (
                    saved_kge_entropy[i:i+1] if saved_kge_entropy is not None else None
                )
                model._last_top_rule_conf = (
                    saved_top_rule_conf[i:i+1] if saved_top_rule_conf is not None else None
                )
                beta_i, _ = model.compute_adaptive_beta(rl_i, r_i)
                model._last_ground_mask = saved_ground_mask
                model._last_num_rules = saved_num_rules
                model._last_kge_max = saved_kge_max
                model._last_kge_entropy = saved_kge_entropy
                model._last_top_rule_conf = saved_top_rule_conf
            elif model.use_per_relation:
                beta_i = torch.sigmoid(model.beta[r_i[0]]).unsqueeze(-1)
            else:
                beta_i = torch.sigmoid(model.global_beta).unsqueeze(-1)

            logit_i = beta_i * rl_i + (1 - beta_i) * kg_i   # [1, E], grad through beta only

            # Filtered negatives: every flag-True entity that is not the true
            # tail. `flag` is built from hr2ooo, so it already excludes all known
            # positives (incl. the held-out tail); the `!= all_t[i]` guard makes
            # the exclusion explicit and robust to the flag convention.
            negative_mask = flag[i] & (torch.arange(logit_i.size(1), device=device) != all_t[i])
            true_score = logit_i[0, all_t[i]]

            if loss_type == 'ce':
                # Filtered softmax cross-entropy over {true tail} + all filtered negatives.
                neg_scores = logit_i[0][negative_mask]
                all_scores = torch.cat([true_score.view(1), neg_scores])   # idx 0 = true tail
                logp = torch.log_softmax(all_scores, dim=-1)
                if label_smoothing > 0.0:
                    # (1-eps) on the true tail + eps uniform over all candidates.
                    sample_loss = -((1.0 - label_smoothing) * logp[0]
                                    + label_smoothing * logp.mean())
                else:
                    sample_loss = -logp[0]
                loss = sample_loss / N
            else:
                # Margin / hinge over a sampled subset of negatives. Rank by the
                # detached scores (hard-negative mining), gather the live scores
                # so grad still flows through beta.
                neg_all = logit_i[0][negative_mask]
                idx = select_negative_indices(
                    neg_all.detach(), neg_sampling, num_negatives, mixed_hard_frac,
                )
                neg_scores = neg_all[idx]
                # Mean over negatives / N so per-sample grads add to the batch mean.
                loss = torch.clamp(1.0 - true_score + neg_scores, min=0).mean() / N
            loss.backward()
            batch_loss_sum += loss.item()

        # L2 on the intercept (stage 3 only; no-op in stage 6). Batch-level term,
        # not divided by N.
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


def compute_feature_stats(model, dataloader, device):
    """Store mean/std of the adaptive feature (from one pass over the beta-train
    split) into model.feature_mean / model.feature_std for z-scoring."""
    model.eval()
    feats = []
    with torch.no_grad():
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)
            model(all_h, all_r, None)
            if model.feature_name in ('kge_max', 'kge_entropy'):
                kge_score = model.compute_g_KGE(all_h, all_r)
                model.update_kge_stash(kge_score, flag)
            feats.append(model.get_query_feature().detach().cpu())
    feats = torch.cat(feats, dim=0).float()
    mean = feats.mean()
    std = feats.std().clamp_min(1e-6)
    model.feature_mean.data = mean.view(1).to(model.feature_mean.device)
    model.feature_std.data = std.view(1).to(model.feature_std.device)
    print(f"Feature '{model.feature_name}' stats (beta-train split): "
          f"mean={mean.item():.4f}  std={std.item():.4f}  "
          f"(n={feats.numel()})")


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
        bd = model.beta_feature.item()
        fn = model.feature_name
        print(f"  Feature slope (beta_feature): {bd:.4f}  (feature='{fn}')")
        if bd > 0:
            print(f"    -> Positive: higher {fn} pushes beta UP (trust rules more)")
        elif bd < 0:
            print(f"    -> Negative: higher {fn} pushes beta DOWN (trust KGE more)")
        else:
            print(f"    -> Zero: {fn} has no effect on mixing")


def save_beta_checkpoint(path, model, **meta):
    """Save beta / beta_feature / global_beta (+ any meta kwargs) to a small .pt.
    Safe to call repeatedly to checkpoint partial progress."""
    state = {
        'beta':             model.beta.data.detach().cpu().clone(),
        'beta_feature':     model.beta_feature.data.detach().cpu().clone(),
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
    """Per-relation stratified split of a fact list -> (train_facts, select_facts);
    single-fact relations go to train."""
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
    """ValidDataset variant taking an explicit fact list (for the beta train/select halves)."""

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

    if args.freeze_intercept and args.skip_adaptive:
        raise ValueError(
            "--freeze_intercept and --skip_adaptive are mutually exclusive: the "
            "first skips intercept training and the second skips beta_feature "
            "training, so together NOTHING would be trained (pure fixed-0.5 mix). "
            "Drop one of them."
        )

    # Seed random / torch / torch.cuda for reproducible beta tensors.
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
    print(f"Loss type: {args.loss}" + (f" (label_smoothing={args.label_smoothing})" if args.loss == 'ce' else ""))
    if args.loss == 'ce':
        print("Negative sampling: N/A (CE uses the full filtered candidate set)")
    else:
        print(f"Negative sampling: {args.neg_sampling} (K={args.num_negatives}, hard_frac={args.mixed_hard_frac})")
    print(f"Adaptive beta stage: {'DISABLED (--skip_adaptive)' if args.skip_adaptive else 'enabled'}")
    print(f"Intercept stage [3]: {'FROZEN at 0.5 (--freeze_intercept)' if args.freeze_intercept else 'trained'}")
    
    # Load config
    config = load_config(args.config)
    if isinstance(config, (list, tuple)):
        config = config[0]

    # Fixed-alpha baseline: CLI override, else the config's per-dataset alpha.
    if args.alpha is None:
        args.alpha = float(getattr(config, 'alpha', 3.0))
        print(f"Fixed-alpha baseline: rule + {args.alpha:g}*kge  (from config)")
    else:
        _cfg_alpha = getattr(config, 'alpha', None)
        note = ''
        if _cfg_alpha is not None and abs(float(_cfg_alpha) - args.alpha) > 1e-9:
            note = f"  [OVERRIDES config alpha={float(_cfg_alpha):g}]"
        print(f"Fixed-alpha baseline: rule + {args.alpha:g}*kge  (from --alpha){note}")
    
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
    # Precompute per-rule plausibility so forward() fills _last_top_rule_conf.
    if hasattr(model, 'eval_compute_rule_confidence'):
        model.eval_compute_rule_confidence(device)

    # Small batches: compute_g_KGE materializes a [batch, num_entities, 2*hidden]
    # tail tensor (WN18RR at g_batch_size=32 would OOM). Valid set is split 50/50
    # per-relation: one half trains beta, the other selects the best epoch.
    beta_batch_size = 4
    train_facts, select_facts = split_facts_per_relation(
        graph.valid_facts, train_ratio=0.5, seed=args.seed
    )
    print(f"Valid split for beta training: "
          f"{len(train_facts)} train / {len(select_facts)} select "
          f"(from {len(graph.valid_facts)} total valid facts)")

    # Beta-checkpoint path; defaults to 'beta.pt' next to the base checkpoint.
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
    print(f"\n[1] Testing with FIXED ALPHA = {args.alpha} (baseline)...")
    metrics_fixed = evaluate(
        model, test_dataloader, device, use_beta=False, alpha=args.alpha,
        normalize_scores=args.normalize_scores,
    )
    print_results(f"Fixed Alpha (alpha={args.alpha})", metrics_fixed)
    
    # Default values used by the metadata save when training is skipped.
    best_mrr = float('nan')
    best_epoch_3 = -1

    if args.skip_beta_training:
        # Load already-trained beta + beta_feature and skip stages [2], [3].
        beta_state = torch.load(beta_ckpt_path, map_location=device)
        model.beta.data.copy_(beta_state['beta'].to(device))
        # Checkpoints written before the beta_density -> beta_feature rename
        # store the slope under the old key; accept both.
        _slope = beta_state.get('beta_feature', beta_state.get('beta_density'))
        if _slope is None:
            raise KeyError(
                f"{beta_ckpt_path} has neither 'beta_feature' nor the legacy "
                "'beta_density' key -- it is not a beta checkpoint.")
        model.beta_feature.data.copy_(_slope.to(device))
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

        # === Train intercept (per-relation or global), unless frozen ===
        if args.freeze_intercept:
            # Leave beta / global_beta at their zero init (sigmoid(0)=0.5,
            # neutral routing). best_mrr / best_epoch_3 keep the NaN / -1
            # defaults set above. Only beta_feature is trained in stage [6].
            print(f"\n[3] SKIPPED -- --freeze_intercept: {_stage3_label.lower()} "
                  f"left at init (logit 0 => sigmoid 0.5, neutral routing). "
                  f"Only the adaptive beta_feature slope is trained (stage [6]).")
        else:
            print(f"\n[3] Training {_stage3_label.lower()} for {args.epochs} epochs...")

            if model.use_per_relation:
                model.beta.requires_grad = True
                model.global_beta.requires_grad = False
                _intercept_param = model.beta
            else:
                model.beta.requires_grad = False
                model.global_beta.requires_grad = True
                _intercept_param = model.global_beta
            model.beta_feature.requires_grad = False
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
                                        beta_l2=args.beta_l2,
                                        loss_type=args.loss,
                                        label_smoothing=args.label_smoothing)
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

        # Save progress after stage 3. beta_feature is still 0 here; it will
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
    _stage4_state = "FROZEN 0.5" if args.freeze_intercept else "TRAINED"
    print(f"\n[4] Testing with {_stage4_state} {_stage3_label.lower()}...")
    metrics_beta_after = evaluate(
        model, test_dataloader, device, use_beta=True,
        normalize_scores=args.normalize_scores,
    )
    print_results(f"{_stage3_label} ({'frozen 0.5' if args.freeze_intercept else 'trained'})", metrics_beta_after)
    print_beta_stats(model)
    
    # === Adaptive Beta (optional; disabled by --skip_adaptive) ===
    best_mrr_adaptive = float('nan')
    best_epoch_6 = -1

    if args.skip_adaptive:
        # Stages [5]-[7] disabled: beta_feature stays 0, so adaptive beta == the
        # per-relation beta. Alias those metrics; the Adaptive column is dropped.
        print(f"\n[5/6/7] SKIPPED -- adaptive beta disabled (--skip_adaptive).")
        print(f"        beta_feature stays {model.beta_feature.item():.4f}; "
              f"adaptive == per-relation beta, so its column is omitted.")
        metrics_adaptive_after = metrics_beta_after
    else:
        if args.skip_beta_training:
            # beta and beta_feature were already loaded above; just skip [5] and [6].
            print(f"\n[5/6] SKIPPED -- using loaded beta_feature")
            print(f"      best_mrr (6):   {beta_state.get('best_mrr_stage6', 'unknown')}")
            print(f"      epoch_best (6): {beta_state.get('epoch_best_stage6', 'unknown')}")
            best_mrr_adaptive = beta_state.get('best_mrr_stage6', best_mrr_adaptive)
            best_epoch_6 = beta_state.get('epoch_best_stage6', best_epoch_6)
        else:
            print(f"\n[5] Testing with ADAPTIVE BETA (before training)...")
            metrics_adaptive_before = evaluate(model, test_dataloader, device, adaptive_beta=True,
                                               normalize_scores=args.normalize_scores)
            print_results("Adaptive Beta (untrained)", metrics_adaptive_before)

            # Train adaptive beta (beta_rel + beta_feature)
            print(f"\n[6] Training adaptive beta for {args.epochs} epochs...")

            model.beta.requires_grad = False
            model.global_beta.requires_grad = False   # frozen from stage 3
            model.beta_feature.requires_grad = True
            optimizer_adaptive = torch.optim.Adam(
                [model.beta_feature], lr=args.feature_lr
            )

            best_mrr_adaptive = 0
            best_beta_adaptive = model.beta.data.clone()
            best_global_adaptive = model.global_beta.data.clone()
            best_density = model.beta_feature.data.clone()
            best_epoch_6 = 0

            for epoch in range(args.epochs):
                loss = train_beta_epoch(model, valid_train_loader, optimizer_adaptive, device, adaptive=True,
                                        normalize_scores=args.normalize_scores,
                                        neg_sampling=args.neg_sampling,
                                        num_negatives=args.num_negatives,
                                        mixed_hard_frac=args.mixed_hard_frac,
                                        beta_l2=args.beta_l2,
                                        loss_type=args.loss,
                                        label_smoothing=args.label_smoothing)
                model.eval()
                val_metrics = evaluate(model, valid_select_loader, device, adaptive_beta=True,
                                       normalize_scores=args.normalize_scores)

                print(f"\nEpoch {epoch+1}/{args.epochs}:")
                print(f"  Loss: {loss:.4f}")
                print(f"  Valid-select MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")
                print(f"  beta_feature = {model.beta_feature.item():.4f}")
                if not model.use_per_relation:
                    print(f"  global_beta  = {model.global_beta.item():.4f}  "
                          f"(prob: {torch.sigmoid(model.global_beta).item():.4f})")

                if val_metrics['MRR'] > best_mrr_adaptive:
                    best_mrr_adaptive = val_metrics['MRR']
                    best_beta_adaptive = model.beta.data.clone()
                    best_global_adaptive = model.global_beta.data.clone()
                    best_density = model.beta_feature.data.clone()
                    best_epoch_6 = epoch + 1
                    print(f"  New best MRR!")

            model.beta.data = best_beta_adaptive
            model.global_beta.data = best_global_adaptive
            model.beta_feature.data = best_density

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
        metrics_adaptive_after = evaluate(
            model, test_dataloader, device, adaptive_beta=True,
            normalize_scores=args.normalize_scores,
        )
        print_results("Adaptive Beta (trained)", metrics_adaptive_after)
        print_beta_stats(model)

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    _rl = "Rel-Beta" if model.use_per_relation else "Global-Beta"
    if args.skip_adaptive:
        print(f"{'Metric':<10} {'Fixed alpha':<14} {_rl:<14}")
        print("-" * 40)
        for metric in ['Hit@1', 'Hit@3', 'Hit@10', 'MR', 'MRR']:
            fixed = metrics_fixed[metric]
            rel = metrics_beta_after[metric]
            print(f"{metric:<10} {fixed:<14.4f} {rel:<14.4f}")
        print(f"\nAdaptive beta disabled (--skip_adaptive); beta_feature not trained.")
    else:
        print(f"{'Metric':<10} {'Fixed alpha':<14} {_rl:<14} {'Adaptive':<12}")
        print("-" * 54)
        for metric in ['Hit@1', 'Hit@3', 'Hit@10', 'MR', 'MRR']:
            fixed = metrics_fixed[metric]
            rel = metrics_beta_after[metric]
            adaptive = metrics_adaptive_after[metric]
            print(f"{metric:<10} {fixed:<14.4f} {rel:<14.4f} {adaptive:<12.4f}")

        print(f"\nbeta_feature learned = {model.beta_feature.item():.4f}")
        fn = model.feature_name
        if model.beta_feature.item() > 0:
            print(f"Interpretation: higher {fn} -> trust rules more (as expected)")
        elif model.beta_feature.item() < 0:
            print(f"Interpretation: higher {fn} -> trust KGE more (counterintuitive)")
        else:
            print(f"Interpretation: {fn} has no effect on mixing")


if __name__ == '__main__':
    main()
