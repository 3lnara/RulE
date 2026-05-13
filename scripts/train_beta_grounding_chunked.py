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
    return parser.parse_args()


def evaluate(model, dataloader, device, use_beta=False, adaptive_beta=False, alpha=3.0):
    """Evaluate the model.

    Ranks are computed per-batch and moved to CPU immediately so the GPU never
    holds more than one batch of [batch, num_entities] logits at a time.  This
    is critical for large datasets such as WN18RR (40 943 entities).
    """
    model.eval()
    all_ranks = []

    with torch.no_grad():
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)

            # Forward pass
            logits, mask = model(all_h, all_r, None)

            # Combine with KGE
            kge_score = model.compute_g_KGE(all_h, all_r)

            if adaptive_beta:
                beta, _ = model.compute_adaptive_beta(logits, all_r)
                logits = beta * logits + (1 - beta) * kge_score
            elif use_beta:
                beta = torch.sigmoid(model.beta[all_r[0]]).unsqueeze(-1)
                logits = beta * logits + (1 - beta) * kge_score
            else:
                logits = logits + alpha * kge_score

            # Move to CPU immediately — avoids accumulating GB of GPU tensors
            logits_cpu = logits.cpu()
            flag_cpu = flag.cpu()
            mask_cpu = mask.cpu()
            all_t_cpu = all_t.cpu()
            del logits, kge_score
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


def train_beta_epoch(model, dataloader, optimizer, device, adaptive=False):
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
            param.requires_grad = (not adaptive)
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
        for i in valid_indices:
            r_i = all_r[i:i+1]
            rl_i = rule_logits[i:i+1]   # [1, E] — detached from frozen no_grad graph
            kg_i = kge_score[i:i+1]     # [1, E]

            if adaptive:
                beta_i, _ = model.compute_adaptive_beta(rl_i, r_i)
            else:
                beta_i = torch.sigmoid(model.beta[r_i[0]]).unsqueeze(-1)

            logit_i = beta_i * rl_i + (1 - beta_i) * kg_i   # [1, E], grad through beta only

            negative_mask = flag[i] & (torch.arange(logit_i.size(1), device=device) != all_t[i])
            neg_scores = logit_i[0][negative_mask]
            if neg_scores.size(0) > 100:
                idx = torch.randperm(neg_scores.size(0), device=device)[:100]
                neg_scores = neg_scores[idx]

            true_score = logit_i[0, all_t[i]]
            # Per-sample mean over negatives, then divide by N so the per-sample
            # gradients add up to the batch mean (equivalent to stacking and
            # calling .mean().backward() in the non-chunked version, but with
            # per-sample memory).
            loss = torch.clamp(1.0 - true_score + neg_scores, min=0).mean() / N
            loss.backward()
            batch_loss_sum += loss.item()

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
        graph.valid_facts, train_ratio=0.5, seed=42
    )
    print(f"Valid split for beta training: "
          f"{len(train_facts)} train / {len(select_facts)} select "
          f"(from {len(graph.valid_facts)} total valid facts)")

    valid_train_set = FactListValidDataset(graph, beta_batch_size, train_facts)
    valid_train_loader = DataLoader(valid_train_set, batch_size=1, num_workers=0)

    valid_select_set = FactListValidDataset(graph, beta_batch_size, select_facts)
    valid_select_loader = DataLoader(valid_select_set, batch_size=1, num_workers=0)

    test_set = TestDataset(graph, beta_batch_size)
    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0)
    
    print(f"\n{'='*60}")
    print("BETA TRAINING DURING GROUNDING PHASE")
    print(f"{'='*60}")
    
    # === Baseline: Fixed Alpha ===
    print(f"\n[1] Testing with FIXED ALPHA = {args.alpha} (baseline)...")
    metrics_fixed = evaluate(model, test_dataloader, device, use_beta=False, alpha=args.alpha)
    print_results(f"Fixed Alpha (alpha={args.alpha})", metrics_fixed)
    
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
            print(f"  New best MRR!")

    model.beta.data = best_beta
    
    print(f"\n[4] Testing with TRAINED per-relation beta...")
    metrics_beta_after = evaluate(model, test_dataloader, device, use_beta=True)
    print_results("Per-relation Beta (trained)", metrics_beta_after)
    print_beta_stats(model)
    
    # === Adaptive Beta ===
    print(f"\n[5] Testing with ADAPTIVE BETA (before training)...")
    metrics_adaptive_before = evaluate(model, test_dataloader, device, adaptive_beta=True)
    print_results("Adaptive Beta (untrained)", metrics_adaptive_before)
    
    # Train adaptive beta (beta_rel + beta_density)
    print(f"\n[6] Training adaptive beta for {args.epochs} epochs...")
    
    model.beta.requires_grad = False
    model.beta_density.requires_grad = True
    optimizer_adaptive = torch.optim.Adam(
        [model.beta_density], lr=args.density_lr
    )
    
    best_mrr_adaptive = 0
    best_beta_adaptive = model.beta.data.clone()
    best_density = model.beta_density.data.clone()
    
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
            print(f"  New best MRR!")
    
    model.beta.data = best_beta_adaptive
    model.beta_density.data = best_density
    
    print(f"\n[7] Testing with TRAINED adaptive beta...")
    metrics_adaptive_after = evaluate(model, test_dataloader, device, adaptive_beta=True)
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


if __name__ == '__main__':
    main()
