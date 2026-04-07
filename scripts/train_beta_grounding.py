"""
Train beta parameter during grounding phase (after main model is trained).
This allows learning optimal KGE/Rule score balance per relation without retraining the full model.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    parser.add_argument('--alpha', type=float, default=3.0, 
                       help='Fixed alpha for comparison baseline')
    return parser.parse_args()


def evaluate(model, dataloader, device, use_beta=False, adaptive_beta=False, alpha=3.0):
    """Evaluate the model."""
    model.eval()
    
    concat_logits = []
    concat_all_h = []
    concat_all_r = []
    concat_all_t = []
    concat_flag = []
    concat_mask = []
    
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
                # Use fixed alpha
                logits = logits + alpha * kge_score
            
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
    
    # Compute ranks
    ranks = []
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
    
    ranks = torch.tensor(ranks, dtype=torch.float)
    
    metrics = {
        'Hit@1': (ranks <= 1).float().mean().item(),
        'Hit@3': (ranks <= 3).float().mean().item(),
        'Hit@10': (ranks <= 10).float().mean().item(),
        'MR': ranks.mean().item(),
        'MRR': (1.0 / ranks).mean().item()
    }
    
    return metrics


def train_beta_epoch(model, dataloader, optimizer, device, adaptive=False):
    """Train beta (and beta_density if adaptive) for one epoch."""
    model.train()
    
    # Freeze all parameters except beta-related ones
    for name, param in model.named_parameters():
        if 'beta' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
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
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval_compute_rule_weight(device)
    
    # Create dataloaders
    batch_size = getattr(config, 'g_batch_size', 8)
    valid_set = ValidDataset(graph, batch_size)
    valid_dataloader = DataLoader(valid_set, batch_size=1, num_workers=0)
    
    test_set = TestDataset(graph, batch_size)
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
        loss = train_beta_epoch(model, valid_dataloader, optimizer, device, adaptive=False)
        model.eval()
        val_metrics = evaluate(model, valid_dataloader, device, use_beta=True)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Valid MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")
        
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
    
    model.beta.requires_grad = True
    model.beta_density.requires_grad = True
    optimizer_adaptive = torch.optim.Adam(
        [model.beta, model.beta_density], lr=args.lr
    )
    
    best_mrr_adaptive = 0
    best_beta_adaptive = model.beta.data.clone()
    best_density = model.beta_density.data.clone()
    
    for epoch in range(args.epochs):
        loss = train_beta_epoch(model, valid_dataloader, optimizer_adaptive, device, adaptive=True)
        model.eval()
        val_metrics = evaluate(model, valid_dataloader, device, adaptive_beta=True)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Valid MRR: {val_metrics['MRR']:.4f}, Hit@10: {val_metrics['Hit@10']:.4f}")
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
