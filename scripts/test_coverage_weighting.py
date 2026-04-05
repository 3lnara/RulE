"""
Test script to compare Fixed Alpha vs Coverage-based Weighting (Option 3).
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import KnowledgeGraph, TestDataset, RuleDataset
from model import RulE
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--min_alpha', type=float, default=0.5, 
                       help='Minimum alpha for coverage weighting (high coverage)')
    parser.add_argument('--max_alpha', type=float, default=5.0,
                       help='Maximum alpha for coverage weighting (low coverage)')
    return parser.parse_args()


def evaluate(model, dataloader, device, use_coverage=False, alpha=3.0):
    """
    Evaluate the model with either fixed alpha or coverage-based weighting.
    """
    model.eval()
    model.use_coverage_weighting = use_coverage
    
    concat_logits = []
    concat_all_h = []
    concat_all_r = []
    concat_all_t = []
    concat_flag = []
    concat_mask = []
    
    with torch.no_grad():
        desc = "Coverage-based" if use_coverage else f"Fixed α={alpha}"
        for batch in tqdm(dataloader, desc=f"Evaluating ({desc})"):
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)
            
            # Forward pass
            logits, mask = model(all_h, all_r, None)
            
            # If not using coverage weighting, add fixed alpha * KGE
            if not use_coverage:
                kge_score = model.compute_g_KGE(all_h, all_r)
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
    
    # Compute ranks (same logic as trainer.evaluate_t)
    ranks = []
    for k in range(concat_all_t.size(0)):
        h = concat_all_h[k]
        r = concat_all_r[k]
        t = concat_all_t[k]
        if concat_mask[k, t].item() == True:
            val = concat_logits[k, t]
            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
        else:
            L = 1
            H = concat_flag.size(1) + 1
        ranks.append((L + H - 1) / 2.0)  # Expected rank
    
    ranks = torch.tensor(ranks, dtype=torch.float)
    
    metrics = {
        'Hit@1': (ranks <= 1).float().mean().item(),
        'Hit@3': (ranks <= 3).float().mean().item(),
        'Hit@10': (ranks <= 10).float().mean().item(),
        'MR': ranks.mean().item(),
        'MRR': (1.0 / ranks).mean().item()
    }
    
    return metrics


def print_results(name, metrics):
    print(f"\n{'='*60}")
    print(f"Results for: {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


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
    print(f"Data path: {data_path}")
    
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
    
    # Set coverage weighting parameters
    model.min_alpha = args.min_alpha
    model.max_alpha = args.max_alpha
    
    # Load rules
    rule_file = config.rule_file
    if not os.path.isabs(rule_file):
        rule_file = os.path.join(os.path.dirname(args.config), rule_file)
    rule_file = os.path.normpath(rule_file)
    print(f"Loading rules from: {rule_file}")
    
    rule_negative_size = getattr(config, 'rule_negative_size', 32)
    ruleset = RuleDataset(graph.relation_size, rule_file, rule_negative_size)
    rules = [rule[0] for rule in ruleset.rules]
    
    model.set_rules(rules)
    print(f"Loaded {len(rules)} rules")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print("Checkpoint loaded")
    
    model = model.to(device)
    model.eval()
    
    # Precompute rule weights for grounding
    model.eval_compute_rule_weight(device)
    
    # Load test set
    batch_size = getattr(config, 'g_batch_size', 8)
    test_set = TestDataset(graph, batch_size)
    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0)
    
    print(f"\n{'='*60}")
    print("COMPARISON: Fixed Alpha vs Coverage-based Weighting")
    print(f"{'='*60}")
    
    # === Method 1: Fixed Alpha (Original) ===
    print(f"\n[1] Testing with FIXED ALPHA = {args.alpha}...")
    metrics_fixed = evaluate(model, test_dataloader, device, 
                            use_coverage=False, alpha=args.alpha)
    print_results(f"Fixed Alpha (α={args.alpha})", metrics_fixed)
    
    # === Method 2: Coverage-based Weighting (Option 3) ===
    print(f"\n[2] Testing with COVERAGE-BASED WEIGHTING...")
    print(f"    (α ranges from {args.min_alpha} to {args.max_alpha} based on rule coverage)")
    metrics_coverage = evaluate(model, test_dataloader, device, 
                                use_coverage=True)
    print_results(f"Coverage-based (α ∈ [{args.min_alpha}, {args.max_alpha}])", metrics_coverage)
    
    # === Summary Comparison ===
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<10} {'Fixed Alpha':<15} {'Coverage-based':<15} {'Difference':<15}")
    print("-" * 60)
    for metric in ['Hit@1', 'Hit@3', 'Hit@10', 'MR', 'MRR']:
        fixed_val = metrics_fixed[metric]
        cov_val = metrics_coverage[metric]
        if metric == 'MR':
            diff = fixed_val - cov_val  # Lower is better
            diff_str = f"{diff:+.2f} ({'better' if diff > 0 else 'worse'})"
        else:
            diff = cov_val - fixed_val  # Higher is better
            diff_str = f"{diff:+.4f} ({'better' if diff > 0 else 'worse'})"
        print(f"{metric:<10} {fixed_val:<15.4f} {cov_val:<15.4f} {diff_str}")
    
    print(f"\nNote: Coverage-based weighting adapts KGE weight based on rule confidence.")
    print(f"When rules have high coverage, α → {args.min_alpha} (rely on rules)")
    print(f"When rules have low coverage, α → {args.max_alpha} (rely on KGE)")


if __name__ == '__main__':
    main()
